# Step4：按请求计算 region 任务的 summary 更新方案

> 本文保留任务划分阶段的历史设计说明。其中的两阶段 update 已被替换；当前实现采用 slot 准备加一次 update，请以[当前流程文档](step4_prefill_region_tasks.md)为准。

本文面向第一次阅读这段代码的读者，解释当前代码在做什么，以及在同一 region 的 token 连续时，如何简化整个流程。

当前实现采用本地 vLLM NVIDIA 路径的任务划分思路：根据每个请求的起始位置和本次长度，直接计算 region 任务。它替代了此前方案一的逐行首行检测，但保留 Ascend 的两阶段更新和中间状态管理。NPU 是执行计算的昇腾加速设备；目前没有完成设备编译、精度或性能验证。

本文中的 kernel 指提交给设备执行的计算程序，Triton 是编写这些程序所使用的工具。

## 1. 先理解这段代码处理的数据

### 1.1 token、请求、批次和 prefill

模型把输入文本转换成一串编号，每个编号代表一个 token。一个 token 不一定对应一个汉字或一个完整单词。

一次用户输入及其生成过程可以称为一个请求。系统可以把多个请求的一部分 token 放在一起计算，这一次送入计算的数据称为一个批次。

prefill 指处理输入文本 token 的阶段。输入很长时，可以拆成多次计算，每次处理的一段称为一个 chunk。decode 指逐步生成后续 token 的阶段。一个批次同时包含这两种阶段的数据时，称为混合批次。

本文讨论的是 prefill 或混合批次的 summary 更新，不修改独立的 decode 更新路径。

### 1.2 region 和 summary

这里把固定数量的位置划成一个 region，可以理解为“需要一起汇总的一小段 token”。一个 region 有多少个位置，由 `REGION_BLOCK_SIZE` 指定。下面统一用 4 举例，这只是示例值。

每个 token 都有一组用于汇总的数值。summary 是把一个 region 中的这些数值按权重合并后得到的结果。可以把它理解为这个 region 的压缩表示；它不是一段自然语言摘要。

region 有一个实际编号，例如 10 或 23。这个编号用于定位缓存中的数据，不是 token 的文本编号，也不一定按请求中的顺序连续递增。

缓存就是保存数据、供后续计算继续使用的内存空间。本文涉及最终 summary 缓存，以及未完成 region 的中间状态缓存。

### 1.3 行号、位置和 region 编号不是同一个东西

数组中的编号从 0 开始。每个 token 在本批次输入中占一行，这个下标称为行号 `row`。

`token_positions[row]` 则是这个 token 在**所属请求序列**中的位置。不同请求的位置可以重复。

`flat_slot[row]` 在此处表示这个 token 对应的实际 region 编号。虽然变量名里有 `slot`，本段代码使用它来识别 region。

下面是一个有 6 行的输入，所有行都有效：

| 输入行号 row | flat_slot：实际 region 编号 | token_positions：序列位置 | region 内的位置 |
| --- | --- | --- | --- |
| 0 | 10 | 0 | 0 |
| 1 | 10 | 1 | 1 |
| 2 | 10 | 2 | 2 |
| 3 | 10 | 3 | 3 |
| 4 | 23 | 8 | 0 |
| 5 | 23 | 9 | 1 |

region 内的位置通过 `token_positions[row] % REGION_BLOCK_SIZE` 计算。`%` 表示取除法的余数，例如 `9 % 4 = 1`。

region 23 的编号不需要等于 `8 // 4`；实际编号还涉及缓存的组织方式。这里直接使用输入提供的 `flat_slot`，不从序列位置重新猜测实际 region 编号。

### 1.4 其他输入是什么

代码中的 Tensor（张量）可以先理解为存放数字的数组或多维表格。

| 名称 | 在本文中的含义 |
| --- | --- |
| `live_rows` | 本次允许处理的输入行数，不能读取超出这个范围的行 |
| `token_valid[row]` | 这一行是否真的参与计算；真表示参与，假表示忽略 |
| `index_k[row, :]` | 这一行 token 用于汇总的一组数值，冒号表示取这一行的所有列 |
| `index_z[row, :]` | 用来计算这组数值权重的分数；后文解释如何转成权重 |
| `reset_slots[row]` | 与本行实际 region 编号相等时，表示本次需要重置这个 region 的旧累计状态 |

为方便设备计算，输入数组可能预留比实际 token 更多的行，称为填充行。`live_rows` 限制处理范围，`token_valid` 再决定范围内哪些行有效，两者都需要检查。

## 2. 当前流程与此前方案的区别

```text
请求的输入行号边界 + 请求总长度
    ↓
任务准备 kernel：生成每个请求的累计任务数 task_prefix
    ↓
第一次 update：只处理完成的 region，释放中间状态空间
    ↓
第二次 update：只处理未完成的 region，保存累计状态
```

这里的“任务”是一次 region 更新。每个任务负责一个请求本次触及的一个逻辑 region。“逻辑”表示按序列位置划分；写缓存时仍要从 `flat_slot` 获取实际 region 编号。

| 环节 | 最初的 unique 方案 | 此前方案一 | 当前实现 |
| --- | --- | --- | --- |
| 找到 region 输入 | 去重后建立全局行号表 | 逐行比较前后 region 编号 | 根据请求位置和长度计算 |
| 临时数据 | 三份 scratch 表及反向索引 | 无全局分组表 | 一个请求级 task_prefix 数组 |
| update 的遍历对象 | 去重后的分组 | 全部输入行，非首行跳过 | region 任务 |
| 更新阶段 | 完成、未完成各一次 | 完成、未完成各一次 | 完成、未完成各一次 |

当前仍不使用 `torch.unique`，也不恢复 clear、fill 或原来的三份 scratch 表。新增的 `task_prefix` 按请求数分配，长度为请求数加一，不按 token 数乘 region 大小分配。

## 3. 需要哪些请求信息

`query_start_loc` 保存每个请求在本批次的起止行号。`seq_lens` 保存每个请求处理完本次输入后的总长度。这些用于描述输入的辅助数组统称为元数据。

例如 region 大小为 8：

| 请求 | 之前已有 token 数 | 本次 token 数 | 本批次输入行 |
| --- | --- | --- | --- |
| A | 6 | 12 | 0～11 |
| B | 0 | 5 | 12～16 |

```python
query_start_loc = [0, 12, 17]
seq_lens = [18, 5]
```

对于请求 A：

```text
本次长度 = 12 - 0 = 12
之前长度 = 18 - 12 = 6
region 内起始位置 = 6 % 8 = 6
```

因此它本次覆盖三个 region：

```text
序列位置：6 7 | 8 9 10 11 12 13 14 15 | 16 17
任务：     0  |           1            |   2
```

## 4. 任务准备 kernel：生成 task_prefix

新增的 `_csa_prefill_task_prefix_kernel` 对每个请求计算：

```python
query_length = query_end - query_start
context_length = sequence_length - query_length
context_offset = context_length % region_size
count = (context_offset + query_length + region_size - 1) // region_size
```

`//` 表示整数除法，上面的写法实现向上取整。本次长度为 0 的请求，其任务数设为 0。

本例的任务数为 `[3, 1]`。把数量依次累加，得到：

```python
task_prefix = [0, 3, 4]
```

这个“累计和”表示每个请求的任务编号边界：A 负责 0、1、2，B 负责 3，最后一项 4 是总任务数。如果中间有空请求，边界可以重复，例如 `[0, 3, 3, 4]`。

kernel 使用 `tl.cumsum` 在设备上计算累计和，每次调用都刷新已有缓冲区的内容。缓冲区是预先分配并重复使用的内存，不能因为它已经存在就复用上一批次的任务数量。

## 5. update kernel：由任务编号定位 token

### 5.1 分配任务给执行实例

一次 kernel 调用启动固定数量的执行实例，每个实例运行相同代码，处理不同任务。`pid` 是实例编号，`NUM_CORES` 是实例数量，取设备的向量计算核数。

```python
num_tasks = task_prefix[num_requests]
for task in range(pid, num_tasks, NUM_CORES):
    ...
```

例如两个实例处理四个任务，实例 0 处理任务 0、2，实例 1 处理任务 1、3。任务总数直接在设备上读取，无需先交给提交任务的 CPU。

### 5.2 找到任务所属请求

每个任务查找满足以下关系的请求：

```text
 task_prefix[request] <= task < task_prefix[request + 1]
```

代码使用二分查找：每次检查候选范围的中间位置，排除一半范围，直到找到所属请求。累计边界重复的空请求不会得到任务。

然后计算请求内任务编号：

```python
ordinal = task - task_prefix[request]
```

### 5.3 直接计算起始行

请求内第一个任务从本次请求的第一行开始。后面的任务先跳过第一个 region 的剩余位置，再按完整 region 大小前进：

```python
if ordinal == 0:
    start_row = query_start
else:
    start_row = (
        query_start
        + region_size - context_offset
        + (ordinal - 1) * region_size
    )
```

本例得到：

| 任务编号 | 请求 | 起始行 | 本次负责的行 |
| --- | --- | --- | --- |
| 0 | A | 0 | 0～1 |
| 1 | A | 2 | 2～9 |
| 2 | A | 10 | 10～11 |
| 3 | B | 12 | 12～16 |

这张表只为解释展示，代码不会把它完整存下来。每个任务现场计算自己的行号。

### 5.4 读取实际 region 和有效 token

从 `flat_slot[start_row]` 获取实际 region 编号，检查它是否在合法缓存范围内，再生成从起始行向后的读取窗口。

第一个任务的窗口最多覆盖 `region_size - context_offset` 行，后续任务最多覆盖 `region_size` 行。窗口同时受请求结束行和 `live_rows` 限制，不能跨到另一个请求。

每行还要满足实际 region 编号一致且 `token_valid=True`。代码使用 mask，即表示哪些位置允许读取和参与计算的真、假条件数组，来屏蔽不符合条件的位置。

当前实现保留首行无效时仍检查后续行的行为，但要求首行的 `flat_slot` 仍提供正确的实际 region 编号。如果首行编号非法，整个任务会跳过；不会搜索另一个编号替代它。

## 6. 完成判断、reset 和累计计算

任务中有效 token 包含 `position % region_size == region_size - 1` 的位置，就说明该 region 本次完成，记为 `complete=True`。之前的位置可以已经在旧累计状态中，不要求它们全在本次输入中。

第一轮只执行完成任务，第二轮只执行未完成任务。两轮复用同一份 `task_prefix`，都在同一设备执行顺序下提交。

未完成的 region 使用一个 active slot 保存中间状态。slot 可以理解为有限数量的存储格子：

| 数据 | 含义 |
| --- | --- |
| `active_slot_by_region` | region 到 slot 的位置提示 |
| `active_region_ids` | slot 的实际拥有者，用于核对提示 |
| `active_numerator` | 已累计的加权数值之和，即分子 |
| `denominator` | 已累计的权重之和，即分母 |
| `max_logits` | 控制指数运算数值范围的最大分数 |

任一有效行的 `reset_slots` 等于实际 region 编号时，就忽略旧累计值，并保留现有代码对 token 暂存有效标记的清理。无效行不能触发重置或完成。

从 `index_k` 读取数值，从 `index_z` 读取分数，用分数计算权重，然后合并旧状态：

```text
新分子 = 缩放后的旧分子 + 当前各行的“权重 × 数值”之和
新分母 = 缩放后的旧分母 + 当前各行的权重之和
summary = 新分子 / 新分母
```

权重使用指数函数计算，分数越大权重越大。计算时先减去共同最大值，防止指数结果过大；旧分子和分母也转换到相同尺度。运算仍使用 FP32，即 32 位浮点数精度，公式沿用原实现。

例如数值为 2、6 的两个 token 权重相同，summary 就是 4。已有分子 8、分母 2，再加入数值为 8、权重为 1 的 token，且无需改变尺度时，新 summary 为 `(8 + 8) / (2 + 1)`。

## 7. 保存结果及两阶段的意义

完成的 region 把最终 summary 写入 `mean_cache`，释放原先占用的 slot。未完成的 region 保存分子、分母和最大值，供下一批次继续累计。

没有已有 slot 的未完成 region 需要申请空位。`atomic_cas` 是不可被其他申请者插入打断的“比较并交换”操作，用来检查空位并占用，防止两个 region 同时写进一个 slot。这个操作仍然保留，分配失败仍通过 `allocation_success` 报告。

两阶段保证本批次完成的 region 先释放 slot，随后未完成的 region 再申请。当前只借鉴 vLLM 的任务划分，没有照搬其 NVIDIA 路径的一次 update，也没有改变 Ascend 只在完成时写最终 summary 的行为。

## 8. 普通批次与 MTP 混合批次如何接入

普通 prefill 或包含单 token decode 的混合批次，传入所有有效请求的边界和长度。

MTP 验证是一次验证多个候选生成 token 的路径。它继续使用原有的暂存和延迟完成处理；只有后面的 prefill 请求交给本方案。

例如批次前 6 行属于验证请求，prefill 的边界为 `[6, 18, 23]`。调用方把 token 数据切成 `[6:]`，但边界无需额外生成一个减去 6 的数组：kernel 读取第一项作为 `row_base`，把边界归一化成 `[0, 12, 17]` 使用。

`task_prefix` 在绑定缓存时按 `max_num_seqs + 1` 预分配。`max_num_seqs` 是配置允许的最大请求数。每次使用已有空间并刷新内容，不恢复此前按 token 数分配的 scratch 表。

## 9. 输入约束与保护范围

1. 每个请求的输入行按连续序列位置排列，不能任意打乱或插入额外占位行。
2. `seq_lens - query_length` 必须正确表示本次之前的长度；边界有序，与 token 数组及切片起点一致。
3. 一个任务中的 token 属于同一个实际 region，首行必须提供正确的实际 region 编号。
4. 每个实际 region 在本次更新中只能有一个任务负责，不能由多个请求或逻辑 region 重复写入。
5. 有效 token 的 region 内位置不能重复，reset 不能表示区间中途重新开始一份独立状态。

外层检查 `live_rows` 范围、正的 region 大小、请求元数据长度及 task_prefix 的 int32 类型和容量。int32 表示 32 位整数。kernel 检查读取范围和实际 region 编号，但没有额外扫描去验证所有输入约定。

同一 region 连续已经不再是唯一依据：当前额外依赖请求的长度与位置关系。若违反这些约定，可能漏算或重复更新，不属于支持的输入。

## 10. 文件和接口变化

- [summary_kernels.py](../vllm_ascend/models/step5/summary_kernels.py)：新增任务累计数 kernel；update 改为任务查找和起始行计算；外层接口新增 `query_start_loc`、`seq_lens`、`task_prefix` 三个必填关键字参数。
- [dsa_attention.py](../vllm_ascend/models/step5/dsa_attention.py)：预分配请求级缓冲区，更新普通和混合验证批次的调用。

未重新引入 `torch.unique`、全局 token 行号表或逐行首行扫描。原有加权计算、两阶段调用及 slot 管理保持原样。

## 11. 验证状态与性能预期

本次不新增或修改测试文件，不运行会改写代码的 Ruff 或格式化操作。已有测试仍使用旧接口，运行前需适配新的参数。

设备编译、精度和性能尚未验证。本地可做语法及调用参数检查，以及任务边界公式的 CPU 推演；这些不能代替实际 NPU 验证。

相对此前方案一，update 不再检查每个输入行，只遍历 region 任务。但新增了一个任务准备 kernel，以及每个任务的请求查找成本，不能预先保证更快。

后续应验证空请求、从 region 中间开始或结束、跨批次累计、混合批次的非零行号起点、reset、slot 耗尽及释放后复用。性能对比应包含任务准备和两次 update 的总耗时，并覆盖小、中、大批次。比较数值时也需考虑浮点加法顺序变化，不能只凭数学公式相同认定逐位一致。
