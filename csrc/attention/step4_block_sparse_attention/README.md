# Step4BlockSparseAttention

## vLLM Ascend integration

The vLLM Ascend build registers this operator for Ascend 950. Set
`VLLM_ASCEND_ENABLE_STEP4_ASCENDC_DSA=1` before starting vLLM to use it for
Step4 DSA prefill and single-token decode. Leaving the variable unset keeps the
Triton implementation as the runtime baseline. Multi-token MTP verify is not
supported by the current Step4 DSA attention path.

面向 Step4 DSA 主 Attention 的块稀疏 FlashAttention：Indexer / CSA / top-k 在上游完成，本算子只消费 `packed_regions` 和主 KV cache。

Prefill 与 Decode **共用一套 aclnn 和一个 kernel**，**没有 `sparse_mode` 属性**。由 `packed_regions` 的 dtype 分派：

| packed dtype | 模式 | tiling key (BF16 / FP16) | `kv_seqlens` |
| --- | --- | --- | --- |
| INT32 | Prefill | 0 / 2 | 不传 |
| INT64 | Decode | 1 / 3 | **必传** |

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

当前仅注册 `ascend950`（Arch35），MIX AIC 1:2。

## 功能说明

每个 `(KV group, query token)` 拿自己的 region 列表，从 TGD KV cache gather 后做 QK → softmax → PV。列表是 **non-union**：query 之间不做 region 并集。

$$
\mathbf{o}_{q,h}
=
\sum_{t\in\mathcal{K}(q)}
\frac{\exp(\lambda\,\mathbf{q}_{q,h}\cdot\mathbf{k}_t)}
{\sum_{t'\in\mathcal{K}(q)}\exp(\lambda\,\mathbf{q}_{q,h}\cdot\mathbf{k}_{t'})}
\mathbf{v}_t,
\qquad
\lambda=\texttt{scale\_value}\ \text{或}\ 1/\sqrt{D}
$$

\(\mathcal{K}(q)\) 由 `packed_regions` 展开：物理 region ID × `region_size`，再截到该 region 的可见 token 数。TGD 扁平 cache 上地址是 `page * page_size + (phys % regions_per_page) * region_size + j`，其中 `regions_per_page = page_size / region_size`；页按 region-id 顺序排时等价于 `phys * region_size + j`。Decode 的 `start` **只用于算可见长度**，不改 gather 基址。

目标模型几何（测试按此钉死，不是 GBSA 那种随机 shape 扫表）：

| 项 | 值 |
| --- | --- |
| GQA | 64 / 4 |
| head dim \(D\) | 192 |
| region_size | 8（Indexer 粒度，不变） |
| page_size / blocksize | 128（一页 16 个 region；以前是 16，一页 2 个 region） |
| 选择的 region | topk ≈ 513（约 4104 个实际 attend token） |
| KV cache / `kv_seqlens` | 32K / 64K |

## packed_regions 编码

布局一律 **group-major** `[G * T, K]`，`K` 为 packed 宽度。有效槽数以 `region_counts`（INT32 `[G * T]`）为准，不要扫到 `K`。

### Prefill（INT32）

```
word = phys_region | (valid_tokens << 24)
```

- `phys_region`：低 24 bit（`0x00FFFFFF`）
- `valid_tokens`：该 region 实际参与 softmax 的 token 数，通常为 `region_size`，尾块可更短；占 bit[31:24]，范围 1–255
- 无效槽：只认哨兵 **`-1`（`0xFFFFFFFF`）**。不要用 `meta < 0`：`valid` 高位会把 INT32 符号位置位

### Decode（INT64）

```
word = start_token | (phys_region << 32)
valid = clamp(kv_seqlen - start_token, 0, region_size)
```

- `kv_seqlens`：INT32 `[T]`，当前 token 的绝对长度（**不是前缀和**）
- **`packed == 0` 是 region-0 的合法编码，不要当 skip**
- 无效槽习惯填 `0` 以外的哨兵（如 `-1` / `0x7FFFFFFFFFFFFFFF`）；可见长度为 0 时该槽不贡献 token

### 空 packed

`region_counts == 0` 或展开后可见长度为 0 时，kernel 不写该行的 O / LSE。Torch wrapper 在整表为空时把 O / LSE 置 0。当前契约是 **O=0、LSE=0**，不是 `-inf`。

## 参数说明

维度：`T` 为 query 总 token（Decode 时常为 batch / MTP 长度），`T_kv` 为 KV cache token 数，`N1` 为 Q head 数，`G` 为 KV head 数，`D` 为 head dim，`K` 为 packed 宽度。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 格式 |
| --- | --- | --- | --- | --- |
| query | 输入 | TND `[T, N1, D]` | BF16、FP16 | ND |
| key | 输入 | TGD `[T_kv, G, D]`，直接读主 KV cache | 与 query 相同 | ND |
| value | 输入 | 与 key 同 shape | 与 query 相同 | ND |
| packed_regions | 输入 | group-major `[G*T, K]`；INT32 Prefill / INT64 Decode | INT32、INT64 | ND |
| region_counts | 输入 | `[G*T]`，每行实际使用的 region 数 | INT32 | ND |
| kv_seqlens | 可选输入 | Decode 必传 `[T]`；Prefill 不传 | INT32 | ND |
| attention_out | 输出 | 与 query 同 shape | 与 query 相同 | ND |
| softmax_lse | 可选输出 | `[T, N1]`，自然对数；不返回时为空 tensor | FP32 | ND |
| num_heads | 属性 | N1 | int | — |
| num_key_value_heads | 属性 | G，需整除 `num_heads` | int | — |
| region_size | 属性 | 每 region 的 token 数，默认 8 | int | — |
| page_size | 属性 | paged block（blocksize），默认 128，须整除 `region_size` | int | — |
| scale_value | 属性 | 0.0 表示 `1/sqrt(D)` | float | — |
| layout_q | 属性 | 当前只用 `"TND"` | string | — |
| layout_kv | 属性 | 当前只用 `"TGD"` | string | — |
| num_splits | 属性 | 仅支持 1；`>1` 未实现 | int | — |
| return_softmax_lse | 属性 | 是否输出 LSE | int / bool | — |

QKV 必须同 dtype。`D` 上限 256。GM 上 Q/O/LSE 按真实 `groupSize = N1/G` 读写；Cube / softmax / PV 用 `qBaseTile = AlignUp(groupSize, 16)`。

## 接口

### Torch

```python
from cann_ops_transformer_custom.ops import step4_block_sparse_attention
# 或：from cann_ops_transformer.ops import step4_block_sparse_attention

# Prefill
attn, lse = step4_block_sparse_attention(
    q, k, v, packed, counts,
    num_heads=64, num_key_value_heads=4,
    region_size=8, return_softmax_lse=True,
)

# Decode（kv_seqlens 必传）
attn, _ = step4_block_sparse_attention(
    q_dec, k, v, packed_dec, counts_dec,
    num_heads=64, num_key_value_heads=4,
    kv_seqlens=seq_lens,
    region_size=8,
)
```

### aclnn

```c
aclnnStatus aclnnStep4BlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *packedRegions, const aclTensor *regionCounts,
    const aclTensor *kvSeqlensOptional,
    int64_t numHeads, int64_t numKeyValueHeads,
    int64_t regionSize, int64_t pageSize, double scaleValue,
    char *layoutQ, char *layoutKv, int64_t numSplits, int64_t returnSoftmaxLse,
    aclTensor *attentionOut, aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnStep4BlockSparseAttention(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

C++ 示例：`examples/test_aclnn_step4_block_sparse_attention.cpp`。精度以 Torch pytest 为准。

## 约束与未支持

- 无 `sparse_mode` 属性；不要按 GBSA 的 metadata 流程调用
- `num_splits > 1` 未实现
- PA `[pages, page_size, G, D]` + `block_table` 未支持；当前 KV 仍是 TGD `[T_kv, G, D]`
- 非 TND·TGD layout 未走通
- Graph convert 目前按 BF16 IR 注册
- 空 packed 的 LSE 为 0，不是 `-inf`

## 编译与安装

```bash
cd <ops-transformer>
bash build.sh --pkg --soc=ascend950 --ops=step4_block_sparse_attention
```

把生成的自定义 opp 装到本机 CANN vendor 路径，并 `source` 对应 `set_env.bash`。Torch 扩展首次调用会 JIT 出 `.so`。磁盘紧张时可把缓存指到 `/tmp`：

```bash
export TORCH_EXTENSIONS_DIR=/tmp/anthony_torch_ext
export TMPDIR=/tmp
```

## 精度测试

入口：`tests/pytest/`。Golden 为低精度输入升 FP32 后的 CPU 参考；O 用 `rtol=1e-2 atol=4e-3`，LSE 用 `1e-3`。每个 case 打印 `max_diff` / `mean_diff`、相对 L1、`cosine_similarity(actual, golden)`。

当前 **100** 个 pytest item（99 条具名 packing / scene / fill + `test_pd_shared`）。

```bash
export TEST_DEVICE_ID=4          # NPU id，默认 5
export TORCH_EXTENSIONS_DIR=/tmp/anthony_torch_ext
export TMPDIR=/tmp
export ASCEND_GLOBAL_LOG_LEVEL=4
source /usr/local/Ascend/cann/set_env.sh
source <custom_opp>/vendors/custom_transformer/bin/set_env.bash

python -m pytest -s -v attention/step4_block_sparse_attention/tests/pytest
# 或：bash attention/step4_block_sparse_attention/tests/pytest/test_run.sh single
# 过滤：pytest -k causal_chunk64
```

用例生成在 `examples/test_torch_step4_block_sparse_attention.py` 的 `build_cases()`。也可直接：

```bash
python attention/step4_block_sparse_attention/examples/test_torch_step4_block_sparse_attention.py
```

`CASE_TAG` / `CASE_ID` / `TEST_VERBOSE` 只作用于该脚本的 `main()`，pytest 走全量 parametrize。

## 目录

```
step4_block_sparse_attention/
├── op_host/            # OpDef、tiling、aclnn
├── op_kernel/          # Arch35 kernel（Prefill/Decode 同一份，按 tiling key 实例化）
├── torch_extension/    # Torch schema / eager / graph convert
├── examples/           # Torch 精度脚本、aclnn C++ 示例
└── tests/pytest/       # pytest 入口
```
