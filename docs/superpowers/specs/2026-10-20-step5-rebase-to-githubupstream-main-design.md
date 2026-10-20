# Step5 NPU 适配 rebase 到 githubupstream/main — 设计

日期：2026-10-20
状态：设计已确认，待用户评审 spec

---

## 1. 目标

把 `vllm-ascend-step4` 仓库 `step5-swf-journal-ascend-ops` 分支上的 Step5 NPU 适配，迁移到
`vllm-ascend-private` 仓库基于 `githubupstream/main` 的新分支上。

原分支的 base（vllm-ascend v0.30.0rc1 快照）和它所配的 vLLM（stepcast fork）都已落后；
`githubupstream/main` 已更新且其 CI 默认跑在 CANN 9.2.0 上，vLLM 官方 main 也已合入 step5。
迁移后与官方 main 同构，降低后续合并成本。

**最终验收标准**：在申请到的开发环境上端到端拉起 Step5 的 vLLM server。

---

## 2. 事实基线（已核实）

### 2.1 两侧对应关系

| | 源（step4 分支） | 目标（rebase 后） |
|---|---|---|
| vllm-ascend | `upstream/step5-swf-journal-ascend-ops` @ `317c7fb99e`，base `5f828ad638`（v0.30.0rc1） | `githubupstream/main` @ `399c30d3a`（2026-10-20） |
| vLLM | fork `step-5-preview-stepcast-plugin` | **官方 vLLM main** + stepcast_plugin（out-of-tree） |
| step5 摘要架构 | **SWF journal**（`sparse_summary_cache.py` 503 行，含 request journal / side-storage 协议） | **TailRing**（159 行，`csa_tail_ring.py` + `CircularBufferSpec`） |
| torch / torch-npu | 2.13.0 / 2.13.0rc1 | **2.10.0 / 2.10.0.post4** |
| triton-ascend | 3.6.0+dev20261013182112 | **3.2.2** |
| CANN | 9.1.0（Dockerfile 默认） | 9.2.0（CI daily 默认） |
| transformers | 5.14.1 | 5.19.0 |
| VLLM_TAG | v0.30.0 | v0.31.0 |

分叉点：`ac1b7bceb`（2026-10-08），相距 302 commit / 12 天。

### 2.2 vLLM 侧 API 兼容性（已逐个符号核实）

step4 的 step5 代码从 `vllm.models.step5.nvidia.*` import 37 个符号，官方 vLLM main **全部命中**
（36 个直接定义 + 1 个 `mtp.py` 转手导入）。

其他 vLLM 侧依赖：

| API | 官方 main | 说明 |
|---|---|---|
| `make_dummy(is_padding=)` | ✅ | 签名一致 |
| `mtp_validation` / `EncoderCache` / `DefaultModelState` / `AttentionGroup` / `AttentionType` / `KVCacheConfig` | ✅ | 全部存在 |
| `share_replayssm_ring_trackers` | ✅ | 在 `mamba_mixer2.py` |
| `KVCacheSideStorageLayer` 协议 | ❌ 不存在 | step4 代码有 `getattr(None)` 守卫，安全降级 |
| `max_returned_cache_hit_length` | ❌ 不存在 | step4 patch 自加参数，默认 `None` 时行为与原版等价 |
| `MultiModuleMTPSpeculator` | ✅ | `multi_module_mtp/speculator.py:28`，`DraftModelSpeculator` 子类 |

**关键差异**：`sparse_summary_cache.py` 的 API 完全不兼容 —— step4 的
`AscendStep5SparseSummaryCache.__init__` 调
`super().__init__(pages, block_size, device, max_num_seqs, request_journal_width)`（5 参数），
而 main 版是 `__init__(self, mean)`（1 参数）。且 main 版无 `mean_cache/num_pages/page_size/
region_block_size/summaries_per_page/num_kv_heads/proxy_dim` 这 7 个 property，无
`update_request_journal` / `update_speculative` / `speculative_prefix_metadata`。

### 2.3 stepcast_plugin（vLLM 侧附加）

- 只存在于 fork（40 文件），官方 main 的 `vllm/` 里没有
- out-of-tree 包，通过 vLLM plugin entry points 加载：
  `vllm.endpoint_plugins → stepcast_platform`、`vllm.stat_logger_plugins → stepcast_rs`
- 两个入口点组在官方 main 的 plugin 机制里都存在
- 需 `VLLM_PLUGINS=stepcast_platform,stepcast_rs`，否则内部路由全 404
- **vllm-ascend 不 import 它**（已 grep 确认为空），与本次 rebase 零耦合
- 它 import 的 `vllm.*` 内部 API（`OpenAIServingChat`、`OnlineRenderer`、`StatLoggerBase`、
  `IterationStats`、`validate_json_request`、`vllm.entrypoints.chat_utils`、
  `vllm.v1.core.kv_cache_utils` 等）可能与官方 main 有偏差 —— fork 与 main 在 2026-09-20
  分叉后 1760 文件不同。**列为开发环境第一阶段自检项**。

### 2.4 冲突面（107 个改动文件）

| 类别 | 数量 | 处置 |
|---|---|---|
| `ci/`（含 82MB `source.bundle`） | 40 | **不搬**（复用 main 构建） |
| `vllm_ascend/` 纯新增代码 | 19 | 搬 |
| `tests/ut/` 纯新增 | 19 | 搬，合并为 6 个 |
| `csrc/third_party/catlass`（submodule 指针） | 1 | **不搬**，见 §7.1 |
| 已存在且 main 未动 | 8 | 干净应用 |
| 已存在且 main 改过 | 21 | 需人工 port |

---

## 3. 已确认决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | vLLM 侧用**官方 main** + stepcast_plugin（out-of-tree） | 与 vllm-ascend main 同构；plugin 不影响 `vllm/` 树 |
| D2 | SWF 摘要链路**按 TailRing 架构重写** NPU 侧 | vLLM main 已移除 SWF journal 基础设施 |
| D3 | torch/triton **跟随 main**（2.10.0 + triton-ascend 3.2.2） | 保持同构；`extension.parallel` 缺失时回落 `tl.range` |
| D4 | 落点：private 仓库**新分支** `step5/tailring-rebase`，基于 `githubupstream/main` | 不污染 `stepfun/v0.19.0` 与本地 `main` ref |
| D5 | 采用**方案 B 分层重建**，不保原始 28 提交粒度 | 同一文件只解一次冲突；每层可独立验证 |
| D6 | `ci/` 不搬 | 复用 main 构建 |
| D7 | `cann_ops_transformer` 依赖文件**照搬 + 标记 + try/except 保护** | 必须但不在 main requirements |
| D8 | 依赖与 CMake **全部优先 main** | main 基线即选定基线 |
| D9 | `patch_bind_kv_cache.py` 的 side-storage 改动**跳过** | TailRing 下无 side-storage 协议，搬过去是死代码 |
| D10 | `patch_kv_cache_coordinator.py` 的 replay ceiling **不搬** | TailRing 下 region 由原始行重算，回放越界不影响正确性 |

---

## 4. 分支拓扑与分层

```
githubupstream/main (399c30d3a)
        │
        └──► step5/tailring-rebase
                 L0  基线（已完成：分支已创建）
                 L1  19 个纯新增代码文件 + 6 个合并后 UT + catlass 不动
                 L2  8 个 main 未动文件（跳过 patch_bind_kv_cache，实为 7 个）
                 L3  低风险 port（envs / models 注册 / modelslim / quant_type /
                     fp8_config / fp8_block / method_adapters / sparse_offload_manager）
                 L4  中风险 port（aclgraph_utils / input_batch / attn_utils /
                     worker / spec_decode.__init__ / lmhead_tp_utils）
                 L5  高风险 port（model_runner / speculator —— 后者需人工重写）
                 L6  TailRing 重写（summary_cache / summary_kernels / dsa_attention）
                 L7  multi_module_mtp + core/step5_kv_cache 适配
                 L8  基线收尾（requirements / pyproject / CMakeLists 保持 main 原样）
```

### 提交组织

- 每层 1–3 个提交，同层同子系统文件合成一个提交
- 提交 message 沿用原始语义，去掉已被重写的 SWF 描述
- 原始提交 `cc30d9ebb`、`3bf138a84`、`8f879885f`、`d98224a02e`、`a096cfcb5`、
  `4a238a666` 的内容被 L6 作废，不单独保留
- MR 描述标注原始工作来源：`upstream/step5-swf-journal-ascend-ops @ 317c7fb99e`

### 每层准入门槛

| 层 | 本地可验证 | 需 NPU |
|---|---|---|
| L0–L2 | `import vllm_ascend` + `pytest --collect-only tests/ut` | — |
| L3–L5 | 上述 + 对应 UT 全绿 | — |
| L6 | libdevice checkpoint + summary contract UT | 数值验证 |
| L7 | MTP contract UT | — |
| L8 | 构建配置核对 | 编译 + 单卡 smoke |

---

## 5. 文件清单

### 5.1 19 个纯新增代码文件

```
vllm_ascend/core/step5_kv_cache.py                                    159 行
vllm_ascend/models/step5/__init__.py                                   26
vllm_ascend/models/step5/_decode_summary_layout.py                    280
vllm_ascend/models/step5/_step5_compat.py                             177
vllm_ascend/models/step5/dsa_attention.py                             971   ← L6 重写
vllm_ascend/models/step5/dsa_kernels.py                              4453   ← parallel 双路径
vllm_ascend/models/step5/fp8_kernels.py                               314
vllm_ascend/models/step5/fused_add_rms_norm.py                        151
vllm_ascend/models/step5/fused_qknorm_rope.py                         812
vllm_ascend/models/step5/generic_block_sparse_attention.py            280   ← 需 cann_ops_transformer
vllm_ascend/models/step5/indexer_region_score.py                       93   ← 需 cann_ops_transformer
vllm_ascend/models/step5/model.py                                    1665
vllm_ascend/models/step5/model_state.py                                62   ← 删除，见 §6.4
vllm_ascend/models/step5/mtp.py                                       178
vllm_ascend/models/step5/summary_cache.py                             391   ← L6 重写
vllm_ascend/models/step5/summary_kernels.py                           542   ← L6 重写
vllm_ascend/models/step5/vl.py                                         41   ← 调整
vllm_ascend/worker/v2/spec_decode/multi_module_mtp/__init__.py          2
vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py      156
```

### 5.2 UT 合并：19 → 6

| 合并后文件 | 来源 | 行数 |
|---|---|---|
| `tests/ut/models/test_step5_model_contract.py` | `test_step5_execution_contract`(573) + `test_step5_fp8_config_contract`(94) + `test_step5_dummy_input_contract`(47) | 714 |
| `tests/ut/models/step5/test_step5_summary_contract.py` | `test_summary_cache_contract`(398) + `test_summary_update_dispatch_contract`(279) + `test_summary_scratch_contract`(155) | 832 |
| `tests/ut/models/step5/test_step5_dsa_contract.py` | `test_native_dsa_contract`(409) + `test_ar_graph_step_contract`(379) | 788 |
| `tests/ut/models/step5/test_step5_mtp_contract.py` | `test_multi_module_mtp_contract`(394) + `test_mtp_quantization_contract`(258) | 652 |
| `tests/ut/quantization/test_step5_quant_contract.py` | `test_fp8_moe_contract`(402) + `test_step5_vlm_loading`(126) + `test_step5_modelslim_loading`(107) + `test_step5_modelslim_mapping`(78) + `test_mxfp8_dummy_scales`(56) | 769 |
| `tests/ut/worker/v2/test_step5_kv_contract.py` | `test_step5_shared_kv`(439) 单独 | 439 |

合计 4,194 行 / 6 文件（原 4,940 行 / 19 文件）。

**这些 UT 全部是 AST isolation 风格**（从源码编译类定义、不初始化 accelerator runtime），
因此能在无 NPU 的开发机上运行 —— 这是本方案验证策略的基础，合并时不得丢失该特性。

**`test_step5_summary_contract.py` 的三个来源测的是 SWF journal**，必须随 L6 一起按
TailRing 重写，不能直接搬。

### 5.3 UT 的其他处置

- `tests/ut/patch/worker/test_step5_side_storage_binding.py`(64) → **删除**（D9）
- `tests/ut/distributed/kv_transfer/test_sparse_offload_lazy_import.py`(81) →
  **并入** main 已有的 `tests/ut/distributed/kv_transfer/sparse_kv_offload/
  test_sparse_kv_offload_manager.py`（已有 14 个 test）。
  注意：main 的 `sparse_kv_offload_manager.py` 用的是**模块级**
  `from memfabric_hybrid import offload`（第 15 行，包在 `contextlib.suppress` 里），
  **没有** `_load_offload_backend`。step4 的懒加载改进 main 确实没有，该测试有效。

---

## 6. L6：TailRing 重写（核心）

### 6.1 main 的 TailRing 架构

**核心思想**：一个 region 的摘要是该 region 8 行 index-key 的定序 online softmax，
所以**原始行就是全部状态** —— 不需要 FP32 累加量，不需要 undo log。

```
Step5CSATailRing  (nn.Module, AttentionLayerBase)
  ├─ 每请求一个环形缓冲，存开放 region 的原始 [index_k | index_z] 行（512 元素 bf16）
  ├─ required_rows = 7 + num_speculative_tokens
  │    （verify 存 k+1 行，下一步可能回放同 region 的前 7 行）
  ├─ get_kv_cache_spec() → CircularBufferSpec(block_size=主层 block_size,
  │                              head_size=page_bytes//(block_size*2), dtype=bf16,
  │                              block_stride_alignment=1024)
  ├─ bind_kv_cache() → self.rows as_strided 视图，capacity = page_bytes // 1024
  └─ metadata() → Step5CSATailRingMetadata(base, slot_mapping)

Step5SparseSummaryCache(mean)     ← 只包 [blocks, block_size//8, 256] uint8
  ├─ update_decode(key, z, slots, positions, ring)
  ├─ update_verify(key, z, slots, query_start_loc, seq_lens, ring)  ← 无主机元数据，可图捕获
  └─ update_prefill(key, z, metadata, positions, ring)

Step5SparseSummaryCacheLayer      ← 普通 FullAttentionSpec(tokens_per_state=8, uint8, 256)
  └─ bind_kv_cache(kv_cache) → Step5SparseSummaryCache(kv_cache[:, 0])
```

**投机解码回滚**：`csa_tail_ring.py` 注释 ——
*"a rejected draft is undone by the next step overwriting its row"*。
被拒 draft 写的摘要没人读；region 重新闭合时自然被重写。这是 SWF journal 的反面：
SWF 需要 513 行 FP32 undo log + tag 失效 + COW 重映射，TailRing 什么都不需要。

### 6.2 `summary_cache.py` 重写映射（391 → ~150 行）

| step4 现状（SWF） | TailRing 目标 |
|---|---|
| `AscendStep5SparseSummaryCache(Step5SparseSummaryCache)`，5 参数构造 | 子类化 `Step5SparseSummaryCache(mean)`，**无构造参数** |
| 7 个 property（`mean_cache` 等） | **全删** |
| `update(key, z, slots, prefill_metadata)` | → `update_decode(key, z, slots, positions, ring)` |
| `update_request_journal(...)` 派发 | → `update_verify(key, z, slots, query_start_loc, seq_lens, ring)` |
| `update_speculative(...)` + `_speculative_tails` dict | **删除** |
| `speculative_prefix_metadata(...)` | **删除** |
| `prefill_metadata(slots, device)` | → `update_prefill(key, z, metadata, positions, ring)` |
| `copy(block_copies)` 的 journal 重映射 + `_invalidate_request_journal` | **删除** |
| `reset(block_ids, clear_mean)` | 简化为 `mean` 清零 |
| `Step5SparseSummaryCacheConfig(pages, block_size, 8, 1, 256)` | **删除** |
| NPU/CPU dispatch 判断模式 | **保留**（换目标函数） |

### 6.3 `summary_kernels.py` 适配（542 → ~300 行）

| step4 kernel | 处置 |
|---|---|
| `_maximum`、`_accumulate`（online softmax 数学） | ✅ 原样复用（与 main 版逐字一致） |
| `_store_summary` | ✅ 复用 |
| `_build_prefill_groups` / `build_prefill_groups` | ✅ 复用 |
| `make_prefill_metadata_from_slots` / `make_prefill_metadata` | ✅ 复用 |
| `_update_decode_values` / `_update_decode` | ⚠️ 改签名 + **新增 `_replay_ring`** |
| `update_prefill` | ⚠️ 改签名 + 落盘改为 ring 回放 |
| `_accumulate_prefill` / `_commit_prefill` | ⚠️ 适配 ring |
| `_update_request_journal` / `update_request_journal` | ❌ **删除**（约 95 行） |

### 6.4 `dsa_attention.py` 去 journal（955 → ~800 行）+ `model_state.py` 删除

| step4 现状 | TailRing 目标 |
|---|---|
| 四级 builder 的 `step5_request_state` 字段（134/163/181/195 行） | ❌ 删除，改用 `ring.metadata().base` / `slot_mapping` |
| `:596-598` 断言 "Step5 journal requires persistent v2 request state indices" | ❌ 删除 |
| `:614` `summary_cache.update_request_journal(...)` | → `update_verify/decode/prefill(...)` |
| `:279` "Schedule native attention ... without touching journals" | 保留，去 journal 语义 |
| native op 适配、DSA metadata 构建 | ✅ 保留 |

`model_state.py` 整个删除的理由：`AscendStep5ModelState` 唯一新增逻辑就是维护
`request_state` 并注入 builder（`builder.step5_request_state = self.request_state`），
而它服务的 SWF undo journal 在 vLLM main 已整体移除（main 无 `step5/nvidia/model_state.py`，
step5 的 model.py/vl.py/mtp.py 均无 `get_model_state_cls`）。main 的
`Step5SparseSummaryCache` docstring 明写 "this cache holds no per-request state"。

#### 待确认项（用户要求记录）

`vl.py` 的 `get_model_state_cls()` override 有两个作用：
1. 注释所写 "Keep multimodal input preparation on the Ascend v2 attention path"
2. main 的 `longcat_flash_ngram.py:10` 说明 —— "`get_model_state_cls` makes the model
   MRV2-only"（定义该方法本身即把模型限定在 MRv2 路径）

**暂定方案 (b)**：保留 override，但改为返回 main 默认的 `AscendModelState`，
既去掉 journal 又保住 MRV2-only 语义。代价是 `vl.py` 留一个 3 行 override。

**备选方案 (a)**：完全删除 override，依赖 main 的 `resolve_model_state_cls()` 默认解析。

**验证通过后的最终决策标准**：在 NPU 上确认删除 override 后 Step5 VL 仍走 Ascend v2
attention 路径。若确认走，则简化为 (a)；否则维持 (b)。

**注意**：vllm-ascend 侧有自己的 `init_asecnd_model_state()` 入口
（`vllm_ascend/worker/v2/model_states/__init__.py`），默认解析如何接入 main 的
`resolve_model_state_cls()` 需在开发环境上验证，本地无法确认。

### 6.5 L6 第一个 checkpoint：libdevice

**问题**：main 的参考实现 `vllm/models/step5/nvidia/ops/summary_update.py:17` 是
`libdevice = tl.extra.cuda.libdevice`，并在 `_accumulate` 里调 `libdevice.exp`。
在 Ascend 上这会导致 Triton 解析 CUDA libdevice 符号而非 CANN 等价物，kernel 编译失败。
vllm-ascend 自己的 `patch/__init__.py:1396-1402` 记录了这个坑。

**处置**：改为 `tl.extra.cann.libdevice`。这不是新风险 —— step4 的
`summary_kernels.py:13` 已经这么做，且 triton-ascend 3.2.2 上该模块存在
（main 的 `quantize_indexer_query.py` 用 `nearbyint`、`num_nans.py` 用 `isnan`）。

**checkpoint 步骤**：
1. 改 import 为 `tl.extra.cann.libdevice`
2. 写最小 `@triton.jit` kernel，内部调一次 `libdevice.exp`
3. NPU 上编译并执行，与 `torch.exp` 对拍
4. 通过 → 继续写 `_replay_ring` / `update_decode` / `update_verify` / `update_prefill`
   失败 → 停止，先解决 triton-ascend 3.2.2 的 libdevice 可用性（环境问题，非代码问题）

---

## 7. L4/L5/L7：spec_decode 与 worker/v2 port

### 7.1 catlass submodule（不搬）

`6bcfb8624` 把 `csrc/third_party/catlass` 从 `41bf90da65` 改为 `038b9315f3`。

核实结果：

| | `41bf90da65` | `c89fe73`（官方修复） | `038b9315f3`（step4 钉的） |
|---|---|---|---|
| 在官方 `gitcode.com/cann/catlass` | ✅ `origin/master` | ✅ 在 | ❌ **不在任何分支** |
| 日期 | 2026-04-09 | 2026-07-28 | — |
| 作者 | pandeng01 | init__zhb__ | — |
| 提交信息 | `add 47\50 test example` | `针对VF函数补充 static 修饰` | — |
| 相对关系 | 是 `c89fe73` 的祖先 | — | — |

`038b9315f3` 是 fork 私有 commit。`c89fe73` 是官方对应修复（19 文件 +300/−277），
但不在 `origin/master` 上。

**决策：不搬**，rebase 后保持 main 的 `41bf90da65`。若 CANN 9.2 下编译失败，
再单独提一个指向官方 `c89fe73` 的 submodule bump MR。

### 7.2 `speculator.py` —— 全场最难冲突，人工重写

三方改动叠加：

1. **main 侧的 vLLM 版本分叉**：
   ```python
   # vLLM #60335 renamed `autoregressive` to `target_dependent_ar` on main;
   # the release lane (v0.31.0) still ships the old module and class name.
   if vllm_version_is("0.31.0"):
       from ...spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
   else:
       from ...spec_decode.target_dependent_ar.speculator import (
           TargetDependentARSpeculator as AutoRegressiveSpeculator)
   ```
   我们的 vLLM 是官方 main（非 v0.31.0 release），走 `else` 分支。

2. **main 侧的 PCP sharded decode 新分支**：`is_decode_sharded` 判断，step4 没有。

3. **step4 侧的 307 行删除重构**（`fc9c5badb`）：重写为基于 vLLM 上游
   `MultiModuleMTPSpeculator` 的薄适配层
   `AscendMultiModuleMTPSpeculator(AscendDraftSpeculatorMixin, MultiModuleMTPSpeculator)`。

### 7.3 `multi_module_mtp/` 可行性（已核实，可行）

`MultiModuleMTPSpeculator` 在官方 vLLM main 存在（`multi_module_mtp/speculator.py:28`，
`DraftModelSpeculator` 子类，工厂注册于 `spec_decode/__init__.py:61,64`）。

**两个注意点**：
- stepcast 独有的 `attn_vllm_config()` property 在官方 main 上**没有**，
  step4 若调用需改走别的路径
- 官方 main 的 `multi_module_mtp/` 多了 `_shift_input_ids_kernel` 等 kernel，上游有演进，
  step4 的薄适配层要按 main 版本对齐而非照抄

**定级：接口对齐，非重写**，风险远低于 L6。

### 7.4 其他 L4/L5 文件

| 文件 | main 侧改动 | step4 侧改动 | 定级 |
|---|---|---|---|
| `model_runner.py` | +298/−89 | +17 | 高 |
| `attn_utils.py` | +194/−41 | +6 | 高 |
| `worker.py` | +129/−34 | +7 | 高 |
| `aclgraph_utils.py` | +25/−11 | +23 | 中 |
| `input_batch.py` | +39/−1 | +4 | 中 |
| `autoregressive/aclgraph.py` | +18/−2 | +56 | 中 |
| `spec_decode/__init__.py` | +7/−1 | +12 | 小 |
| `lmhead_tp_utils.py` | — | +27 | 干净应用 |

---

## 8. 构建基线

**D8：全部优先 main。**

`requirements.txt` / `pyproject.toml` / `CMakeLists.txt` **完全保持 main 原样，不搬 step4 改动**：

- main 的 torch 2.10.0 正是我们选定的基线
- step4 对 `CMakeLists.txt` 的改动（`90057c38f7`）是把 torch 校验从"只接受 2.10.0"
  放宽到"接受 2.10.0 或 2.13.0" —— 我们只要 2.10.0，无需该改动
- step4 的 torchaudio 是被注释掉的（"torch 2.13-compatible torchaudio is not in this
  wheelhouse"），main 有 `torchaudio==2.10.0`，跟随 main

**因此 L8 几乎为空**，只剩：
1. `dsa_kernels.py` 的 `tl.extra.cann.extension.parallel` 双路径
   （triton-ascend 3.2.2 可用性未知；main 从未用过 `parallel`，缺失时回落 `tl.range`）
2. L6 的 libdevice 换 cann flavor
3. catlass 保持 `41bf90da65` 不动

### `cann_ops_transformer` 依赖（D7）

被 `generic_block_sparse_attention.py` 和 `indexer_region_score.py` 依赖
（提交 `c8cb3880a` 引入）。main requirements 里没有，且只以 wheel 形式存在于 fork 的
`ci/inputs/`（而 `ci/` 不搬）。

**处置**：两个文件照搬，但加显式标记 + try/except 保护：

```python
# REQUIRES: cann_ops_transformer (Ascend native sparse attention ops).
# Not in vllm-ascend requirements. Install from the StepFun wheelhouse or
# vendor under csrc/third_party/ before enabling Step5 native DSA.
try:
    from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (...)
except ImportError as exc:
    raise ImportError(
        "Step5 native DSA requires cann_ops_transformer; see the note above."
    ) from exc
```

这样分支在未安装该包的环境下仍能 import，只是 Step5 原生 DSA 路径不可用。

---

## 9. 验证矩阵

### 9.1 本机可验证（无 NPU，torch 2.9.1+cu130）

| 层 | 检查 | 命令 |
|---|---|---|
| L0–L2 | import + UT 收集 | `python -c "import vllm_ascend"`、`pytest --collect-only tests/ut` |
| L3–L5 | 对应 UT 全绿 | `pytest tests/ut/models/step5 tests/ut/worker/v2 tests/ut/quantization` |
| L6 | libdevice checkpoint + summary contract UT | 见 §6.5 |
| L7 | MTP contract UT | `pytest tests/ut/worker/v2/test_multi_module_mtp_contract.py` |

### 9.2 必须 NPU（开发环境）

| 项 | 内容 |
|---|---|
| V1 | libdevice checkpoint：最小 triton kernel 调 `libdevice.exp` 编译通过 |
| V2 | `_replay_ring` 数值正确性（与 main 的 CPU 参考实现对拍） |
| V3 | 950DT UB 容量（step4 有 `cbb432f62 fix(step5): fit indexer cache gather in 950DT UB`） |
| V4 | verify 路径图捕获（TailRing 的 `update_verify` 标榜 "graph capturable"） |
| V5 | 单卡 bf16 / FP8-BLOCK / ModelSlim 三权重路径 |
| V6 | MTP + VL 前端 |
| V7 | TP8+EP dummy run（step4 基线：92 层 75.68 GiB/卡） |
| V8 | **端到端拉起 Step5 vLLM server**（最终验收标准） |
| V9 | stepcast_plugin 的 `import stepcast_plugin` 自检 + `VLLM_PLUGINS` 生效验证 |

### 9.3 回退点

每层结束都是可提 MR 的状态。任何一层卡住，已完成的部分不受影响。

```
L0 ──► L1 ──► L2 ──► L3 ──► L4 ──► L5 ──► L6 ──► L7 ──► L8
 │      │            │            │      │      └─ NPU: V4,V5,V6,V7,V8,V9
 │      │            │            │      └─ NPU: V1,V2,V3 + libdevice checkpoint
 │      │            │            └─ 本机 UT 全绿（含 spec_decode 重构后）
 │      │            └─ 本机 UT 全绿
 │      └─ 本机 import + collect
 └─ 分支基于 main，import 通过（已完成）
```

---

## 10. 已知风险与待确认项

### 10.1 优先级：高

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **L6 TailRing 重写**（`summary_kernels.py` 约 300 行，新增 `_replay_ring`，三路签名变更） | 唯一真正的大工作量 + 最不可能本地验证 | §6.5 的 libdevice checkpoint 提前暴露地基问题 |
| R2 | `speculator.py` 人工重写（三方改动叠加） | 可能反复 | 独立成 L5，与 L6 解耦 |
| R3 | `tl.extra.cann.extension.parallel` 在 triton-ascend 3.2.2 上可用性未知 | `dsa_kernels.py` 两处（2805/3088 行） | 双路径：`hasattr` 守卫 + 回落 `tl.range` |

### 10.2 依赖与基线类（用户特别要求：后续验证出问题先回顾这些）

> **D8 决策：requirements / pyproject / CMakeLists 全部优先 main。**
> **D3 决策：torch 2.10.0 + triton-ascend 3.2.2 跟随 main。**
> 若后续验证出现编译失败、kernel 行为异常、数值不一致等问题，
> **首先回顾是否由这两条决策引起**，而不是先怀疑 step5 业务逻辑。

| # | 风险 | 影响 | 回顾要点 |
|---|---|---|---|
| R4 | triton-ascend 3.2.2 可能缺 step4 kernel 用到的 API（`parallel`、部分 `extension.*`） | 编译失败或行为差异 | 对照 R3；确认是版本问题而非逻辑问题 |
| R5 | torch 2.10.0 vs 2.13.0 的 API/行为差异 | 可能的 subtle 行为变化 | 确认是否 torch 版本导致 |
| R6 | CANN 9.2 + catlass `41bf90da65` 不匹配 | native extension 编译失败 | §7.1；若是，单独提 submodule bump MR 指向官方 `c89fe73` |
| R7 | `cann_ops_transformer` 未安装 | Step5 原生 DSA 路径不可用 | §8 的标记与 try/except |

### 10.3 待确认项（开发环境验证后定）

| # | 项 | 暂定 | 最终决策标准 |
|---|---|---|---|
| R8 | `vl.py` 的 `get_model_state_cls()` override：删 (a) 还是改为返回 `AscendModelState` (b) | **(b)** | NPU 上确认删除后 Step5 VL 仍走 Ascend v2 attention 路径；确认走则简化为 (a) |
| R9 | `patch_kv_cache_coordinator.py` replay ceiling 是否真不需要 | 不搬（D10） | NPU 上验证前缀缓存命中时的摘要正确性 |
| R10 | stepcast_plugin 的 `vllm.*` import 与官方 main 的兼容性 | 未知 | V9：`pip install ./stepcast_plugin && python -c "import stepcast_plugin"` |
| R11 | `_step5_compat.py` 的 `Step5SparseSummaryCacheConfig` 删除后，`_decode_summary_layout.py` 是否仍成立 | 待查 | L6 实现时确认 |

---

## 11. 不做的事

- 不带 `ci/` 目录（含 82MB `source.bundle`）
- 不保留原始 28 提交粒度
- 不动 `stepfun/v0.19.0` 和 private 本地 `main` ref
- 不给 vLLM 打自制 patch
- 不搬 `patch_bind_kv_cache.py` 的 side-storage 改动（D9）
- 不搬 `patch_kv_cache_coordinator.py` 的 replay ceiling（D10）
- 不搬 catlass submodule 指针改动（§7.1）

---

## 12. 参考

- 源分支：`upstream/step5-swf-journal-ascend-ops` @ `317c7fb99e`（107 文件，+46,405/−615）
- 目标分支：`step5/tailring-rebase` @ `399c30d3a`（已创建）
- vLLM 侧：官方 main @ 2026-10-20 + `stepcast_plugin`（out-of-tree）
- main 的 TailRing 参考实现：`vllm/models/step5/nvidia/{csa_tail_ring,sparse_summary_cache}.py`、
  `vllm/models/step5/nvidia/ops/summary_update.py`
