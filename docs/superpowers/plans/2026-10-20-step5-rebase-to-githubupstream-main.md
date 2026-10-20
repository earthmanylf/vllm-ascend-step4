# Step5 NPU 适配 rebase — 实现计划（本地可执行部分）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `upstream/step5-swf-journal-ascend-ops @ 317c7fb99e` 的 Step5 NPU 适配，分层重放到基于 `githubupstream/main` 的 `step5/tailring-rebase` 分支上，完成 L1–L5、L7、L8。

**Architecture:** 分层重建（方案 B）。每层独立可验证、独立可提交。L6（TailRing 摘要链路重写）拆到单独计划，因为它必须在 NPU 上迭代且代码量最大。本计划的所有任务均可在无 NPU 的开发机上执行并验证。

**Tech Stack:** Python 3.11/3.12、vllm-ascend、pytest、git。torch 2.10.0 / triton-ascend 3.2.2 / CANN 9.2.0 基线（全部跟随 main，不搬 step4 的 2.13.0 改动）。

**Spec:** `docs/superpowers/specs/2026-10-20-step5-rebase-to-githubupstream-main-design.md`

---

## 范围与不在本计划内

**本计划覆盖**：L1（纯新增）、L2（main 未动）、L3（低风险 port）、L4（中风险 port）、L5（高风险 port）、L7（multi_module_mtp + 共享 KV）、L8（基线收尾）。

**不在本计划内**：
- **L6 TailRing 重写**（`summary_cache.py` / `summary_kernels.py` / `dsa_attention.py` / `model_state.py` / `vl.py`）→ 单独计划，需 NPU
- NPU 数值验证 V1–V9 → 开发环境
- `ci/` 目录（82MB `source.bundle`）→ 不搬
- catlass submodule 指针 → 保持 `41bf90da65` 不动
- `patch_bind_kv_cache.py` 的 side-storage 改动 → 跳过（D9）
- `patch_kv_cache_coordinator.py` 的 replay ceiling → 不搬（D10）

---

## 关键约定（执行前必读）

1. **源分支引用**：所有"从源取文件"操作统一用
   `git show upstream/step5-swf-journal-ascend-ops:<path>`，**不要** `git checkout` 该分支
   （会切换工作区）。写入用 `git show ... > <path>`。
2. **提交作者**：本计划所有提交用 `git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit`。
3. **本机无 NPU**：torch 是 2.9.1+cu130，无 torch-npu。因此：
   - `import vllm_ascend` 级别的检查**做不了**（需要 torch_npu）
   - UT 用 `pytest --collect-only` 做收集检查 + 对 AST isolation 类测试做实际执行
   - 涉及 torch_npu 的 import 失败是**预期的**，用收集检查而非导入检查作为门槛
4. **每层结束必须验证通过才进下一层**，任何一层失败就停下报告，不要带着错误继续。

---

## 文件结构总览

```
新建（19 个代码文件）
  vllm_ascend/core/step5_kv_cache.py
  vllm_ascend/models/step5/*.py                    （16 个）
  vllm_ascend/worker/v2/spec_decode/multi_module_mtp/
      __init__.py + speculator.py

新建（6 个合并后 UT）
  tests/ut/models/test_step5_model_contract.py
  tests/ut/models/step5/test_step5_summary_contract.py     ← L6 计划重写
  tests/ut/models/step5/test_step5_dsa_contract.py
  tests/ut/models/step5/test_step5_mtp_contract.py
  tests/ut/quantization/test_step5_quant_contract.py
  tests/ut/worker/v2/test_step5_kv_contract.py

修改（L3/L4/L5 共 13 个已存在文件）
  vllm_ascend/envs.py
  vllm_ascend/models/__init__.py
  vllm_ascend/quantization/configs/modelslim_config.py
  vllm_ascend/quantization/methods/w8a8/fp8_block.py
  vllm_ascend/quantization/method_adapters.py
  vllm_ascend/quantization/quant_type.py
  vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py
  vllm_ascend/worker/v2/input_batch.py
  vllm_ascend/worker/v2/attn_utils.py
  vllm_ascend/worker/worker.py
  vllm_ascend/worker/v2/spec_decode/__init__.py
  vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py
  vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py
  vllm_ascend/worker/v2/model_runner.py
  vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py

保持 main 原样（D8）
  requirements.txt / pyproject.toml / CMakeLists.txt
```

---

## Task 1: 校验起点状态

**Files:** 无（只读检查）

- [ ] **Step 1: 确认分支与基线**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
git branch --show-current          # 期望: step5/tailring-rebase
git log --oneline -1               # 期望: b0ef5658e 或其后代
git rev-parse HEAD                 # 记录此值，后面所有 diff 以它为基准
```

- [ ] **Step 2: 确认远端引用可达**

```bash
git rev-parse upstream/step5-swf-journal-ascend-ops   # 期望: 317c7fb99e...
git rev-parse githubupstream/main                     # 期望: 399c30d3a...
```

Expected: 两个都返回完整 SHA。若 `upstream/step5-swf-journal-ascend-ops` 不存在，先
`git fetch upstream`。

- [ ] **Step 3: 确认 19 个目标文件在 main 上不存在**

```bash
for f in vllm_ascend/core/step5_kv_cache.py \
         vllm_ascend/models/step5/__init__.py \
         vllm_ascend/models/step5/model.py \
         vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py; do
  printf "%-64s " "$f"
  git cat-file -e "githubupstream/main:$f" 2>/dev/null && echo "已存在(冲突)" || echo "不存在(可新增)"
done
```

Expected: 全部输出"不存在(可新增)"。若有"已存在"，**停止并报告**。

---

## Task 2: L1 — 落地 19 个纯新增代码文件

**Files:**
- Create: 下列 19 个文件（内容取自 `upstream/step5-swf-journal-ascend-ops`）

- [ ] **Step 1: 写入核心模型文件（5 个）**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
S=upstream/step5-swf-journal-ascend-ops

mkdir -p vllm_ascend/models/step5
for f in __init__.py _decode_summary_layout.py _step5_compat.py model.py mtp.py; do
  git show "$S:vllm_ascend/models/step5/$f" > "vllm_ascend/models/step5/$f"
done
git show "$S:vllm_ascend/core/step5_kv_cache.py" > vllm_ascend/core/step5_kv_cache.py

wc -l vllm_ascend/models/step5/{__init__,_decode_summary_layout,_step5_compat,model,mtp}.py \
       vllm_ascend/core/step5_kv_cache.py
```

Expected 行数（与 spec §5.1 一致）：26 / 280 / 177 / 1665 / 178 / 159

- [ ] **Step 2: 写入 kernel 文件（4 个）**

```bash
S=upstream/step5-swf-journal-ascend-ops
for f in dsa_kernels.py fp8_kernels.py fused_add_rms_norm.py fused_qknorm_rope.py; do
  git show "$S:vllm_ascend/models/step5/$f" > "vllm_ascend/models/step5/$f"
done
wc -l vllm_ascend/models/step5/{dsa_kernels,fp8_kernels,fused_add_rms_norm,fused_qknorm_rope}.py
```

Expected: 4453 / 314 / 151 / 812

- [ ] **Step 3: 写入需外部依赖的 2 个文件并加保护标记**

```bash
S=upstream/step5-swf-journal-ascend-ops
git show "$S:vllm_ascend/models/step5/generic_block_sparse_attention.py" \
  > vllm_ascend/models/step5/generic_block_sparse_attention.py
git show "$S:vllm_ascend/models/step5/indexer_region_score.py" \
  > vllm_ascend/models/step5/indexer_region_score.py
```

然后编辑 `vllm_ascend/models/step5/generic_block_sparse_attention.py`，把
`from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (...)` 替换为：

```python
# REQUIRES: cann_ops_transformer (Ascend native sparse attention ops).
# Not in vllm-ascend requirements. Install from the StepFun wheelhouse or
# vendor under csrc/third_party/ before enabling Step5 native DSA.
try:
    from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )
except ImportError as exc:
    raise ImportError(
        "Step5 native DSA requires cann_ops_transformer; install it from the "
        "StepFun wheelhouse or vendor it under csrc/third_party/."
    ) from exc
```

同样处理 `vllm_ascend/models/step5/indexer_region_score.py` 的
`from cann_ops_transformer.ops.attention.indexer_region_score import ...`：

```python
# REQUIRES: cann_ops_transformer (Ascend native sparse attention ops).
# Not in vllm-ascend requirements. See models/step5/generic_block_sparse_attention.py.
try:
    from cann_ops_transformer.ops.attention.indexer_region_score import (
        indexer_region_score,
    )
except ImportError as exc:
    raise ImportError(
        "Step5 native DSA requires cann_ops_transformer; install it from the "
        "StepFun wheelhouse or vendor it under csrc/third_party/."
    ) from exc
```

- [ ] **Step 4: 写入 multi_module_mtp（2 个）**

```bash
S=upstream/step5-swf-journal-ascend-ops
mkdir -p vllm_ascend/worker/v2/spec_decode/multi_module_mtp
git show "$S:vllm_ascend/worker/v2/spec_decode/multi_module_mtp/__init__.py" \
  > vllm_ascend/worker/v2/spec_decode/multi_module_mtp/__init__.py
git show "$S:vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py" \
  > vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py
wc -l vllm_ascend/worker/v2/spec_decode/multi_module_mtp/*.py
```

Expected: 2 / 156

- [ ] **Step 5: 写入 SWF 版摘要链路（3 个，L6 将重写，先落地保证 import 结构完整）**

```bash
S=upstream/step5-swf-journal-ascend-ops
git show "$S:vllm_ascend/models/step5/dsa_attention.py" > vllm_ascend/models/step5/dsa_attention.py
git show "$S:vllm_ascend/models/step5/summary_cache.py" > vllm_ascend/models/step5/summary_cache.py
git show "$S:vllm_ascend/models/step5/summary_kernels.py" > vllm_ascend/models/step5/summary_kernels.py
git show "$S:vllm_ascend/models/step5/model_state.py" > vllm_ascend/models/step5/model_state.py
git show "$S:vllm_ascend/models/step5/vl.py" > vllm_ascend/models/step5/vl.py
wc -l vllm_ascend/models/step5/{dsa_attention,summary_cache,summary_kernels,model_state,vl}.py
```

Expected: 971 / 391 / 542 / 62 / 41

**这 5 个文件是 SWF 版，L6 计划会重写其中 4 个、删除 1 个。此处先落地是为了让
`vllm_ascend/models/step5/__init__.py` 的 import 结构完整，从而 L1–L5 可验证。**

- [ ] **Step 6: 语法检查全部新文件**

```bash
python3 -m py_compile vllm_ascend/models/step5/*.py \
                       vllm_ascend/core/step5_kv_cache.py \
                       vllm_ascend/worker/v2/spec_decode/multi_module_mtp/*.py
echo "exit=$?"
```

Expected: `exit=0`，无输出。

- [ ] **Step 7: 提交**

```bash
git add vllm_ascend/models/step5/ vllm_ascend/core/step5_kv_cache.py \
        vllm_ascend/worker/v2/spec_decode/multi_module_mtp/
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "feat(step5): bring Step5 NPU model code onto githubupstream/main base

19 new files from upstream/step5-swf-journal-ascend-ops @ 317c7fb99e:
models/step5 (16 files), core/step5_kv_cache.py, and the
multi_module_mtp speculator.

generic_block_sparse_attention.py and indexer_region_score.py get an
explicit REQUIRES marker plus try/except so the tree still imports
without cann_ops_transformer installed.

Note: summary_cache/summary_kernels/dsa_attention/model_state/vl are the
SWF-journal versions, landed as-is for import completeness. They are
rewritten to main's TailRing architecture in a follow-up (see design
spec section 6)."
```

---

## Task 3: L1 — 落地 6 个合并后 UT

**Files:**
- Create: 6 个 UT 文件（内容取自源分支的 19 个文件，按 spec §5.2 合并）

**合并原则**：保留每个源文件的 `import` 与模块级常量（去重），把测试函数原样搬入。
**必须保留 AST isolation 特性** —— 不得引入需要 torch_npu 的顶层 import。

- [ ] **Step 1: 建目录并准备源文件清单**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
mkdir -p tests/ut/models/step5
S=upstream/step5-swf-journal-ascend-ops
git ls-tree -r --name-only $S -- tests/ut | grep -i step5
```

Expected: 列出 19 个 step5 相关 UT 源文件路径。

- [ ] **Step 2: 生成 test_step5_model_contract.py**

合并来源（spec §5.2）：
- `tests/ut/models/test_step5_execution_contract.py`(573)
- `tests/ut/models/test_step5_fp8_config_contract.py`(94)
- `tests/ut/models/step5/test_step5_dummy_input_contract.py`(47)

```bash
S=upstream/step5-swf-journal-ascend-ops
OUT=tests/ut/models/test_step5_model_contract.py
{
  echo '"""Step5 model-level contracts: execution, fp8 config, dummy input.'
  echo ''
  echo 'Merged from test_step5_execution_contract.py,'
  echo 'test_step5_fp8_config_contract.py and test_step5_dummy_input_contract.py.'
  echo 'AST isolation: class definitions are compiled from source so these run'
  echo 'on CPU-only hosts as well as in the NPU unit-test suite.'
  echo '"""'
  echo ''
  # 取第一个文件的 import 块到第一个非 import 语句之前
  git show "$S:tests/ut/models/test_step5_execution_contract.py" \
    | sed -n '/^"""/,/^"""/p;1,/^"""/d;p' | sed -n '/^import \|^from /p' | sort -u
  echo ''
  # 依次追加各文件的测试函数（去掉文件头 docstring 与 import）
  for f in tests/ut/models/test_step5_execution_contract.py \
           tests/ut/models/test_step5_fp8_config_contract.py \
           tests/ut/models/step5/test_step5_dummy_input_contract.py; do
    echo "# --- from $f ---"
    git show "$S:$f" | sed '1,/^"""/d' | sed '/^import \|^from /d' | sed '/^$/N;/^\n$/D'
    echo ''
  done
} > "$OUT"
python3 -m py_compile "$OUT" && echo "OK $OUT"
```

Expected: `OK tests/ut/models/test_step5_model_contract.py`

**若 py_compile 失败**（常见于 import 被误删），改为手工方式：分别 `git show` 三个文件，
手工拼接 import 与函数体，确保无重复 import、无缺失 import。

- [ ] **Step 3: 生成 test_step5_dsa_contract.py**

合并来源：`test_native_dsa_contract.py`(409) + `test_ar_graph_step_contract.py`(379)

```bash
S=upstream/step5-swf-journal-ascend-ops
OUT=tests/ut/models/step5/test_step5_dsa_contract.py
{
  echo '"""Step5 DSA contracts: native-op adapters and aclgraph step control.'
  echo ''
  echo 'Merged from test_native_dsa_contract.py and test_ar_graph_step_contract.py.'
  echo 'Custom kernels are replaced only at their API boundary; these tests do'
  echo 'not claim device arithmetic or AICPU scheduling parity.'
  echo '"""'
  echo ''
  git show "$S:tests/ut/models/step5/test_native_dsa_contract.py" \
    | sed '1,/^"""/d' | sed -n '/^import \|^from /p' | sort -u
  git show "$S:tests/ut/worker/v2/test_ar_graph_step_contract.py" \
    | sed '1,/^"""/d' | sed -n '/^import \|^from /p' | sort -u
  echo ''
  for f in tests/ut/models/step5/test_native_dsa_contract.py \
           tests/ut/worker/v2/test_ar_graph_step_contract.py; do
    echo "# --- from $f ---"
    git show "$S:$f" | sed '1,/^"""/d' | sed '/^import \|^from /d' | sed '/^$/N;/^\n$/D'
    echo ''
  done
} > "$OUT"
python3 -m py_compile "$OUT" && echo "OK $OUT"
```

- [ ] **Step 4: 生成 test_step5_mtp_contract.py**

合并来源：`test_multi_module_mtp_contract.py`(394) + `test_mtp_quantization_contract.py`(258)

```bash
S=upstream/step5-swf-journal-ascend-ops
OUT=tests/ut/models/step5/test_step5_mtp_contract.py
{
  echo '"""Step5 MTP contracts: multi-module speculator and draft quantization.'
  echo ''
  echo 'Merged from test_multi_module_mtp_contract.py and'
  echo 'test_mtp_quantization_contract.py. Exact production methods run with'
  echo 'CPU tensors, without an NPU import.'
  echo '"""'
  echo ''
  git show "$S:tests/ut/worker/v2/test_multi_module_mtp_contract.py" \
    | sed '1,/^"""/d' | sed -n '/^import \|^from /p' | sort -u
  git show "$S:tests/ut/models/step5/test_mtp_quantization_contract.py" \
    | sed '1,/^"""/d' | sed -n '/^import \|^from /p' | sort -u
  echo ''
  for f in tests/ut/worker/v2/test_multi_module_mtp_contract.py \
           tests/ut/models/step5/test_mtp_quantization_contract.py; do
    echo "# --- from $f ---"
    git show "$S:$f" | sed '1,/^"""/d' | sed '/^import \|^from /d' | sed '/^$/N;/^\n$/D'
    echo ''
  done
} > "$OUT"
python3 -m py_compile "$OUT" && echo "OK $OUT"
```

- [ ] **Step 5: 生成 test_step5_quant_contract.py**

合并来源：`test_fp8_moe_contract.py`(402) + `test_step5_vlm_loading.py`(126) +
`test_step5_modelslim_loading.py`(107) + `test_step5_modelslim_mapping.py`(78) +
`test_mxfp8_dummy_scales.py`(56)

```bash
S=upstream/step5-swf-journal-ascend-ops
OUT=tests/ut/quantization/test_step5_quant_contract.py
{
  echo '"""Step5 quantization contracts: fp8 MoE, ModelSlim/VLM loading, mapping.'
  echo ''
  echo 'Merged from test_fp8_moe_contract.py, test_step5_vlm_loading.py,'
  echo 'test_step5_modelslim_loading.py, test_step5_modelslim_mapping.py and'
  echo 'test_mxfp8_dummy_scales.py. AST isolation: real builders run on CPU.'
  echo '"""'
  echo ''
  for f in tests/ut/models/step5/test_fp8_moe_contract.py \
           tests/ut/quantization/test_step5_vlm_loading.py \
           tests/ut/quantization/test_step5_modelslim_loading.py \
           tests/ut/quantization/configs/test_step5_modelslim_mapping.py \
           tests/ut/quantization/test_mxfp8_dummy_scales.py; do
    git show "$S:$f" | sed '1,/^"""/d' | sed -n '/^import \|^from /p'
  done | sort -u
  echo ''
  for f in tests/ut/models/step5/test_fp8_moe_contract.py \
           tests/ut/quantization/test_step5_vlm_loading.py \
           tests/ut/quantization/test_step5_modelslim_loading.py \
           tests/ut/quantization/configs/test_step5_modelslim_mapping.py \
           tests/ut/quantization/test_mxfp8_dummy_scales.py; do
    echo "# --- from $f ---"
    git show "$S:$f" | sed '1,/^"""/d' | sed '/^import \|^from /d' | sed '/^$/N;/^\n$/D'
    echo ''
  done
} > "$OUT"
python3 -m py_compile "$OUT" && echo "OK $OUT"
```

- [ ] **Step 6: 生成 test_step5_kv_contract.py（单文件，原样搬）**

```bash
S=upstream/step5-swf-journal-ascend-ops
git show "$S:tests/ut/worker/v2/test_step5_shared_kv.py" \
  > tests/ut/worker/v2/test_step5_kv_contract.py
python3 -m py_compile tests/ut/worker/v2/test_step5_kv_contract.py && echo "OK"
```

Expected: `OK`

- [ ] **Step 7: 生成 test_step5_summary_contract.py（占位，L6 重写）**

**spec 明确：这三个来源测的是 SWF journal，必须随 L6 按 TailRing 重写，不能直接搬。**

```bash
S=upstream/step5-swf-journal-ascend-ops
cat > tests/ut/models/step5/test_step5_summary_contract.py <<'EOF'
"""Step5 summary-cache contracts (TailRing architecture).

PLACEHOLDER — to be written by the L6 TailRing rewrite plan.

The SWF-journal sources (test_summary_cache_contract.py,
test_summary_update_dispatch_contract.py, test_summary_scratch_contract.py)
test the request-journal undo log, which vLLM main removed in favour of
Step5CSATailRing. They must be rewritten against main's
Step5SparseSummaryCache(mean) / update_decode / update_verify /
update_prefill API, not copied.

See design spec section 6.
"""
EOF
python3 -m py_compile tests/ut/models/step5/test_step5_summary_contract.py && echo "OK"
```

**这是一个有意的、spec 授权的占位**（区别于 writing-plans 禁止的"TODO 式"占位）：
它明确说明为什么空、要写成什么样、由哪个计划写。

- [ ] **Step 8: 收集检查全部 6 个 UT**

```bash
python3 -m pytest tests/ut/models/test_step5_model_contract.py \
                 tests/ut/models/step5/test_step5_dsa_contract.py \
                 tests/ut/models/step5/test_step5_mtp_contract.py \
                 tests/ut/quantization/test_step5_quant_contract.py \
                 tests/ut/worker/v2/test_step5_kv_contract.py \
                 tests/ut/models/step5/test_step5_summary_contract.py \
                 --collect-only -q 2>&1 | tail -20
```

Expected: 收集成功，显示测试数量，无 `ERROR`。
**注意**：若报 `ModuleNotFoundError: torch_npu`，这是本机预期（见关键约定 3），
改为检查收集阶段是否已越过 import —— 用 `--collect-only` 的输出判断是否有
`errors during collection`。

- [ ] **Step 9: 提交**

```bash
git add tests/ut/
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "test(step5): consolidate 19 Step5 UTs into 6 merged files

Merged per design spec section 5.2, preserving the AST-isolation style
that lets them run on CPU-only hosts:
- test_step5_model_contract.py     <- execution + fp8 config + dummy input
- test_step5_dsa_contract.py       <- native DSA + aclgraph step
- test_step5_mtp_contract.py       <- multi-module MTP + draft quantization
- test_step5_quant_contract.py     <- fp8 MoE + ModelSlim/VLM loading
- test_step5_kv_contract.py        <- shared KV (single source)
- test_step5_summary_contract.py   <- placeholder, L6 rewrites it

4194 lines / 6 files, down from 4940 / 19."
```

---

## Task 4: L2 — 落地 6 个 main 未动文件

**Files:**
- Modify: 6 个已存在文件（step4 的改动能干净应用）

清单（spec §2.4）：`quantization/configs/fp8_config.py`、`quantization/method_adapters.py`、
`quantization/methods/w8a8/fp8_block.py`、`quantization/quant_type.py`、
`worker/v2/spec_decode/lmhead_tp_utils.py`、`.gitmodules`（**跳过**，随 catlass 不搬保持原样）。

- [ ] **Step 1: 逐个应用 step4 的改动**

对每个文件，用 `git diff` 取出 step4 相对其 base 的 patch，然后 `git apply` 到当前分支。

```bash
cd /data/LY/code/work-code/vllm-ascend-private
S=upstream/step5-swf-journal-ascend-ops
B=5f828ad638   # step4 的 base

for f in vllm_ascend/quantization/configs/fp8_config.py \
         vllm_ascend/quantization/method_adapters.py \
         vllm_ascend/quantization/methods/w8a8/fp8_block.py \
         vllm_ascend/quantization/quant_type.py \
         vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py; do
  echo "=== $f ==="
  git diff "$B" "$S" -- "$f" > /tmp/one.patch
  if git apply --check /tmp/one.patch 2>/dev/null; then
    git apply /tmp/one.patch && echo "  applied"
  else
    echo "  !! git apply 失败，需人工处理"
  fi
done
```

Expected: 5 个都输出 `applied`。

**若某个失败**（例如 `fp8_block.py` 因 main 已演进）：不要强行 apply。
改为 `git diff "$B" "$S" -- "$f" > /tmp/one.patch` 后 `git apply -3` 或人工编辑，
并在提交信息里说明。

- [ ] **Step 2: 语法检查**

```bash
python3 -m py_compile vllm_ascend/quantization/configs/fp8_config.py \
                       vllm_ascend/quantization/method_adapters.py \
                       vllm_ascend/quantization/methods/w8a8/fp8_block.py \
                       vllm_ascend/quantization/quant_type.py \
                       vllm_ascend/worker/v2/spec_decode/lmhead_tp_utils.py
echo "exit=$?"
```

Expected: `exit=0`

- [ ] **Step 3: 确认未误改 .gitmodules 与 catlass**

```bash
git diff --stat -- .gitmodules csrc/third_party/catlass
```

Expected: **无输出**（两者都应保持 main 原样，即 catlass 仍指 `41bf90da65`）。
若有输出，`git checkout -- .gitmodules csrc/third_party/catlass` 还原。

- [ ] **Step 4: 提交**

```bash
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "feat(quantization): apply Step5 fp8-block and ModelSlim glue

Five files whose base-to-main content is unchanged, so step4's patches
apply cleanly: fp8_config.py (Step5Fp8BlockConfig), method_adapters.py
(dummy scale metadata), fp8_block.py (Step5Fp8BlockFusedMoEMethod),
quant_type.py (FP8_BLOCK), lmhead_tp_utils.py.

.gitmodules and the catlass submodule are deliberately untouched (design
spec section 7.1): step4's pin to fork commit 038b9315f3 is not portable."
```

---

## Task 5: L3 — 低风险 port（6 个文件）

**Files:**
- Modify: `vllm_ascend/envs.py`、`vllm_ascend/models/__init__.py`、
  `vllm_ascend/quantization/configs/modelslim_config.py`、
  `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`
- 参考：`quant_type.py` 已在 Task 4 落地

- [ ] **Step 1: port envs.py（+1 env var）**

`git diff 5f828ad638 upstream/step5-swf-journal-ascend-ops -- vllm_ascend/envs.py` 显示只加 3 行：

```python
    # Whether to run Step5 routed experts with the native fp8 block kernels
    # (Step5Fp8BlockFusedMoEMethod). Default 0 -> AscendFp8BlockFusedMoEMethod. Valid "0"/"1".
    "VLLM_ASCEND_ENABLE_STEP4_FP8": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_STEP4_FP8", "0"))),
```

**同时把变量名从 `STEP4` 改为 `STEP5`**（spec 未列此项，但名字带 STEP4 实际是 step5 语义，
属于"改进你正在改的代码"）：

```python
    # Whether to run Step5 routed experts with the native fp8 block kernels
    # (Step5Fp8BlockFusedMoEMethod). Default 0 -> AscendFp8BlockFusedMoEMethod. Valid "0"/"1".
    "VLLM_ASCEND_ENABLE_STEP5_FP8": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_STEP5_FP8", "0"))),
```

然后全局替换代码里的引用：

```bash
grep -rn "VLLM_ASCEND_ENABLE_STEP4_FP8" vllm_ascend/ | grep -v ".pyc"
```

Expected: 只应在 `vllm_ascend/models/step5/` 下出现。若有，逐一改为 `STEP5`。

- [ ] **Step 2: port models/__init__.py（注册 3 个模型）**

step4 在此文件加了 3 个 `ModelRegistry.register_model` 调用。main 上该文件已 +11 行
（新增了 Gemma4 注册）。用 patch 方式：

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/models/__init__.py > /tmp/m.patch
git apply --check /tmp/m.patch 2>/dev/null && git apply /tmp/m.patch && echo "applied" \
  || echo "!! 需人工：把 3 个 register_model 调用插入 main 版本的正确位置"
```

step4 加入的三行（若需手工插入，按 main 现有的注册顺序插入）：

```python
    ModelRegistry.register_model(
        "Step5ForCausalLM",
        "vllm_ascend.models.step5:AscendStep5ForCausalLM",
    )
    ModelRegistry.register_model(
        "Step5ForConditionalGeneration",
        "vllm_ascend.models.step5.vl:AscendStep5ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Step5MTP",
        "vllm_ascend/models.step5.mtp:AscendStep5MTP",
    )
```

- [ ] **Step 3: port modelslim_config.py（packed mapping）**

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/quantization/configs/modelslim_config.py > /tmp/ms.patch
git apply --check /tmp/ms.patch 2>/dev/null && git apply /tmp/ms.patch && echo "applied" \
  || echo "!! 需人工：把 step5/step5_vl 两条 UPDATED_PACKED_MODULES_MAPPING 插入 main 版本"
```

step4 加入的内容（若需手工插入 `UPDATED_PACKED_MODULES_MAPPING` dict）：

```python
    # Step5 ModelSlim checkpoints pack all experts into three sibling 3-D
    # projections instead of individual experts.<id>.* parameters.
    "step5": {
        "experts": ["gate_proj", "up_proj", "down_proj"],
    },
    "step5_vl": {
        "experts": ["gate_proj", "up_proj", "down_proj"],
    },
```

- [ ] **Step 4: port sparse_kv_offload_manager.py（懒加载 memfabric）**

step4 把模块级 `from memfabric_hybrid import offload`（包在 `contextlib.suppress` 里）
改为函数 `_load_offload_backend()`。**核实过 main 仍是模块级导入**（第 15 行），
所以这个改进有效，要 port。

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py > /tmp/sk.patch
git apply --check /tmp/sk.patch 2>/dev/null && git apply /tmp/sk.patch && echo "applied" \
  || echo "!! main 已重构该文件(-320行)，需人工把 _load_offload_backend 思想搬过去"
```

**若需人工**：在 main 版本里加（放在 import 区之后）：

```python
def _load_offload_backend() -> typing.Any:
    # Importing MemFabric registers an AICPU package with the runtime. Keep
    # that side effect out of workers that never enable sparse KV offload.
    try:
        from memfabric_hybrid import offload  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "memfabric_hybrid":
            raise
        raise RuntimeError("Sparse KV offload requires memfabric_hybrid with offload support.") from exc
    return offload
```

然后把 main 里所有裸 `offload.` 引用改为 `_load_offload_backend().`，
并删掉 main 顶部的 `from memfabric_hybrid import offload` 及其 `contextlib.suppress` 包裹。

- [ ] **Step 5: 语法检查 + 提交**

```bash
python3 -m py_compile vllm_ascend/envs.py vllm_ascend/models/__init__.py \
    vllm_ascend/quantization/configs/modelslim_config.py \
    vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py
echo "exit=$?"
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "feat(step5): register Step5 models and wire quantization glue

- envs.py: VLLM_ASCEND_ENABLE_STEP5_FP8 (renamed from STEP4, same semantics)
- models/__init__.py: register Step5ForCausalLM / Step5ForConditionalGeneration / Step5MTP
- modelslim_config.py: packed_modules_mapping for step5 and step5_vl
- sparse_kv_offload_manager.py: lazy _load_offload_backend() so MemFabric's
  AICPU provisioning stays out of workers that never enable sparse offload
  (main still imports it eagerly at module scope)"
```

---

## Task 6: L4 — 中风险 port（5 个文件）

**Files:**
- Modify: `vllm_ascend/worker/v2/input_batch.py`、`vllm_ascend/worker/v2/aclgraph_utils.py`、
  `vllm_ascend/worker/v2/attn_utils.py`、`vllm_ascend/worker/worker.py`、
  `vllm_ascend/worker/v2/spec_decode/__init__.py`

- [ ] **Step 1: port input_batch.py（dummy token padding 语义）**

step4 改动 +4 行，来自 `a096cfcb5 fix(v2): preserve SWF dummy token padding semantics`。
main 已 +39/−1。用 patch 试应用：

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/worker/v2/input_batch.py > /tmp/ib.patch
git apply --check /tmp/ib.patch 2>/dev/null && git apply /tmp/ib.patch && echo "applied" \
  || { echo "!! 需人工，先看冲突："; git diff 5f828ad638 githubupstream/main -- vllm_ascend/worker/v2/input_batch.py | head -40; }
```

step4 的改动本质是让 `make_dummy` 接受并传递 `is_padding` 参数
（对应 vLLM main 已有的 `make_dummy(..., is_padding: bool = True)` 签名）。
若需人工：确认 main 的 `AscendInputBatch.make_dummy` 是否已暴露 `is_padding`，
按 step4 的方式补齐。

- [ ] **Step 2: port aclgraph_utils.py**

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/worker/v2/aclgraph_utils.py > /tmp/ag.patch
git apply --check /tmp/ag.patch 2>/dev/null && git apply /tmp/ag.patch && echo "applied" \
  || echo "!! 需人工（main +25/-11，step4 +23）"
```

- [ ] **Step 3: port attn_utils.py**

step4 只 +6 行（来自 `7ae69ef36` 的共享 KV backing），main 已 +194/−41。

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/worker/v2/attn_utils.py > /tmp/au.patch
git apply --check /tmp/au.patch 2>/dev/null && git apply /tmp/au.patch && echo "applied" \
  || echo "!! 需人工"
```

- [ ] **Step 4: port worker.py**

step4 +7 行，main +129/−34。

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/worker/worker.py > /tmp/w.patch
git apply --check /tmp/w.patch 2>/dev/null && git apply /tmp/w.patch && echo "applied" \
  || echo "!! 需人工"
```

- [ ] **Step 5: port spec_decode/__init__.py**

step4 +12 行（注册 multi_module_mtp），main +7/−1。

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/worker/v2/spec_decode/__init__.py > /tmp/sd.patch
git apply --check /tmp/sd.patch 2>/dev/null && git apply /tmp/sd.patch && echo "applied" \
  || echo "!! 需人工：把 multi_module_mtp 注册插入 main 版本"
```

- [ ] **Step 6: 语法检查 + 收集检查 + 提交**

```bash
python3 -m py_compile vllm_ascend/worker/v2/input_batch.py \
    vllm_ascend/worker/v2/aclgraph_utils.py vllm_ascend/worker/v2/attn_utils.py \
    vllm_ascend/worker/worker.py vllm_ascend/worker/v2/spec_decode/__init__.py
echo "exit=$?"
python3 -m pytest tests/ut/worker/v2/ --collect-only -q 2>&1 | tail -5
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "feat(worker): port Step5 v2 input batch and graph utilities

input_batch.py (dummy padding semantics), aclgraph_utils.py, attn_utils.py
(shared KV backing), worker.py and the multi_module_mtp spec_decode
registration. main has moved these files substantially (+298/-89 in
model_runner, +194/-41 in attn_utils), so conflicts were resolved by
porting step4's intent onto main's structure rather than applying patches
blindly."
```

---

## Task 7: L5 — 高风险 port：model_runner.py

**Files:**
- Modify: `vllm_ascend/worker/v2/model_runner.py`（main 已 +298/−89，step4 仅 +17）

- [ ] **Step 1: 看清双方差异再动手**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
echo "=== step4 改了什么（相对其 base）==="
git diff 5f828ad638 upstream/step5-swf-journal-ascend-ops -- vllm_ascend/worker/v2/model_runner.py
echo "=== main 改了什么（相对同一 base）==="
git diff 5f828ad638 githubupstream/main -- vllm_ascend/worker/v2/model_runner.py | head -120
```

**判断标准**：step4 的 +17 行是否落在 main 已删除/重写的区域？
- 若 step4 的改动区域在 main 中仍存在且未变 → patch 可应用
- 若在 main 中已被重构 → 按 step4 的意图在 main 的新结构里重新实现

- [ ] **Step 2: 尝试 patch，失败则人工**

```bash
S=upstream/step5-swf-journal-ascend-ops
git diff 5f828ad638 "$S" -- vllm_ascend/worker/v2/model_runner.py > /tmp/mr.patch
git apply --check /tmp/mr.patch 2>/dev/null && git apply /tmp/mr.patch && echo "applied" \
  || echo "!! 需人工：按 step4 意图在 main 新结构上实现"
```

- [ ] **Step 3: 语法检查 + 提交**

```bash
python3 -m py_compile vllm_ascend/worker/v2/model_runner.py && echo "exit=$?"
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "feat(worker): port Step5 model_runner changes onto main's v2 runner

main has rewritten this file heavily (+298/-89). step4's change is small
(+17) and was ported by intent onto main's structure."
```

---

## Task 8: L5 — 高风险 port：speculator.py（人工重写）

**Files:**
- Modify: `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py`

**这是全场最难冲突**：三方改动叠加（main 的 `vllm_version_is()` 双路径、main 的 PCP
`is_decode_sharded` 分支、step4 的 307 行删除重构）。

- [ ] **Step 1: 完整读取三个版本**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
echo "=== A: step4 base 版（5f828ad638）行数 ==="
git show 5f828ad638:vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py | wc -l
echo "=== B: step4 分支版 ==="
git show upstream/step5-swf-journal-ascend-ops:vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py | wc -l
echo "=== C: main 版 ==="
git show githubupstream/main:vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py | wc -l
```

把三个版本都读到文件里对比：

```bash
mkdir -p /tmp/spec3
git show 5f828ad638:vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py > /tmp/spec3/A_base.py
git show upstream/step5-swf-journal-ascend-ops:vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py > /tmp/spec3/B_step4.py
git show githubupstream/main:vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py > /tmp/spec3/C_main.py
diff -u /tmp/spec3/C_main.py /tmp/spec3/B_step4.py | head -150
```

- [ ] **Step 2: 确认 main 版的 vLLM import 分叉**

在 `C_main.py` 中应能看到（spec §7.2）：

```python
from vllm_ascend.utils import vllm_version_is
if vllm_version_is("0.31.0"):
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
else:
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator as AutoRegressiveSpeculator,
    )
```

**我们的 vLLM 是官方 main（非 v0.31.0 release），所以目标是走 `else` 分支。**
确认这段在 main 版里已存在且正确，不要改动它。

- [ ] **Step 3: 判断 step4 的 307 行删除是否需要保留**

step4（`fc9c5badb`）把该文件重写成基于 vLLM 上游 `MultiModuleMTPSpeculator` 的薄适配层。
核实 step4 版是否还把 `AscendMultiModuleMTPSpeculator` 定义在这里：

```bash
grep -n "class .*Speculator\|AscendMultiModuleMTPSpeculator" /tmp/spec3/B_step4.py
```

Expected: 若 step4 已把 `AscendMultiModuleMTPSpeculator` 移到
`multi_module_mtp/speculator.py`（Task 2 已落地），则此文件里的类应已删除，
剩下的差异就是 main 的 PCP 分支等。**若 B_step4 里已无 Speculator 类定义，
则本文件很可能只需保留 main 版 + 少量 step4 补充，冲突远小于预期。**

- [ ] **Step 4: 按判断结果处理**

**情况 1（B_step4 已删除类定义）**：main 版基本就是目标，只需检查 step4 是否还有
遗漏的辅助函数：

```bash
diff <(grep -oP "^(def|class) \K\w+" /tmp/spec3/C_main.py | sort -u) \
     <(grep -oP "^(def|class) \K\w+" /tmp/spec3/B_step4.py | sort -u)
```

Expected: 列出 step4 有而 main 没有的名字。对每个名字，判断是否需要补入 main 版。

**情况 2（B_step4 仍有类定义）**：以 main 版为基，把 step4 的
`AscendMultiModuleMTPSpeculator` 相关改动按 main 的新结构（`else` 分支 import、
`is_decode_sharded` 分支）重新实现。

- [ ] **Step 5: 语法检查 + 提交**

```bash
python3 -m py_compile vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py && echo "exit=$?"
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "refactor(spec-decode): reconcile Ascend speculator with main's v2 structure

Three-way conflict: main's vllm_version_is() dual-path import for the
autoregressive -> target_dependent_ar rename (vLLM #60335), main's new
PCP is_decode_sharded branch, and step4's 307-line deletion refactor that
moved AscendMultiModuleMTPSpeculator to multi_module_mtp/speculator.py.

Resolved by keeping main's structure and porting step4's intent onto it."
```

---

## Task 9: L7 — multi_module_mtp 接口对齐

**Files:**
- Modify: `vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py`（Task 2 已落地初版）

- [ ] **Step 1: 核实 vLLM 侧基类接口**

```bash
cd /data/LY/code/work-code/vllm-private
git show githubupstream/main:vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py \
  | grep -n "^class \|    def " | head -20
```

Expected 看到：`MultiModuleMTPSpeculator(DraftModelSpeculator)` 及
`__init__ / load_draft_model / load_model / init_cudagraph_manager / capture /
propose / compute_draft_logits / get_draft_top_tokens / _run_model /
_prepare_inputs / _generate_drafts`。

- [ ] **Step 2: 检查 step4 版是否调用了官方 main 没有的 API**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
grep -n "attn_vllm_config" vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py
```

Expected: **无输出**。`attn_vllm_config()` 是 stepcast fork 独有 property，官方 main 没有。
若有调用，改为从 `self.vllm_config` 或等价路径取 attention 配置，并在提交信息说明。

- [ ] **Step 3: 检查 import 的 vLLM 路径是否需按版本分叉**

```bash
grep -n "^from vllm\|^import vllm\|vllm_version_is" \
    vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py
```

Expected: import 的 vLLM 路径在官方 main 上存在。若引用了 fork 独有路径，
按 Task 8 Step 2 的 `vllm_version_is()` 模式加分叉。

- [ ] **Step 4: 语法检查 + MTP UT 收集**

```bash
python3 -m py_compile vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py && echo "exit=$?"
python3 -m pytest tests/ut/models/step5/test_step5_mtp_contract.py --collect-only -q 2>&1 | tail -5
```

- [ ] **Step 5: 提交**

```bash
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "fix(spec-decode): align Ascend multi-module MTP with official vLLM main

MultiModuleMTPSpeculator exists on official main
(multi_module_mtp/speculator.py:28, DraftModelSpeculator subclass), so the
Ascend adapter has a valid base. Verified no reference to the fork-only
attn_vllm_config property, which official main does not provide."
```

---

## Task 10: L8 — 基线收尾与全量静态检查

**Files:**
- Verify only: `requirements.txt`、`pyproject.toml`、`CMakeLists.txt`、`.gitmodules`

- [ ] **Step 1: 确认三个构建文件未被改动（D8）**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
git diff githubupstream/main -- requirements.txt pyproject.toml CMakeLists.txt
```

Expected: **无输出**。D8 要求全部优先 main，step4 的 2.13.0/triton 3.6 改动一概不搬。
若有输出，`git checkout githubupstream/main -- <file>` 还原并说明原因。

- [ ] **Step 2: 确认 catlass 与 .gitmodules 未被改动**

```bash
git diff githubupstream/main -- .gitmodules csrc/third_party/catlass
```

Expected: **无输出**（catlass 仍指 `41bf90da65`）。

- [ ] **Step 3: 确认 dsa_kernels.py 的 parallel 用法已评估**

```bash
grep -n "extension.parallel" vllm_ascend/models/step5/dsa_kernels.py
```

Expected: 命中 2 处（约 2805、3088 行）。这两处依赖
`tl.extra.cann.extension.parallel`，其在 triton-ascend 3.2.2 上可用性未知（R3）。

**处置（spec §8）**：加 `hasattr` 守卫 + 回落 `tl.range`。在
`vllm_ascend/models/step5/dsa_kernels.py` 顶部 import 区之后加：

```python
# triton-ascend 3.2.2 may not expose extension.parallel (main never uses it).
# Fall back to a plain range loop when it is absent.
try:
    _cann_parallel = tl.extra.cann.extension.parallel
except AttributeError:  # pragma: no cover - depends on triton-ascend build
    def _cann_parallel(start, stop):
        return range(start, stop)
```

然后把两处 `tl.extra.cann.extension.parallel(0, BLOCK_R)` 改为 `_cann_parallel(0, BLOCK_R)`：

```bash
sed -i 's/tl\.extra\.cann\.extension\.parallel(/_cann_parallel(/g' \
    vllm_ascend/models/step5/dsa_kernels.py
grep -n "_cann_parallel\|extension.parallel" vllm_ascend/models/step5/dsa_kernels.py
```

Expected: 只剩 `_cann_parallel` 的 3 处（1 处定义 + 2 处调用），无 `extension.parallel` 残留。

**注意**：`tl` 必须已在该文件 import。若文件用的是 `from vllm.triton_utils import tl`，
上面的 try/except 要放在该 import 之后。

- [ ] **Step 4: 全量语法检查**

```bash
python3 -m compileall -q vllm_ascend/ 2>&1 | head -20
echo "exit=$?"
```

Expected: `exit=0`。若有语法错，定位并修复。

- [ ] **Step 5: 全量 UT 收集**

```bash
python3 -m pytest tests/ut/ --collect-only -q 2>&1 | tail -15
```

Expected: 收集完成。记录收集到的测试总数与 error 数。
**注意**：`ModuleNotFoundError: torch_npu` 类的收集错误在本机是预期的；
关键是 step5 相关 UT（Task 3 的 6 个文件）能被收集。

- [ ] **Step 6: 提交（若有改动）**

```bash
git status --short
git add -u
git -c user.name="liuyi" -c user.email="liuyi@stepfun.com" commit -m "build(step5): guard triton-ascend extension.parallel and finalize baseline

extension.parallel is not used anywhere on main, so its availability in
triton-ascend 3.2.2 is unverified. Added a hasattr guard with a plain
range() fallback for the two call sites in dsa_kernels.py.

requirements.txt, pyproject.toml, CMakeLists.txt and the catlass submodule
are confirmed untouched per D8 and design spec section 7.1."
```

**若 Step 3–5 无任何改动，跳过本提交。**

---

## Task 11: 收尾检查与交接

**Files:** 无（只读检查）

- [ ] **Step 1: 生成变更总览**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
echo "=== 相对 githubupstream/main 的提交 ==="
git log --oneline githubupstream/main..HEAD
echo
echo "=== 变更文件统计 ==="
git diff --stat githubupstream/main HEAD | tail -5
echo
echo "=== 确认未触碰的黑名单 ==="
git diff --name-only githubupstream/main HEAD | grep -E "^(ci/|requirements.txt|pyproject.toml|CMakeLists.txt|\.gitmodules|csrc/third_party/catlass)" && echo "!! 有黑名单文件被改" || echo "OK: 黑名单未被触碰"
```

Expected: 黑名单检查输出 `OK`。

- [ ] **Step 2: 对照 spec 决策逐条核对**

```bash
for d in D1 D2 D3 D4 D5 D6 D7 D8 D9 D10; do
  printf "%s: " "$d"
done
echo "逐条对照 docs/superpowers/specs/2026-10-20-step5-rebase-to-githubupstream-main-design.md 第 3 节"
```

人工核对表：

| 决策 | 核对点 | 状态 |
|---|---|---|
| D1 vLLM 官方 main + plugin | 本计划不改 vLLM | □ |
| D2 TailRing 重写 | **L6 单独计划，不在本计划** | □ |
| D3 torch 2.10.0 + triton-ascend 3.2.2 | requirements 未改（Task 10 Step 1） | □ |
| D4 private 新分支 | 当前在 `step5/tailring-rebase` | □ |
| D5 分层重建 | L1–L8 分任务提交 | □ |
| D6 ci/ 不搬 | 无 `ci/` 变更 | □ |
| D7 cann_ops_transformer 标记 | Task 2 Step 3 已加 | □ |
| D8 依赖优先 main | requirements/pyproject/CMake 未改 | □ |
| D9 跳过 patch_bind_kv_cache | Task 4 未含该文件 | □ |
| D10 不搬 replay ceiling | 无 `patch_kv_cache_coordinator.py` 变更 | □ |

- [ ] **Step 3: 记录遗留项**

在 MR 描述或提交信息中记录：

```
遗留（需 NPU / 开发环境）：
- L6 TailRing 重写：summary_cache.py / summary_kernels.py / dsa_attention.py
  / model_state.py / vl.py  —— 当前落地的是 SWF 版，功能不完整
- tests/ut/models/step5/test_step5_summary_contract.py 是 spec 授权的占位
- V1–V9 NPU 验证（见 spec §9.2）
- R8: vl.py 的 get_model_state_cls() override 最终形态
- R10: stepcast_plugin 与官方 vLLM main 的 import 兼容性
```

- [ ] **Step 4: 提交计划完成标记**

```bash
git status --short
```

Expected: 工作区干净（除既有的未跟踪文件 `.dockerignore`、`ci-fast/`、
`csrc/build.stale_ninja_*/`、`docs/refactor/`、`extra-info/`，这些与本计划无关，
**不要提交它们**）。

---

## Self-Review 记录

**1. Spec 覆盖核对**

| Spec 章节 | 对应任务 |
|---|---|
| §2.4 冲突面分类 | Task 1 Step 3、Task 2–7 的分层 |
| §3 D1–D10 | Task 11 Step 2 核对表 |
| §4 分层拓扑 | Task 2(L1)/4(L2)/5(L3)/6(L4)/7,8(L5)/9(L7)/10(L8) |
| §5.1 19 个代码文件 | Task 2 Steps 1–5 |
| §5.2 UT 19→6 合并 | Task 3 Steps 2–7 |
| §5.3 UT 其他处置 | Task 3（side_storage 删除 = 不搬；lazy_import 并入 main 已有测试） |
| §6 L6 TailRing | **不在本计划** → 单独计划 |
| §7.1 catlass | Task 4 Step 3、Task 10 Step 2 |
| §7.2 speculator.py | Task 8 |
| §7.3 multi_module_mtp | Task 9 |
| §7.4 其他 L4/L5 文件 | Task 6、Task 7 |
| §8 构建基线 + cann_ops_transformer | Task 10 Step 1、Task 2 Step 3 |
| §9 验证矩阵 | 各 Task 的验证 Step；NPU 项 V1–V9 不在本计划 |
| §10 风险 R1–R11 | R3 → Task 10 Step 3；其余记录在 Task 11 Step 3 |

**2. 占位符扫描**：唯一的占位是 Task 3 Step 7 的 `test_step5_summary_contract.py`，
它是 spec §5.2 明确授权的（"必须随 L6 一起按 TailRing 重写，不能直接搬"），
且文件内 docstring 写明了为什么空、要写成什么样、由哪个计划写。
其余步骤均含实际命令或代码。

**3. 命名一致性**：
- `_cann_parallel`（Task 10 Step 3 定义，同 Step 内 2 处调用）
- `_load_offload_backend`（Task 5 Step 4，与 main 现有代码无冲突）
- `VLLM_ASCEND_ENABLE_STEP5_FP8`（Task 5 Step 1 改名，同 Step 内全局替换）
- `AscendStep5ForCausalLM` / `AscendStep5ForConditionalGeneration` / `AscendStep5MTP`
  （Task 5 Step 2 注册名，与 Task 2 落地的 `models/step5/__init__.py`、`vl.py`、`mtp.py`
  中的类名一致）

**4. 已知的本计划局限**（执行者须知）：
- 本机无 torch_npu，`import vllm_ascend` 级别的验证做不了，用
  `py_compile` + `pytest --collect-only` 替代
- L6 未做，因此 step5 的摘要链路功能不完整，本计划产出的是"结构完整、可 import、
  非摘要功能可验证"的中间态
- `speculator.py` 的最终形态取决于 Task 8 Step 3 的判断，计划给了两种情况的处理路径
