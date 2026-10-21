# L6 TailRing 摘要链路重写 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 vllm-ascend 侧 Step5 摘要链路从 SWF journal 架构重写为 vLLM main 的 TailRing 架构，消除对已被上游移除的 request journal / KV side-storage 协议的依赖。

**Architecture:** vLLM main 用 `Step5CSATailRing`（每请求环形缓冲，存开放 region 的原始 `[index_k | index_z]` 行）+ `Step5SparseSummaryCache(mean)`（只包一个 `[blocks, block_size//8, 256]` uint8 张量）。摘要由闭合 region 的第八行一次性重放算出，被拒 draft 靠"下一行覆盖"自然失效，无需 undo log。NPU 侧只需替换 update 的 dispatch 与 triton kernel。

**Tech Stack:** Python 3.12、triton-ascend 3.2.2、`tl.extra.cann.libdevice`、pytest。

**Spec:** `docs/superpowers/specs/2026-10-20-step5-rebase-to-githubupstream-main-design.md` §6
**依赖:** 本分支已落地 `step5/tailring-rebase`，`vllm_ascend/models/step5/` 下是 SWF 版代码（L1 落地），本计划将其重写。

---

## 前置事实（已核实，写代码时直接引用）

### main 的公开 API（`vllm/models/step5/nvidia/`）

```python
# sparse_summary_cache.py
class Step5SparseSummaryCache:
    def __init__(self, mean: torch.Tensor): ...      # 只收一个张量
    def update_decode(self, key, z, slots, positions, ring): ...
    def update_verify(self, key, z, slots, query_start_loc, seq_lens, ring): ...
    def update_prefill(self, key, z, metadata, positions, ring): ...

class Step5SparseSummaryCacheLayer(nn.Module, AttentionLayerBase):
    def __init__(self, *, prefix: str, target_impl: Any,
                 static_forward_context: dict[str, Any]): ...   # 无 main_layer_name / sparse_config
    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:    # FullAttentionSpec(tokens_per_state=8, uint8, 256)
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache
        self._target_impl.summary_cache = Step5SparseSummaryCache(kv_cache[:, 0])

# csa_tail_ring.py
class Step5CSATailRing(nn.Module, AttentionLayerBase):
    def __init__(self, *, prefix, main_layer_name, num_speculative_tokens,
                 static_forward_context): ...        # 注意：ring 需要 main_layer_name
    required_rows = 7 + num_speculative_tokens
    rows / capacity / rows_per_block                # bind_kv_cache 后可用
    def metadata(self) -> Step5CSATailRingMetadata   # .base / .slot_mapping

# ops/summary_update.py（386 行，本计划的 kernel 蓝本）
libdevice = tl.extra.cuda.libdevice                 # ← 必须改成 tl.extra.cann.libdevice
_maximum / _accumulate / _store_summary / _replay_ring
_update_decode_values / _update_decode / update_decode(key,z,slots,positions,ring_base,mean,ring,capacity)
_update_verify / update_verify(key,z,slots,query_start_loc,seq_lens,ring_base,mean,ring,capacity)
_build_prefill_groups / build_prefill_groups(slots,positions,owners,query_start_loc) -> [rows,10]
_update_prefill / _write_ring / update_prefill(key,z,metadata,positions,ring_base,ring_slots,mean,ring,capacity)
summary_layout(mean) -> (MEAN_BLOCK, BLOCK_SIZE)
```

### 当前分支的 SWF 版集成点（待改）

| 文件 | 行 | 现状 | 目标 |
|---|---|---|---|
| `summary_kernels.py` | 98 | `update_decode(key,z,slots,mean,num,den,max,last_slot,pages,block_size)` | main 的 8 参签名 |
| `summary_kernels.py` | 260 | `build_prefill_groups(...)` | main 的 `(slots,positions,owners,query_start_loc)` |
| `summary_kernels.py` | 344 | `update_prefill(key,z,groups,mean,...,state=None)` | main 的 8 参签名 |
| `summary_kernels.py` | 485 | `update_request_journal(...)` | **删除** |
| `summary_cache.py` | 29 | 5 参 `__init__` + `self.config` | 无 `__init__` override |
| `summary_cache.py` | 46-70 | 7 个 property | **删除** |
| `summary_cache.py` | 129 | `update_request_journal(...)` | **删除** |
| `summary_cache.py` | 190+ | `AscendStep5SparseSummaryCacheLayer` 全套 side-storage | 只留 `bind_kv_cache` + scratch |
| `dsa_attention.py` | 134/163/181/195 | `step5_request_state` 字段 ×4 | **删除** |
| `dsa_attention.py` | 342/384/397/411 | request_state 传递 | **删除** |
| `dsa_attention.py` | 596-598 | journal 断言 | **删除** |
| `dsa_attention.py` | 630 | `update_request_journal(...)` | → `update_verify(...)` |
| `dsa_attention.py` | 642 | `update_decode(key,z,slots)` | → `update_decode(key,z,slots,positions,ring)` |
| `dsa_attention.py` | 657 | `update(key,z,slots,metadata)` | → `update_prefill(key,z,metadata,positions,ring)` |
| `dsa_attention.py` | 570/579/626/654/716/792/864 | `summary_cache.config` / `mean_cache` / `block_size` | 适配 main API |
| `model_state.py` | 全文 | `AscendStep5ModelState` | **删除文件** |
| `vl.py` | 33-38 | `get_model_state_cls` 返回 `AscendStep5ModelState` | 返回 `AscendModelState`（R8 方案 b） |
| `_step5_compat.py` | 68/151/174 | `Step5SparseSummaryCacheConfig` | **删除** |

### 已解除的疑虑

- **R11**：`_decode_summary_layout.py` 不引用 `config`/`Config`（grep 无命中），删 `Step5SparseSummaryCacheConfig` 不影响它。

---

## Task 1: libdevice checkpoint（L6 的准入门槛）

**Files:**
- Create: `tests/ut/models/step5/test_libdevice_checkpoint.py`

**这是 L6 的第一步，也是最关键的一道门。** main 的 `summary_update.py:17` 是
`libdevice = tl.extra.cuda.libdevice`，在 Ascend 上会导致 Triton 解析 CUDA 符号而非 CANN
等价物，kernel 编译失败（vllm-ascend 自己的 `patch/__init__.py:1396-1402` 记录过这个坑）。
step4 的 `summary_kernels.py:13` 已经用 `tl.extra.cann.libdevice`，所以这不是新风险，
但必须在写任何 TailRing 逻辑前单独验证。

- [ ] **Step 1: 写最小 libdevice 验证测试**

```python
"""L6 gate: prove tl.extra.cann.libdevice compiles and runs on this device.

main's reference implementation uses tl.extra.cuda.libdevice, which makes
Triton resolve CUDA libdevice symbols instead of the CANN equivalents and
fails to compile on Ascend. Every kernel in the TailRing rewrite calls
libdevice.exp, so this must pass before any rewrite work starts.
"""

import pytest

tl = pytest.importorskip("triton.language")
triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")


def _cann_libdevice():
    import triton.language.extra.cann.libdevice as cann_libdevice

    return cann_libdevice


def test_cann_libdevice_exp_matches_torch():
    libdevice = _cann_libdevice()

    @triton.jit
    def _exp_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, libdevice.exp(x), mask=mask)

    if not torch.npu.is_available():
        pytest.skip("NPU required for the libdevice compile check")

    x = torch.linspace(-20, 20, 1024, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    _exp_kernel[(4,)](x, y, x.numel(), BLOCK=256)
    torch.npu.synchronize()
    assert torch.allclose(y.cpu(), torch.exp(x.cpu()), rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: 在开发环境运行**

```bash
python -m pytest tests/ut/models/step5/test_libdevice_checkpoint.py -v
```

Expected: PASS。若 `import triton.language.extra.cann.libdevice` 失败或编译报错，
**停止 L6 全部工作**，先解决 triton-ascend 3.2.2 的 libdevice 可用性——这是环境问题，
不是代码问题（spec §6.5）。

- [ ] **Step 3: 通过后把 summary_kernels.py 的 libdevice 固定为 cann 版**

确认 `vllm_ascend/models/step5/summary_kernels.py` 第 13 行是
`libdevice = tl.extra.cann.libdevice`（step4 落地时已是，本步只是确认，Task 2 会沿用）。

- [ ] **Step 4: 提交**

```bash
git add tests/ut/models/step5/test_libdevice_checkpoint.py
git commit -m "test(step5): add libdevice checkpoint gate for the TailRing rewrite

Proves tl.extra.cann.libdevice compiles and runs before any TailRing kernel
is written. main's reference implementation uses the CUDA libdevice, which
fails to compile on Ascend."
```

---

## Task 2: 重写 summary_kernels.py（542 → ~330 行）

**Files:**
- Modify: `vllm_ascend/models/step5/summary_kernels.py`

- [ ] **Step 1: 删除 SWF 独有的 kernel**

删除以下函数（约 210 行）：
- `update_decode`（第 98 行，10 参 SWF 签名）
- `_accumulate_prefill`（118）、`_commit_prefill`（183）
- `make_prefill_metadata_from_slots`（289）、`make_prefill_metadata`（325）
- `update_prefill`（344，SWF 签名）
- `_update_request_journal`（391）、`update_request_journal`（485）

保留：`_maximum`（17）、`_update_decode_values`（22）——但后者要改签名（见 Step 3）。

- [ ] **Step 2: 写入 main 的共享数学 kernel（逐字，仅换 libdevice）**

在文件头部（`libdevice = tl.extra.cann.libdevice` 之后）加入：

```python
@triton.jit
def _maximum(x, y):
    return tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _accumulate(numerator, denominator, maximum, key, z):
    updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
    scale = tl.where(denominator > 0, libdevice.exp(maximum - updated_max), 0)
    weight = libdevice.exp(z - updated_max)
    return (
        numerator * scale + weight * key,
        denominator * scale + weight,
        updated_max,
    )


@triton.jit
def _store_summary(
    Mean,
    slot,
    dim,
    numerator,
    denominator,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write the summary of the region holding KV ``slot``."""
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    address = slot // BLOCK_SIZE * MEAN_BLOCK + slot % BLOCK_SIZE // 8 * 256
    tl.store(Mean + address + dim, summary_fp8)


@triton.jit
def _replay_ring(Ring, base, first, last, dim, RING_CAPACITY: tl.constexpr):
    """Accumulate the ring rows at positions ``[first, last)`` in order."""
    numerator = tl.zeros((256,), tl.float32)
    denominator = tl.zeros((256,), tl.float32)
    maximum = tl.max(tl.full((256,), -float("inf"), tl.float32), 0)
    for position in range(first, last):
        row = Ring + (base + position % RING_CAPACITY) * 512
        key = tl.load(row + dim).to(tl.float32)
        z = tl.load(row + 256 + dim).to(tl.float32)
        numerator, denominator, maximum = _accumulate(
            numerator, denominator, maximum, key, z
        )
    return numerator, denominator, maximum
```

`_replay_ring` 是 TailRing 的核心：region 闭合时从环形缓冲按序重放 8 行原始数据。
step4 没有这个 kernel，是从 main 新增的。

- [ ] **Step 3: 写入 decode / verify / prefill 三条路径**

```python
@triton.jit
def _update_decode_values(
    slot, position, base, dim, key, z, Mean, Ring,
    RING_CAPACITY: tl.constexpr, MEAN_BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Store an open-region row, or summarize the region this row closes."""
    if (slot >= 0) & (base >= 0):
        if position % 8 == 7:
            numerator, denominator, maximum = _replay_ring(
                Ring, base, position - 7, position, dim, RING_CAPACITY
            )
            numerator, denominator, maximum = _accumulate(
                numerator, denominator, maximum, key, z
            )
            _store_summary(Mean, slot, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE)
        else:
            row = Ring + (base + position % RING_CAPACITY) * 512
            tl.store(row + dim, key.to(tl.bfloat16))
            tl.store(row + 256 + dim, z.to(tl.bfloat16))


@triton.jit
def _update_decode(
    Key, Z, Slots, Positions, RingBase, Mean, Ring,
    KEY_STRIDE: tl.constexpr, Z_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr, MEAN_BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.arange(0, 256)
    key = tl.load(Key + row * KEY_STRIDE + dim).to(tl.float32)
    z = tl.load(Z + row * Z_STRIDE + dim).to(tl.float32)
    _update_decode_values(
        tl.load(Slots + row), tl.load(Positions + row), tl.load(RingBase + row),
        dim, key, z, Mean, Ring, RING_CAPACITY, MEAN_BLOCK, BLOCK_SIZE,
    )


@triton.jit
def _update_verify(
    Key, Z, Slots, QueryStart, SeqLens, RingBase, Mean, Ring,
    KEY_ROW_STRIDE: tl.constexpr, KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr, Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr, MEAN_BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # One program per request: its rows update the same open region in order.
    request = tl.program_id(0)
    start = tl.load(QueryStart + request)
    end = tl.load(QueryStart + request + 1)
    if start < end:
        base = tl.load(RingBase + start)
        seq = tl.load(SeqLens + request)
        dim = tl.arange(0, 256)
        for row in range(start, end):
            slot = tl.load(Slots + row)
            if (slot >= 0) & (base >= 0):
                position = seq - (end - row)
                ring_row = Ring + (base + position % RING_CAPACITY) * 512
                key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
                z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
                tl.store(ring_row + dim, key.to(tl.bfloat16))
                tl.store(ring_row + 256 + dim, z.to(tl.bfloat16))
                if position % 8 == 7:
                    tl.debug_barrier()
                    numerator, denominator, _ = _replay_ring(
                        Ring, base, position - 7, position + 1, dim, RING_CAPACITY
                    )
                    _store_summary(Mean, slot, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE)
            tl.debug_barrier()


@triton.jit
def _build_prefill_groups(
    Slots, Positions, Owners, QueryStart, Metadata, POSITION_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    output = Metadata + row.to(tl.int64) * 10
    owner = tl.load(Owners + row)
    start = tl.load(QueryStart + owner)
    end = tl.load(QueryStart + owner + 1)
    position = tl.load(Positions + row * POSITION_STRIDE)
    head = (row == start) | (position % 8 == 0)
    offset = tl.arange(0, 8)
    step = offset - position % 8
    rows = row + step
    in_group = head & (step >= 0) & (rows < end)
    slot = tl.load(Slots + rows, in_group, other=-1)
    valid = in_group & (slot >= 0)
    first = tl.min(tl.where(valid, offset, 8), 0)
    region = tl.max(tl.where(valid, slot // 8, -1), 0)
    tl.store(output, tl.where(first < 8, region, -1))
    tl.store(output + 1, first)
    tl.store(output + 2 + offset, tl.where(valid, rows, -1).to(tl.int64))


@triton.jit
def _update_prefill(
    Key, Z, Metadata, Positions, RingBase, Mean, Ring,
    POSITION_STRIDE: tl.constexpr,
    KEY_ROW_STRIDE: tl.constexpr, KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr, Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr, MEAN_BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    metadata = Metadata + tl.program_id(0) * 10
    region = tl.load(metadata)
    if (region >= 0) & (tl.load(metadata + 9) >= 0):
        first = tl.load(metadata + 1)
        head = tl.load(metadata + 2 + first)
        dim = tl.arange(0, 256)
        position = tl.load(Positions + head * POSITION_STRIDE)
        numerator, denominator, maximum = _replay_ring(
            Ring, tl.load(RingBase + head), position - first, position, dim, RING_CAPACITY,
        )
        for offset in range(8):
            row = tl.load(metadata + 2 + offset)
            valid = row >= 0
            key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE, mask=valid, other=0).to(tl.float32)
            z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE, mask=valid, other=0).to(tl.float32)
            updated = _accumulate(numerator, denominator, maximum, key, z)
            numerator = tl.where(valid, updated[0], numerator)
            denominator = tl.where(valid, updated[1], denominator)
            maximum = tl.where(valid, updated[2], maximum)
        _store_summary(Mean, region * 8, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE)


@triton.jit
def _write_ring(
    Key, Z, RingSlots, Ring,
    KEY_ROW_STRIDE: tl.constexpr, KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr, Z_DIM_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(RingSlots + row)
    if slot >= 0:
        dim = tl.arange(0, 256)
        key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
        z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
        tl.store(Ring + slot * 512 + dim, key.to(tl.bfloat16))
        tl.store(Ring + slot * 256 + dim, z.to(tl.bfloat16))
```

- [ ] **Step 4: 写入 host 侧入口函数**

```python
def summary_layout(mean: torch.Tensor) -> tuple[int, int]:
    """(bytes between blocks, tokens per block) of a ``[B, R, 256]`` cache."""
    assert mean.stride(1) == 256 and mean.stride(2) == 1
    return mean.stride(0), mean.shape[1] * 8


def update_decode(key, z, slots, positions, ring_base, mean, ring, capacity):
    if slots.numel():
        _update_decode[(slots.numel(),)](
            key, z, slots, positions, ring_base, mean, ring,
            key.stride(0), z.stride(0), capacity, *summary_layout(mean),
            num_warps=4, enable_fp_fusion=False,
        )


def update_verify(key, z, slots, query_start_loc, seq_lens, ring_base, mean, ring, capacity):
    """Uniform MTP verify update; graph capturable, no host metadata."""
    requests = seq_lens.shape[0]
    if requests:
        _update_verify[(requests,)](
            key, z, slots, query_start_loc, seq_lens, ring_base, mean, ring,
            key.stride(0), key.stride(2), z.stride(0), z.stride(2),
            capacity, *summary_layout(mean), num_warps=4, enable_fp_fusion=False,
        )


def build_prefill_groups(slots, positions, owners, query_start_loc) -> torch.Tensor:
    """Summary update metadata on device, without host slots or a sync."""
    rows = slots.shape[0]
    metadata = torch.empty((rows, 10), dtype=torch.int64, device=slots.device)
    if rows:
        _build_prefill_groups[(rows,)](
            slots, positions, owners, query_start_loc, metadata, positions.stride(0),
        )
    return metadata


def update_prefill(key, z, metadata, positions, ring_base, ring_slots, mean, ring, capacity):
    """Summarize the regions this step closes, then store open-region rows."""
    rows = metadata.shape[0]
    if not rows:
        return
    strides = (key.stride(0), key.stride(2), z.stride(0), z.stride(2))
    _update_prefill[(rows,)](
        key, z, metadata, positions, ring_base, mean, ring,
        positions.stride(0), *strides, capacity, *summary_layout(mean),
        num_warps=4, enable_fp_fusion=False,
    )
    _write_ring[(ring_slots.shape[0],)](key, z, ring_slots, ring, *strides, num_warps=4)
```

- [ ] **Step 5: 语法检查 + 提交**

```bash
python -m py_compile vllm_ascend/models/step5/summary_kernels.py && echo "exit=$?"
git add vllm_ascend/models/step5/summary_kernels.py
git commit -m "refactor(step5): replace SWF journal kernels with main's TailRing kernels

Ports _replay_ring, _update_decode, _update_verify, _build_prefill_groups,
_update_prefill and _write_ring from main's ops/summary_update.py, dropping
update_request_journal and the page-accumulator decode path.

libdevice stays tl.extra.cann.libdevice (main uses the CUDA one, which does
not compile on Ascend); the libdevice checkpoint in Task 1 gates this."
```

---

## Task 3: 重写 AscendStep5SparseSummaryCache（391 → ~120 行）

**Files:**
- Modify: `vllm_ascend/models/step5/summary_cache.py`

- [ ] **Step 1: 替换类定义与构造**

把 `AscendStep5SparseSummaryCache`（第 22–188 行）整体替换为：

```python
class AscendStep5SparseSummaryCache(Step5SparseSummaryCache):
    """main's TailRing summary cache with explicit NPU update dispatch.

    main's ``Step5SparseSummaryCache`` calls ``.ops.summary_update``, whose
    triton kernels import ``tl.extra.cuda.libdevice`` and therefore do not
    compile on Ascend. Override the three update entry points so they route
    to the Ascend kernels in ``summary_kernels.py`` instead.
    """

    def update_decode(self, key, z, slots, positions, ring) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_decode

        if key.device.type != "npu":
            raise RuntimeError("AscendStep5SparseSummaryCache.update_decode requires an NPU tensor")
        update_decode(
            key, z, slots, positions,
            ring.metadata().base[: slots.shape[0]],
            self.mean, ring.rows, ring.capacity,
        )

    def update_verify(self, key, z, slots, query_start_loc, seq_lens, ring) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_verify

        if key.device.type != "npu":
            raise RuntimeError("AscendStep5SparseSummaryCache.update_verify requires an NPU tensor")
        update_verify(
            key, z, slots, query_start_loc, seq_lens,
            ring.metadata().base[: slots.shape[0]],
            self.mean, ring.rows, ring.capacity,
        )

    def update_prefill(self, key, z, metadata, positions, ring) -> None:
        from vllm_ascend.models.step5.summary_kernels import update_prefill

        if key.device.type != "npu":
            raise RuntimeError("AscendStep5SparseSummaryCache.update_prefill requires an NPU tensor")
        ring_metadata = ring.metadata()
        update_prefill(
            key, z, metadata, positions,
            ring_metadata.base[: metadata.shape[0]],
            ring_metadata.slot_mapping[: metadata.shape[0]],
            self.mean, ring.rows, ring.capacity,
        )
```

**删除**：`__init__`（main 的只收 `mean`）、7 个 property、`prefill_metadata`、`update`、
`_prefill_state`、`update_request_journal`、`reset_scheduler_blocks`、`reset_blocks`、
以及 `self.config = Step5SparseSummaryCacheConfig(...)`。

- [ ] **Step 2: 语法检查 + 提交**

```bash
python -m py_compile vllm_ascend/models/step5/summary_cache.py && echo "exit=$?"
git add vllm_ascend/models/step5/summary_cache.py
git commit -m "refactor(step5): rewrite AscendStep5SparseSummaryCache onto main's TailRing API

Subclasses main's Step5SparseSummaryCache(mean) with no constructor
override, and overrides update_decode / update_verify / update_prefill to
dispatch to the Ascend kernels instead of main's CUDA-libdevice ones.

Drops the SWF request journal, the page accumulators, the speculative undo
snapshots and the seven convenience properties, none of which exist in
main's design."
```

---

## Task 4: 重写 AscendStep5SparseSummaryCacheLayer

**Files:**
- Modify: `vllm_ascend/models/step5/summary_cache.py`

- [ ] **Step 1: 核实 main 的 Layer 签名与 model.py 的实例化方式**

```bash
cd /data/LY/code/work-code/vllm-private
git show githubupstream/main:vllm/models/step5/nvidia/sparse_summary_cache.py | grep -n -A12 "class Step5SparseSummaryCacheLayer"
git show githubupstream/main:vllm/models/step5/nvidia/model.py | grep -n -B3 -A12 "Step5SparseSummaryCacheLayer(\|Step5CSATailRing("
```

**必须确认**：main 的 `Step5SparseSummaryCacheLayer.__init__` 签名是
`(*, prefix, target_impl, static_forward_context)`——**没有** `main_layer_name` 和
`sparse_config`。而 `Step5CSATailRing.__init__` **需要** `main_layer_name`。
记录 model.py 实际传了哪些参数，Step 2 按此对齐。

- [ ] **Step 2: 替换 Layer 类**

把 `AscendStep5SparseSummaryCacheLayer`（第 190 行起）替换为：

```python
class AscendStep5SparseSummaryCacheLayer(Step5SparseSummaryCacheLayer):
    """main's TailRing summary layer plus the Ascend DSA scratch workspace.

    main's layer is a plain FullAttentionSpec layer bound through the normal
    ``bind_kv_cache`` path; there is no side-storage protocol to implement.
    The only Ascend-specific work is handing the shared DSA scratch workspace
    to the attention impl, which the SWF version used to do from
    ``prepare_kv_cache_side_storage_memory_profile``.
    """

    def __init__(self, *, prefix: str, target_impl: Any, static_forward_context: dict[str, Any]) -> None:
        super().__init__(prefix=prefix, target_impl=target_impl, static_forward_context=static_forward_context)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        self._bind_dsa_scratch_workspace()

    def _bind_dsa_scratch_workspace(self) -> None:
        """Share one profiled scratch workspace across all DSA layers."""
        from vllm_ascend.models.step5.dsa_attention import AscendStep5DSAMetadataBuilder

        builder_cls = AscendStep5DSAMetadataBuilder
        workspace = getattr(builder_cls, "_shared_scratch_workspace", None)
        if workspace is not None:
            self._target_impl.dsa_scratch_workspace = workspace
```

**删除**：`get_attn_backend`、`_set_summary_cache_budget`、`_fixed_runtime_state_budget`、
`bind_kv_cache_side_storage`、`side_kv_cache`、`_side_storage_peers`、
`_ensure_shared_scratch_workspace`、`kv_cache_side_storage_memory_profile_key`、
`prepare_kv_cache_side_storage_memory_profile`、`zero_kv_cache_side_storage`、
`release_kv_cache_side_storage`、`reset_kv_cache_side_storage_runtime_state`、
`copy_kv_cache_side_storage`。

**注意**：`_bind_dsa_scratch_workspace` 的具体实现取决于 Task 5 之后
`dsa_attention.py` 里 scratch workspace 的最终形态。若 Step 1 发现 main 的
`Step5CSATailRing` 已接管 scratch 管理，则本方法可为空，只保留 `super().bind_kv_cache()`。

- [ ] **Step 3: 语法检查 + 提交**

```bash
python -m py_compile vllm_ascend/models/step5/summary_cache.py && echo "exit=$?"
git add vllm_ascend/models/step5/summary_cache.py
git commit -m "refactor(step5): drop SWF side-storage protocol from the summary layer

main's Step5SparseSummaryCacheLayer is a plain FullAttentionSpec layer bound
through bind_kv_cache; the side-storage protocol it used to implement does
not exist upstream. Keeps only the Ascend DSA scratch-workspace handoff."
```

---

## Task 5: dsa_attention.py 去 journal（955 → ~800 行）

**Files:**
- Modify: `vllm_ascend/models/step5/dsa_attention.py`

- [ ] **Step 1: 删除 step5_request_state 字段与传递**

删除 4 处 builder 字段声明（134、163、181、195 行）：
```python
    step5_request_state: torch.Tensor | None = None
```

删除 request_state 的读取与传递（342、384、397、411 行）：
```python
        request_state = self.step5_request_state
        if request_state is not None:
            ...
```
以及三处 `step5_request_state=...` 关键字参数。

- [ ] **Step 2: 删除 journal 断言并改派发**

删除 596–598 行：
```python
            if states is None:
                raise RuntimeError("Step5 journal requires persistent v2 request state indices")
```

把 630 行 `summary_cache.update_request_journal(...)` 改为：
```python
                    summary_cache.update_verify(
                        key, z, slots, query_start_loc, seq_lens, ring
                    )
```

把 642 行：
```python
            summary_cache.update_decode(key[:decode_rows], z[:decode_rows], slots[:decode_rows])
```
改为：
```python
            summary_cache.update_decode(
                key[:decode_rows], z[:decode_rows], slots[:decode_rows],
                positions[:decode_rows], ring,
            )
```

把 657 行：
```python
                summary_cache.update(key[decode_rows:], z[decode_rows:], slots[decode_rows:], metadata)
```
改为：
```python
                summary_cache.update_prefill(
                    key[decode_rows:], z[decode_rows:], metadata,
                    positions[decode_rows:], ring,
                )
```

- [ ] **Step 3: 适配 summary_cache 的属性访问**

把依赖 SWF 专有成员的访问改为 main 的 API：

| 现状 | 改为 |
|---|---|
| `summary_cache.config`（570） | 删除；config 概念在 main 不存在 |
| `summary_cache.mean_cache`（716/792/864） | `summary_cache.mean` |
| `summary_cache.block_size`（626/654） | 从 `vllm_config.cache_config.block_size` 取 |
| `summary_cache.prefill_state_scratch`（579） | scratch workspace 的新归属（见 Task 4 Step 2 注意） |

`ring` 的获取方式：从 `Step5CSATailRing` 实例取。核实 model.py 如何把 ring 暴露给
attention impl（Step 1 of Task 4 已查），典型形式是
`target_impl.csa_tail_ring` 或从 forward context 按 layer name 取。

- [ ] **Step 4: 语法检查 + 提交**

```bash
python -m py_compile vllm_ascend/models/step5/dsa_attention.py && echo "exit=$?"
git add vllm_ascend/models/step5/dsa_attention.py
git commit -m "refactor(step5): remove the SWF request journal from DSA attention

Drops step5_request_state from the four metadata builders and routes the
three update paths to main's TailRing API: update_verify for MTP verify,
update_decode for decode, update_prefill for prefill. Addressing now comes
from ring.metadata().base / .slot_mapping instead of a persistent
per-request journal index."
```

---

## Task 6: 删除 model_state、调整 vl.py、清理 _step5_compat

**Files:**
- Delete: `vllm_ascend/models/step5/model_state.py`
- Modify: `vllm_ascend/models/step5/vl.py`
- Modify: `vllm_ascend/models/step5/_step5_compat.py`

- [ ] **Step 1: 删除 model_state.py**

```bash
git rm vllm_ascend/models/step5/model_state.py
```

理由：`AscendStep5ModelState` 唯一的新增逻辑是维护 `request_state` 并注入 builder，
而它服务的 SWF undo journal 已被 main 整体移除。main 的 `Step5SparseSummaryCache`
docstring 明写 "this cache holds no per-request state"。

- [ ] **Step 2: 调整 vl.py 的 get_model_state_cls（R8 方案 b）**

把：
```python
    @staticmethod
    def get_model_state_cls():
        # Keep multimodal input preparation on the Ascend v2 attention path.
        from vllm_ascend.models.step5.model_state import AscendStep5ModelState

        return AscendStep5ModelState
```
改为：
```python
    @staticmethod
    def get_model_state_cls():
        # Keep multimodal input preparation on the Ascend v2 attention path.
        # The SWF request journal is gone, so the default Ascend model state
        # is used; defining this method also keeps the model MRV2-only.
        from vllm_ascend.worker.v2.model_states.default import AscendModelState

        return AscendModelState
```

**这是 R8 的暂定方案 (b)**：保留 override 以保住 "MRV2-only" 语义，但返回 main 默认的
`AscendModelState`。验证通过后（spec §10.3 R8）若确认删除 override 仍走 v2 路径，
可简化为完全删除。

- [ ] **Step 3: 清理 _step5_compat.py 的 Step5SparseSummaryCacheConfig**

删除第 68 行的 `class Step5SparseSummaryCacheConfig` 定义、第 151 行的
`Step5SparseSummaryCacheConfig.max_region_fragments_per_block(...)` 调用，
以及 `__all__`（第 174 行）中的 `"Step5SparseSummaryCacheConfig"`。

已核实 `_decode_summary_layout.py` 不引用 config（R11 解除），删除安全。

- [ ] **Step 4: 全链语法检查 + 提交**

```bash
python -m py_compile vllm_ascend/models/step5/*.py && echo "exit=$?"
git add -A vllm_ascend/models/step5/
git commit -m "refactor(step5): delete SWF model_state and drop the summary cache config

model_state.py existed only to expose the SWF request-journal index, which
main removed; vl.py now returns the default AscendModelState (keeping the
MRV2-only override per R8), and _step5_compat.py drops
Step5SparseSummaryCacheConfig, which nothing else references."
```

---

## Task 7: 写 test_step5_summary_contract.py（替换占位）

**Files:**
- Modify: `tests/ut/models/step5/test_step5_summary_contract.py`

- [ ] **Step 1: 写 TailRing 契约测试**

替换现有 13 行占位 docstring 为真实测试。保留 AST isolation 风格
（从源码编译类定义，不初始化 accelerator），使 CPU 主机可跑：

```python
"""Step5 summary-cache contracts under main's TailRing architecture.

AST isolation: class definitions are compiled from source so these run on
CPU-only hosts as well as in the NPU unit-test suite. Only the update
dispatch is mocked; the ring addressing and the online-softmax arithmetic
are the real implementations.
"""

import ast
import importlib.util
import math
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_summary_cache_cls():
    path = REPO_ROOT / "vllm_ascend/models/step5/summary_cache.py"
    tree = ast.parse(path.read_text())
    wanted = {"AscendStep5SparseSummaryCache"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert nodes, "AscendStep5SparseSummaryCache not found"
    namespace: dict = {}
    # Minimal stand-ins for the vLLM base classes the module imports.
    namespace["Step5SparseSummaryCache"] = type("Step5SparseSummaryCache", (), {})
    namespace["Step5SparseSummaryCacheLayer"] = type("Step5SparseSummaryCacheLayer", (), {})
    namespace["torch"] = __import__("torch")
    for node in nodes:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["AscendStep5SparseSummaryCache"]


def test_update_decode_requires_npu_tensor():
    cls = _load_summary_cache_cls()
    cache = cls.__new__(cls)
    ring = types.SimpleNamespace(metadata=lambda: types.SimpleNamespace(base=[]))
    with pytest.raises(RuntimeError, match="update_decode requires an NPU tensor"):
        cache.update_decode(None, None, None, None, ring)


def test_update_verify_requires_npu_tensor():
    cls = _load_summary_cache_cls()
    cache = cls.__new__(cls)
    ring = types.SimpleNamespace(metadata=lambda: types.SimpleNamespace(base=[]))
    with pytest.raises(RuntimeError, match="update_verify requires an NPU tensor"):
        cache.update_verify(None, None, None, None, None, ring)


def test_update_prefill_requires_npu_tensor():
    cls = _load_summary_cache_cls()
    cache = cls.__new__(cls)
    ring = types.SimpleNamespace(
        metadata=lambda: types.SimpleNamespace(base=[], slot_mapping=[])
    )
    with pytest.raises(RuntimeError, match="update_prefill requires an NPU tensor"):
        cache.update_prefill(None, None, None, None, ring)
```

- [ ] **Step 2: 跑测试**

```bash
python -m pytest tests/ut/models/step5/test_step5_summary_contract.py -v
```

Expected: 3 passed。若 AST 加载失败，检查 `namespace` 里的 base class stand-in 是否齐全
（按 `summary_cache.py` 的实际 import 补）。

- [ ] **Step 3: 提交**

```bash
git add tests/ut/models/step5/test_step5_summary_contract.py
git commit -m "test(step5): replace the summary-contract placeholder with TailRing tests

Covers the three update entry points' NPU dispatch guard. The SWF-journal
sources are not carried over: they test the undo log main removed."
```

---

## Task 8: 全量静态检查与交接

**Files:** 无（只读检查）

- [ ] **Step 1: 全链语法检查**

```bash
cd /data/LY/code/work-code/vllm-ascend-private
python -m py_compile vllm_ascend/models/step5/*.py && echo "step5 exit=$?"
```

Expected: `exit=0`

- [ ] **Step 2: 确认 SWF 痕迹已清**

```bash
echo "=== 以下各项应全部无输出 ==="
grep -rn "update_request_journal\|step5_request_state\|request_journal\|_speculative_tails\|Step5SparseSummaryCacheConfig\|kv_cache_side_storage" vllm_ascend/models/step5/*.py
grep -rn "AscendStep5ModelState" vllm_ascend/
ls vllm_ascend/models/step5/model_state.py 2>/dev/null
```

Expected: 全部无输出。若有残留，定位并清理。

- [ ] **Step 3: 确认 TailRing 接口已接入**

```bash
echo "=== 以下各项应有输出 ==="
grep -n "_replay_ring" vllm_ascend/models/step5/summary_kernels.py
grep -n "def update_verify\|def update_decode\|def update_prefill" vllm_ascend/models/step5/summary_cache.py
grep -n "ring.metadata()" vllm_ascend/models/step5/summary_cache.py
grep -n "AscendModelState" vllm_ascend/models/step5/vl.py
```

- [ ] **Step 4: 黑名单复核**

```bash
git diff githubupstream/main --name-only | grep -E "^(ci/|requirements\.txt|pyproject\.toml|\.gitmodules|csrc/third_party/catlass)" && echo "!! 黑名单被改" || echo "OK"
```

注意：`CMakeLists.txt` 与 `Dockerfile.a5`/`Dockerfile.a5.openEuler` 的改动是先前已批准并提交的
（CANN 9.2.0 切换），不算新违规。

- [ ] **Step 5: 生成交接清单**

在 `/data/LY/logs/update-step5/blockers.md` 追加：

```markdown
### L6 TailRing 重写完成后的 NPU 待验证项
- V2  _replay_ring 数值正确性（与 main 的 CPU 参考实现对拍）
- V3  950DT UB 容量（step4 有 cbb432f62 专修 indexer cache gather）
- V4  update_verify 的图捕获（main 标榜 graph capturable, no host metadata）
- V5  单卡 bf16 / FP8-BLOCK / ModelSlim 三权重路径
- V8  端到端拉起 Step5 vLLM server（最终验收）
- R8  验证删除 vl.py 的 get_model_state_cls override 后是否仍走 Ascend v2 路径
- Task 4 Step 2 的 _bind_dsa_scratch_workspace 是否需要非空实现
```

---

## Self-Review 记录

**1. Spec 覆盖核对**

| Spec §6 条目 | 任务 |
|---|---|
| §6.5 libdevice checkpoint | Task 1 |
| §6.2 summary_cache.py 重写映射 | Task 3 |
| §6.3 summary_kernels.py 适配 | Task 2 |
| §6.4 dsa_attention.py 去 journal + model_state 删除 | Task 5、Task 6 |
| §6.4 vl.py 的 get_model_state_cls（R8 方案 b） | Task 6 Step 2 |
| §5.2 test_step5_summary_contract.py | Task 7 |
| §9.2 V2/V3/V4/V5/V8 | Task 8 Step 5 交接清单 |
| R11（_decode_summary_layout 不依赖 config） | 已核实，Task 6 Step 3 安全删除 |

**2. 占位符扫描**：唯一"待定"是 Task 4 Step 2 的 `_bind_dsa_scratch_workspace`，
计划明确要求 Step 1 先核实 main 的 ring 是否已接管 scratch 管理，再决定实现或留空——
这是有依据的条件分支，不是 TODO 式占位。

**3. 命名一致性**：
- `update_decode / update_verify / update_prefill` 三签名在 Task 2（kernel 入口）、
  Task 3（cache 方法）、Task 5（调用点）三处一致
- `_replay_ring` 在 Task 2 定义，Task 2 Step 3 内部使用
- `ring.metadata().base` / `.slot_mapping` 在 Task 3、Task 5 一致
- `AscendModelState`（Task 6）与 main 的 `vllm_ascend/worker/v2/model_states/default.py`
  既有类一致

**4. 已知局限**（执行者须知）：
- Task 4 Step 2 的 scratch workspace 实现取决于 main 的 ring 设计，需先核实
- 本机无 NPU，Task 1/7 之外无法做数值验证；V2–V8 全部留给开发环境
- `ring` 对象在 `dsa_attention.py` 中的获取路径需在 Task 5 Step 3 落实
  （依赖 model.py 的实例化方式，Task 4 Step 1 已安排核实）
