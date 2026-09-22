from collections.abc import Iterable
from itertools import product as iprod
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer, KVCacheSideStorageLayer

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    PAGE_SIZE_EL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped as (block_index, seg_index, chunk_index).
    """
    pid = tl.program_id(0)
    chunks = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    total_work = n_blocks * work_per_block
    for work_idx in range(pid, total_work, GRID_SIZE):
        block_index = work_idx // work_per_block
        remainder = work_idx % work_per_block
        seg_index = remainder // chunks
        chunk_index = remainder % chunks
        block_id = tl.load(block_ids_ptr + block_index)
        seg_addr = tl.load(seg_addrs_ptr + seg_index)
        ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
        offset = block_id.to(tl.int64) * PAGE_SIZE_EL + chunk_index.to(tl.int64) * BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


class AscendKVBlockZeroer(KVBlockZeroer):
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Call :meth:`init_meta` once after KV caches are allocated to precompute
    segment addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(self, device: torch.device, pin_memory: bool) -> None:
        self.device = device
        self.pin_memory = pin_memory
        self._meta: tuple[torch.Tensor, int, int, int] | None = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None
        # group_id -> side-storage zero hooks, mirroring the upstream
        # KVBlockZeroer collection (e.g. Step5 DSA summary sidecar reset).
        self._side_storage_zero_hooks: dict[int, list[Any]] = {}

    def _collect_side_storage_zero_hooks(
        self,
        attn_groups: list["AttentionGroup"],
        static_forward_context: dict[str, Any],
    ) -> None:
        """Register each side-storage layer's zero hook under its owner group."""
        layer_name_to_group_id: dict[str, int] = {}
        for group in attn_groups:
            for layer_name in group.layer_names:
                previous_group_id = layer_name_to_group_id.setdefault(
                    layer_name, group.kv_cache_group_id
                )
                if previous_group_id != group.kv_cache_group_id:
                    raise ValueError(
                        "KV cache layer belongs to multiple groups: "
                        f"layer={layer_name!r}, groups="
                        f"{previous_group_id},{group.kv_cache_group_id}"
                    )
        seen_side_storage_layers: set[int] = set()
        for layer_name, layer in static_forward_context.items():
            if not isinstance(layer, KVCacheSideStorageLayer):
                continue
            layer_id = id(layer)
            if layer_id in seen_side_storage_layers:
                continue
            seen_side_storage_layers.add(layer_id)
            owner_name = layer.main_layer_name
            group_id = layer_name_to_group_id.get(owner_name)
            if group_id is None:
                raise ValueError(
                    "KV side-storage layer has no owner KV cache group: "
                    f"layer={layer_name!r}, owner={owner_name!r}"
                )
            self._side_storage_zero_hooks.setdefault(group_id, []).append(
                layer.zero_kv_cache_side_storage
            )
        for group in attn_groups:
            if (
                group.kv_cache_spec.requires_zeroing
                and group.kv_cache_group_id not in self._side_storage_zero_hooks
            ):
                raise ValueError(
                    "KV cache group declares side-storage zeroing but no "
                    "KVCacheSideStorageLayer owns the group: "
                    f"group={group.kv_cache_group_id}, "
                    f"layers={group.layer_names!r}"
                )

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[list[int]],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        """One-time precomputation for zero_block_ids.

        Builds absolute-address table for the Triton zeroing kernel.
        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        PAGE_SIZE_EL accounts for this ratio so that
        ``block_id * PAGE_SIZE_EL`` lands at the correct offset.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        attn_groups = list(attn_groups_iter)
        self._collect_side_storage_zero_hooks(attn_groups, static_forward_context)
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        page_size_el: int | None = None

        for group in attn_groups:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            ratio = spec.block_size // kernel_bs
            block_dim = 0

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_tuple = static_forward_context[layer_name].kv_cache
                assert len(kv_tuple) == 2, "K and V are not stored separately"
                for kv in kv_tuple:
                    block_dim = 0
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)

                    el = kv.element_size()
                    cur_bytes = kv.stride(block_dim) * el
                    assert cur_bytes % 4 == 0
                    kernel_block_el = cur_bytes // 4
                    cur_page_el = kernel_block_el * ratio
                    if page_size_el is None:
                        page_size_el = cur_page_el
                    else:
                        assert page_size_el == cur_page_el, f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"

                    block_stride_bytes = cur_bytes
                    outer_dims = [d for d in range(block_dim) if kv.stride(d) * el > block_stride_bytes]
                    outer_strides = [kv.stride(d) * el for d in outer_dims]
                    for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                        off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                        seg_addrs.append(dp + off_bytes)

        if not seg_addrs or page_size_el is None:
            self._meta = None
            return

        # _zero_kv_blocks_kernel will use int64 zeros, to meet the UB size, we use blk_size=64B/8B=8192
        blk_size = min(largest_power_of_2_divisor(page_size_el), 8192)
        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            page_size_el,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(
        self,
        block_ids: list[int],
        block_ids_by_group: list[list[int]] | None = None,
    ) -> None:
        """Zero the KV cache memory for the given block IDs.

        ``block_ids_by_group`` carries the scheduler's per-KV-group split;
        side-storage layers (e.g. the Step5 DSA summary sidecar) are reset
        through their registered hooks with their owner group's share.
        """
        if block_ids and self._meta is not None:
            self._zero_base_kv_blocks(block_ids)
        self._zero_side_storage_blocks(block_ids, block_ids_by_group)

    def _zero_base_kv_blocks(self, block_ids: list[int]) -> None:
        seg_addrs, page_size_el, blk_size, n_segs = self._meta
        n_blocks = len(block_ids)
        if n_blocks > self._id_cap:
            self._id_cap = n_blocks * 2
            self._ids_pinned = torch.empty(
                self._id_cap,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            )
            self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        assert self._ids_pinned is not None and self._ids_gpu is not None
        self._ids_pinned[:n_blocks].numpy()[:] = block_ids
        idx = self._ids_gpu[:n_blocks]
        idx.copy_(self._ids_pinned[:n_blocks], non_blocking=True)
        chunks = page_size_el // blk_size
        total_work = n_blocks * n_segs * chunks
        grid = min(total_work, get_vectorcore_num()) if total_work > 0 else 0
        if grid == 0:
            return
        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )

    def _zero_side_storage_blocks(
        self,
        block_ids: list[int],
        block_ids_by_group: list[list[int]] | None,
    ) -> None:
        """Dispatch per-group side-storage resets (mirrors upstream)."""
        side_storage_zero_hooks = self._side_storage_zero_hooks
        if not side_storage_zero_hooks:
            return
        if block_ids_by_group is None:
            if block_ids:
                raise RuntimeError(
                    "KV side-storage reset requires block IDs grouped by KV "
                    "cache group."
                )
            return
        required_group_count = max(side_storage_zero_hooks) + 1
        if len(block_ids_by_group) < required_group_count:
            raise RuntimeError(
                "KV side-storage reset is missing owner group IDs: "
                f"required_groups={required_group_count}, "
                f"actual_groups={len(block_ids_by_group)}"
            )
        flat_group_ids = [
            block_id for group_ids in block_ids_by_group for block_id in group_ids
        ]
        if not flat_group_ids:
            return
        group_ids_dev = async_tensor_h2d(
            flat_group_ids,
            device=self.device,
            dtype=torch.int64,
        )
        offset = 0
        for group_id, group_ids in enumerate(block_ids_by_group):
            next_offset = offset + len(group_ids)
            hooks = side_storage_zero_hooks.get(group_id, ())
            if hooks and next_offset > offset:
                ids = group_ids_dev[offset:next_offset]
                for hook in hooks:
                    hook(ids)
            offset = next_offset
