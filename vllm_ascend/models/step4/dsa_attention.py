# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 DSA (DeepSeek-style sparse attention) backend.

Structure mirrors the upstream CUDA implementation
(``vllm.models.step4.sparse_attention.Step4DSAAttentionImpl``): per-region
summaries live in a page-backed fp8 side cache whose lifetime follows the
owner KV blocks (``vllm_ascend.models.step4.summary_cache``), the per-step
token -> region layout is built from ``slot_mapping`` by the upstream Triton
layout builder, and ``forward`` is ``_maybe_update_summary_cache`` followed by
the sparse-GQA decode/prefill paths.

The CuTeDSL-only operators are replaced by the Triton kernels in
``vllm_ascend.models.step4.summary_kernels`` (region update, paged indexer
scoring); region selection, metadata packing and the sparse attention itself
reuse the parity-pinned kernels of ``dsa_kernels.py``.

Spec-decode verify batches (multi-token decode rows, ``query_len = K+1``)
follow the upstream ``Step4DSAMTP`` contract with an equivalent Ascend
mechanism: verify rows are *staged* into the active-slot tail state instead
of folded (staging is last-write-wins per (region, offset), so the next
round's rewrite of rejected positions is an exact rollback to the accepted
prefix), and completed regions are only materialized into the fp8 page
cache by the deferred flush once their last position is below the request's
committed context. The verify attention itself reuses the prefill-style
causal selection (``prefill_sparse_meta`` bounds each row's candidate
regions by its own position and force-appends the row's current region),
which is bit-compatible with the eager prefill path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.step4.sparse_summary_cache import Step4DSAScratchWorkspace
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_ascend.models.step4.dsa_kernels import (
    decode_sparse_meta,
    indexer_logits_paged,
    prefill_sparse_meta,
    sparse_attention_decode,
    sparse_attention_prefill,
)
from vllm_ascend.models.step4.summary_cache import AscendStep4SparseSummaryCache
from vllm_ascend.models.step4.summary_kernels import (
    build_decode_summary_layout_step3p5,
    csa_compact_decode_update,
    csa_compact_prefill_update,
    csa_deferred_flush,
    csa_verify_stage_update,
)

logger = init_logger(__name__)

_CSA_ALLOCATION_ERROR = (
    "Step4 DSA active-slot capacity is exhausted. Lower max_num_seqs or "
    "increase the CSA capacity policy."
)


def _is_graph_capturing() -> bool:
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


@dataclass
class AscendStep4DSARuntimeLayout:
    """Per-step token -> physical summary region mapping (see upstream)."""

    token_flat_slot: torch.Tensor
    token_positions: torch.Tensor
    reset_slots: torch.Tensor
    token_valid: torch.Tensor


class AscendStep4DSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_STEP4_DSA"

    @staticmethod
    def get_impl_cls():
        return AscendStep4DSAImpl

    @staticmethod
    def get_builder_cls():
        return AscendStep4DSAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 192]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Unify with the sliding-window layers' backend (AscendAttentionBackend
        # declares [128]) and the scheduler block size: with hybrid virtual
        # block splitting the runner reshapes the owner KV tensor to this
        # size, and keeping it at 128 makes scheduler block == kernel block ==
        # summary sidecar page. The kernels themselves only need
        # block_size % region_size == 0.
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2, 3, 4)


@dataclass
class AscendStep4DSAPrefillMetadata:
    seq_lens: torch.Tensor
    """(num_prefills,) full sequence lengths (cached prefix + this chunk)."""

    context_lens: torch.Tensor
    """(num_prefills,) tokens already cached before this chunk."""

    block_table: torch.Tensor
    """(num_prefills, max_pages) int32 rows for the prefill requests."""

    query_start_loc: torch.Tensor
    """(num_prefills + 1,) start of each prefill request within the query."""

    max_query_len: int

    num_query_tokens: int = 0
    """Total prefill query tokens, filled from the CPU start-loc copy so the
    step never pays a device sync for it."""

    seq_lens_host: list = None
    context_lens_host: list = None
    """CPU copies produced by the builder's existing seq_lens transfer; the
    impl reads these instead of calling ``.tolist()`` on device tensors."""

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    @property
    def num_query_tokens_padded(self) -> int:
        return int(self.query_start_loc[-1])


@dataclass
class AscendStep4DSADecodeMetadata:
    seq_lens: torch.Tensor
    """(num_decodes,) context length including the current token."""

    block_table: torch.Tensor
    """(num_decodes, max_pages) int32 rows for the decode requests."""

    query_len: int = 1
    """Uniform decode query length K (1 for plain decode, 1+num_spec for
    MTP verify batches)."""


@dataclass
class AscendStep4DSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    """(num_reqs + 1,) full-batch token start locations, including padding."""
    seq_lens: torch.Tensor
    """(num_reqs,) full-batch sequence lengths, including padding rows."""
    prefill: AscendStep4DSAPrefillMetadata | None = None
    decode: AscendStep4DSADecodeMetadata | None = None


class AscendStep4DSAMetadataBuilder(
    AttentionMetadataBuilder[AscendStep4DSAMetadata]
):
    # Decode forward is fully tensor-driven (see AscendStep4DSAImpl
    # ._forward_sparse_gqa_decode): static shapes per capture bucket, all
    # data-dependent choices realized with device-side validity masks, so
    # uniform single-token decode batches are ACL-graph capturable under
    # FULL_DECODE_ONLY; mixed prefill+decode batches still run eager -- the
    # contract of UNIFORM_BATCH.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendStep4DSAMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        prefill_metadata: AscendStep4DSAPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(
                num_prefills, num_prefill_tokens, qsl_cpu, num_decodes
            )
            prefill_end = num_decodes + active_prefills
            prefill_seq_lens = seq_lens[num_decodes:prefill_end]
            prefill_query_lens = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[
                num_decodes:prefill_end
            ]
            # One CPU round trip that already existed for context_lens: keep the
            # host lists around so the prefill step never calls .tolist() on a
            # device tensor again (each of those stalls the NPU pipeline).
            seq_lens_host_t = seq_lens[num_decodes:prefill_end].detach().cpu()
            context_lens_host_t = seq_lens_host_t - prefill_query_lens
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                context_lens_host_t.to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            prefill_metadata = AscendStep4DSAPrefillMetadata(
                seq_lens=prefill_seq_lens,
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                query_start_loc=query_start_loc[num_decodes : prefill_end + 1]
                - num_decode_tokens,
                max_query_len=int(prefill_query_lens.max()),
                num_query_tokens=int(prefill_query_lens.sum()),
                seq_lens_host=seq_lens_host_t.tolist(),
                context_lens_host=context_lens_host_t.tolist(),
            )

        decode_metadata: AscendStep4DSADecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(
                num_decodes, num_decode_tokens, decode_query_len
            )
            decode_metadata = AscendStep4DSADecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                query_len=decode_query_len,
            )

        return AscendStep4DSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


def _active_decode_num_reqs(
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
) -> int:
    """Return the number of real decode requests, ignoring padding."""
    if decode_query_len <= 0:
        return 0
    return min(num_decodes, num_decode_tokens // decode_query_len)


def _active_prefill_num_reqs(
    num_prefills: int,
    num_prefill_tokens: int,
    query_start_loc_cpu: torch.Tensor,
    num_decodes: int,
) -> int:
    """Return real prefill requests, ignoring tail padding segments."""
    if num_prefills <= 0 or num_prefill_tokens <= 0:
        return 0
    qsl_cpu = query_start_loc_cpu.detach().cpu()
    num_reqs = int(qsl_cpu.shape[0] - 1)
    active = 0
    tokens_accounted = 0
    for i in range(num_decodes, min(num_reqs, num_decodes + num_prefills)):
        query_len = int(qsl_cpu[i + 1] - qsl_cpu[i])
        if query_len <= 0:
            continue
        if tokens_accounted + query_len > num_prefill_tokens:
            break
        tokens_accounted += query_len
        active += 1
    if active > 0:
        return active
    return min(1, num_prefills)


class AscendStep4DSAImpl(AttentionImplBase[AscendStep4DSAMetadata]):
    # Read by AscendStep4SparseSummaryCacheLayer when sizing/binding the
    # page-backed summary sidecar (same names as the upstream impl).
    summary_cache_num_proxy_kv_heads: int = 1

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_regions: int,
        region_size: int,
        proxy_dim: int,
        max_num_seqs: int,
        max_model_len: int,
        block_size: int,
        max_num_batched_tokens: int,
        decode_out_dtype: torch.dtype | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_regions = topk_regions
        self.region_size = region_size
        self.proxy_dim = proxy_dim
        self.block_size = block_size
        if block_size % region_size != 0:
            raise ValueError(
                "Step4 DSA requires block_size to be a multiple of the region "
                f"size, got block_size={block_size}, region_size={region_size}."
            )
        self.num_speculative_tokens = int(num_speculative_tokens)
        if self.num_speculative_tokens > 0 and (
            self.num_speculative_tokens + 1 > region_size
        ):
            # Same constraint as the upstream CUDA impl: a verify batch's
            # K+1 rows must fit inside one region for the staged-rewrite
            # rollback to stay exact.
            raise ValueError(
                "Step4 DSA MTP requires num_speculative_tokens + 1 <= "
                f"region_size, got {self.num_speculative_tokens + 1} > "
                f"{region_size}."
            )
        self.regions_per_page = block_size // region_size
        self.num_kv_groups = self.num_kv_heads
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.max_model_len = int(max_model_len)
        self.max_num_batched_tokens = max(1, int(max_num_batched_tokens))
        self._decode_out_dtype = decode_out_dtype or torch.bfloat16
        # Selector width: every region a max-length sequence can fill.
        self._max_regions = -(-self.max_model_len // self.region_size)
        self._summary_cache: AscendStep4SparseSummaryCache | None = None
        self._summary_cache_config = None
        self._dsa_scratch_workspace = Step4DSAScratchWorkspace()
        self._dsa_scratch_bound = False

    # -- upstream-compatible side-storage contract ---------------------------

    @property
    def sparse_region_block_size(self) -> int:
        return self.region_size

    def bind_scratch_workspace(self, workspace: Step4DSAScratchWorkspace) -> None:
        """Use one fixed scratch pool for every DSA layer in this model."""
        self._dsa_scratch_workspace = workspace
        self._dsa_scratch_bound = False

    def _dsa_virtual_engine(self) -> int:
        from vllm.forward_context import is_forward_context_available

        if is_forward_context_available():
            return int(getattr(get_forward_context(), "virtual_engine", 0) or 0)
        return 0

    def _get_dsa_tensor_buffer_at_least(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fixed-capacity scratch from the model-wide DSA workspace.

        All DSA layers share one pool and run sequentially on one stream, so
        a per-name buffer is reused across layers; preallocating the maximum
        size at bind keeps the pointers stable for ACL-graph capture.
        """
        virtual_engine = self._dsa_virtual_engine()
        self._dsa_scratch_workspace.get_order_token(virtual_engine, device=device)
        buffers = self._dsa_scratch_workspace.tensor_buffers_by_engine.setdefault(
            virtual_engine, {}
        )
        required = 1
        for dim in shape:
            required *= int(dim)
        out = buffers.get(name)
        if (
            out is None
            or out.device != device
            or out.dtype != dtype
            or int(out.numel()) < int(required)
        ):
            if self._dsa_scratch_workspace.allocations_locked or getattr(
                self, "_dsa_scratch_bound", False
            ):
                raise RuntimeError(
                    "Step4 DSA scratch capacity was exceeded after bind: "
                    f"name={name!r}, requested_shape={shape}, "
                    f"required_elements={int(required)}."
                )
            if _is_graph_capturing():
                raise RuntimeError(
                    "Step4 DSA tensor scratch buffers must be initialized "
                    f"before graph capture; name={name!r}, shape={shape}."
                )
            out = torch.empty((int(required),), device=device, dtype=dtype)
            buffers[name] = out
        return out[: int(required)].view(*shape)

    def _csa_active_region_capacity(self) -> int:
        """Bind-time slot capacity for incomplete regions (see upstream).

        One prefill launch can hold two tail generations per request (the
        carry-over tail has not released its slot while the new tail already
        claimed one), so one slot per request is not enough; the 8x policy
        matches upstream, and every allocation failure is fail-closed through
        the device-side ``allocation_success`` flag.
        """
        return 8 * int(self.max_num_seqs)

    def csa_fixed_runtime_state_size_bytes(self) -> int:
        """Persistent CSA bytes that do not scale with KV blocks."""
        capacity = self._csa_active_region_capacity()
        proxy_dim = int(self.proxy_dim)
        region_block_size = int(self.region_size)
        stage_element_size = torch.tensor(
            [], dtype=self._decode_out_dtype
        ).element_size()
        state_elements = capacity * proxy_dim  # num_kv_heads == 1
        active_token_elements = state_elements * region_block_size
        return int(
            capacity * torch.tensor([], dtype=torch.int32).element_size()
            + torch.tensor([], dtype=torch.int32).element_size()
            + 2 * state_elements * torch.tensor([], dtype=torch.float32).element_size()
            + capacity * torch.tensor([], dtype=torch.float32).element_size()
            + 2 * active_token_elements * stage_element_size
            + capacity
            * region_block_size
            * torch.tensor([], dtype=torch.uint8).element_size()
        )

    # -- summary cache binding ------------------------------------------------

    def _initialize_csa_summary_state(
        self,
        summary_cache: AscendStep4SparseSummaryCache,
        *,
        capacity: int | None = None,
    ) -> None:
        capacity = int(capacity or self._csa_active_region_capacity())
        device = summary_cache.mean_cache.device
        total_regions = int(summary_cache.num_pages) * int(
            summary_cache.summaries_per_page
        )
        proxy_dim = int(self.proxy_dim)
        summary_cache._step4_csa_active_region_ids = torch.full(
            (capacity,), -1, device=device, dtype=torch.int32
        )
        summary_cache._step4_csa_active_slot_by_region = torch.full(
            (total_regions,), -1, device=device, dtype=torch.int32
        )
        summary_cache._step4_csa_allocation_success = torch.ones(
            (1,), device=device, dtype=torch.int32
        )
        summary_cache._step4_csa_numerator_cache = torch.zeros(
            (capacity, 1, proxy_dim), device=device, dtype=torch.float32
        )
        summary_cache._step4_csa_denominator_cache = torch.zeros_like(
            summary_cache._step4_csa_numerator_cache
        )
        summary_cache._step4_csa_max_cache = torch.full(
            (capacity, 1), float("-inf"), device=device, dtype=torch.float32
        )
        summary_cache._step4_csa_active_token_k = torch.zeros(
            (capacity, self.region_size, 1, proxy_dim),
            device=device,
            dtype=self._decode_out_dtype,
        )
        summary_cache._step4_csa_active_token_z = torch.zeros_like(
            summary_cache._step4_csa_active_token_k
        )
        summary_cache._step4_csa_active_token_valid = torch.zeros(
            (capacity, self.region_size), device=device, dtype=torch.uint8
        )

    def bind_summary_cache(
        self,
        summary_cache: AscendStep4SparseSummaryCache,
        *,
        initialize_runtime_state: bool = True,
    ) -> None:
        """Bind the page-backed summary sidecar and preallocate all scratch.

        Called twice per layer: once during memory profiling with a 1-page
        dummy cache (``initialize_runtime_state=False``; scratch sizing is
        host-config based so the real bind never grows it), and once with the
        real page-backed cache.
        """
        self._dsa_scratch_bound = False
        if initialize_runtime_state:
            self._summary_cache = summary_cache
            self._summary_cache_config = summary_cache.config
            self._initialize_csa_summary_state(summary_cache)
        device = summary_cache.mean_cache.device
        max_layout_rows = self._round_up(
            max(self.max_num_batched_tokens, self.max_num_seqs), 64
        )
        for name, dtype in (
            ("csa_layout_flat_slot", torch.int64),
            ("csa_layout_token_slots", torch.int64),
            ("csa_layout_token_positions", torch.int64),
            ("csa_layout_reset_slots", torch.int64),
            ("csa_layout_token_valid", torch.bool),
        ):
            self._get_dsa_tensor_buffer_at_least(
                name, (max_layout_rows,), device=device, dtype=dtype
            )
        self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_region_ids",
            (max_layout_rows,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_row_map",
            (max_layout_rows * self.region_size,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_reset_map",
            (max_layout_rows,),
            device=device,
            dtype=torch.int32,
        )
        # Decode indexer-score workspace, shared by all layers (they run
        # sequentially on one stream and each consumes its scores before the
        # next layer overwrites them). With MTP verify batches every decode
        # request contributes K+1 scoring rows.
        decode_rows_per_req = 1 + (self.num_speculative_tokens if self.num_speculative_tokens > 0 else 0)
        self._get_dsa_tensor_buffer_at_least(
            "sparse_decode_logits",
            (
                self.num_kv_groups * self.max_num_seqs * decode_rows_per_req,
                self._max_regions,
            ),
            device=device,
            dtype=torch.float32,
        )
        self._dsa_scratch_bound = True

    @staticmethod
    def _round_up(value: int, alignment: int) -> int:
        return (int(value) + alignment - 1) // alignment * alignment

    @staticmethod
    def _begin_csa_allocation_check(
        summary_cache: AscendStep4SparseSummaryCache,
    ) -> torch.Tensor:
        """Reset and return the graph-stable device allocation status."""
        success = summary_cache._step4_csa_allocation_success
        success.fill_(1)
        return success

    @staticmethod
    def _assert_csa_allocation_success(success: torch.Tensor) -> None:
        """Fail closed after a device-side active-slot allocation attempt."""
        if _is_graph_capturing():
            # The deferred assert cannot be observed inside a captured graph;
            # capacity is a host-config policy validated by the eager warmup
            # runs that precede capture.
            return
        torch._assert_async(success, _CSA_ALLOCATION_ERROR)

    # -- summary cache update --------------------------------------------------

    def _get_or_build_runtime_layout(
        self,
        *,
        summary_cache: AscendStep4SparseSummaryCache,
        md: AscendStep4DSAMetadata,
        num_actual_tokens: int,
    ) -> AscendStep4DSARuntimeLayout:
        live_tokens = int(num_actual_tokens)
        padded_rows = self._round_up(live_tokens, 64)
        device = md.slot_mapping.device
        token_flat_slot = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_flat_slot", (padded_rows,), device=device, dtype=torch.int64
        )
        token_slots = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_token_slots", (padded_rows,), device=device, dtype=torch.int64
        )
        token_positions = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_token_positions",
            (padded_rows,),
            device=device,
            dtype=torch.int64,
        )
        reset_slots = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_reset_slots", (padded_rows,), device=device, dtype=torch.int64
        )
        token_valid = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_token_valid", (padded_rows,), device=device, dtype=torch.bool
        )
        (
            token_flat_slot,
            _,
            token_positions,
            reset_slots,
            token_valid,
        ) = build_decode_summary_layout_step3p5(
            md.slot_mapping,
            md.query_start_loc,
            md.seq_lens,
            num_actual_tokens=live_tokens,
            num_pages=int(summary_cache.num_pages),
            page_size=int(summary_cache.page_size),
            region_block_size=int(summary_cache.region_block_size),
            summaries_per_page=int(summary_cache.summaries_per_page),
            padded_rows=padded_rows,
            out_flat_slot=token_flat_slot,
            out_token_slots=token_slots,
            out_token_positions=token_positions,
            out_reset_slots=reset_slots,
            out_token_valid=token_valid,
        )
        return AscendStep4DSARuntimeLayout(
            token_flat_slot=token_flat_slot,
            token_positions=token_positions,
            reset_slots=reset_slots,
            token_valid=token_valid,
        )

    def _maybe_update_summary_cache(
        self,
        *,
        summary_cache: AscendStep4SparseSummaryCache,
        md: AscendStep4DSAMetadata,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> None:
        """Fold this step's indexer K/Z tokens into the paged summary state.

        Mirrors upstream: a pure decode batch (single token per request)
        stages tokens and flushes completed regions; any batch containing
        prefill tokens (mixed included) runs the ordered two-pass update over
        all tokens of the batch. MTP verify batches (uniform K+1 rows per
        decode request) stage their rows without flushing and materialize
        fully-committed regions through the deferred flush, so rejected
        tokens never pollute the fp8 page cache.
        """
        num_tokens = int(md.num_actual_tokens)
        if num_tokens <= 0:
            return
        layout = self._get_or_build_runtime_layout(
            summary_cache=summary_cache,
            md=md,
            num_actual_tokens=num_tokens,
        )
        index_k = index_k[:num_tokens].contiguous()
        index_z = index_z[:num_tokens].contiguous()
        allocation_success = self._begin_csa_allocation_check(summary_cache)
        mean_cache = summary_cache.mean_cache.view(torch.float8_e4m3fn)
        common = dict(
            num_pages=int(summary_cache.num_pages),
            summaries_per_page=int(summary_cache.summaries_per_page),
            proxy_dim=int(self.proxy_dim),
            region_block_size=int(self.region_size),
        )
        verify_batch = (
            self.num_speculative_tokens > 0
            and md.decode is not None
            and md.decode.query_len > 1
        )
        if verify_batch:
            self._update_summary_cache_verify(
                summary_cache=summary_cache,
                md=md,
                layout=layout,
                index_k=index_k,
                index_z=index_z,
                allocation_success=allocation_success,
                mean_cache=mean_cache,
                common=common,
            )
            return
        use_decode_update = md.num_prefills == 0 and md.decode is not None
        if use_decode_update:
            csa_compact_decode_update(
                mean_cache,
                summary_cache._step4_csa_active_region_ids,
                summary_cache._step4_csa_active_slot_by_region,
                allocation_success,
                summary_cache._step4_csa_numerator_cache,
                summary_cache._step4_csa_denominator_cache,
                summary_cache._step4_csa_max_cache,
                summary_cache._step4_csa_active_token_k,
                summary_cache._step4_csa_active_token_z,
                summary_cache._step4_csa_active_token_valid,
                layout.token_flat_slot,
                layout.reset_slots,
                layout.token_valid,
                layout.token_positions,
                index_k,
                index_z,
                live_rows=int(layout.token_flat_slot.numel()),
                **common,
            )
        else:
            device = index_k.device
            padded_rows = self._round_up(num_tokens, 64)
            scratch_region_ids = self._get_dsa_tensor_buffer_at_least(
                "csa_prefill_scratch_region_ids",
                (padded_rows,),
                device=device,
                dtype=torch.int32,
            )
            scratch_row_map = self._get_dsa_tensor_buffer_at_least(
                "csa_prefill_scratch_row_map",
                (padded_rows * self.region_size,),
                device=device,
                dtype=torch.int32,
            )
            scratch_reset_map = self._get_dsa_tensor_buffer_at_least(
                "csa_prefill_scratch_reset_map",
                (padded_rows,),
                device=device,
                dtype=torch.int32,
            )
            csa_compact_prefill_update(
                mean_cache,
                summary_cache._step4_csa_active_region_ids,
                summary_cache._step4_csa_active_slot_by_region,
                allocation_success,
                summary_cache._step4_csa_numerator_cache,
                summary_cache._step4_csa_denominator_cache,
                summary_cache._step4_csa_max_cache,
                summary_cache._step4_csa_active_token_valid,
                scratch_region_ids,
                scratch_row_map,
                scratch_reset_map,
                layout.token_flat_slot,
                layout.reset_slots,
                layout.token_valid,
                layout.token_positions,
                index_k,
                index_z,
                live_rows=num_tokens,
                **common,
            )
        self._assert_csa_allocation_success(allocation_success)

    def _update_summary_cache_verify(
        self,
        *,
        summary_cache: AscendStep4SparseSummaryCache,
        md: AscendStep4DSAMetadata,
        layout: AscendStep4DSARuntimeLayout,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        allocation_success: torch.Tensor,
        mean_cache: torch.Tensor,
        common: dict,
    ) -> None:
        """Summary update for MTP verify batches (uniform K+1 decode rows).

        Deferred flush first: any region whose last position is below the
        committed context (``seq_len - query_len``) is provably final and is
        materialized into the fp8 page cache. The verify rows themselves are
        only staged, so the next round's rewrite of rejected positions
        replaces their contribution exactly (the Ascend equivalent of the
        upstream MTP transaction's rollback-to-accepted-prefix). In a mixed
        batch, the prefill tail still runs the ordinary ordered fold over
        its own rows; a request never contributes rows to both segments.
        """
        d = md.decode
        n = md.num_decodes
        query_len = int(d.query_len)
        # 1) Materialize fully-committed tail regions (releases their slots
        #    for reuse by this step's staging).
        csa_deferred_flush(
            mean_cache,
            summary_cache._step4_csa_active_region_ids,
            summary_cache._step4_csa_active_slot_by_region,
            summary_cache._step4_csa_numerator_cache,
            summary_cache._step4_csa_denominator_cache,
            summary_cache._step4_csa_max_cache,
            summary_cache._step4_csa_active_token_k,
            summary_cache._step4_csa_active_token_z,
            summary_cache._step4_csa_active_token_valid,
            d.seq_lens,
            d.block_table,
            num_reqs=n,
            query_len=query_len,
            num_pages=int(summary_cache.num_pages),
            summaries_per_page=int(summary_cache.summaries_per_page),
            region_block_size=int(summary_cache.region_block_size),
            page_size=int(summary_cache.page_size),
        )
        # 2) Stage the verify rows (decode segment leads the batch).
        nv = int(md.num_decode_tokens)
        csa_verify_stage_update(
            mean_cache,
            summary_cache._step4_csa_active_region_ids,
            summary_cache._step4_csa_active_slot_by_region,
            allocation_success,
            summary_cache._step4_csa_numerator_cache,
            summary_cache._step4_csa_denominator_cache,
            summary_cache._step4_csa_max_cache,
            summary_cache._step4_csa_active_token_k,
            summary_cache._step4_csa_active_token_z,
            summary_cache._step4_csa_active_token_valid,
            layout.token_flat_slot[:nv],
            layout.reset_slots[:nv],
            layout.token_valid[:nv],
            layout.token_positions[:nv],
            index_k[:nv],
            index_z[:nv],
            live_rows=nv,
            query_len=query_len,
            **common,
        )
        # 3) Mixed batch: fold the prefill tail with the ordinary ordered
        #    two-pass update (prefill requests own private pages; their
        #    regions never collide with the verify requests' regions).
        if md.num_prefills > 0:
            device = index_k.device
            prefill_rows = int(md.num_actual_tokens) - nv
            padded_rows = self._round_up(prefill_rows, 64)
            scratch_region_ids = self._get_dsa_tensor_buffer_at_least(
                "csa_prefill_scratch_region_ids",
                (padded_rows,),
                device=device,
                dtype=torch.int32,
            )
            scratch_row_map = self._get_dsa_tensor_buffer_at_least(
                "csa_prefill_scratch_row_map",
                (padded_rows * self.region_size,),
                device=device,
                dtype=torch.int32,
            )
            scratch_reset_map = self._get_dsa_tensor_buffer_at_least(
                "csa_prefill_scratch_reset_map",
                (padded_rows,),
                device=device,
                dtype=torch.int32,
            )
            csa_compact_prefill_update(
                mean_cache,
                summary_cache._step4_csa_active_region_ids,
                summary_cache._step4_csa_active_slot_by_region,
                allocation_success,
                summary_cache._step4_csa_numerator_cache,
                summary_cache._step4_csa_denominator_cache,
                summary_cache._step4_csa_max_cache,
                summary_cache._step4_csa_active_token_valid,
                scratch_region_ids,
                scratch_row_map,
                scratch_reset_map,
                layout.token_flat_slot[nv:],
                layout.reset_slots[nv:],
                layout.token_valid[nv:],
                layout.token_positions[nv:],
                index_k[nv:],
                index_z[nv:],
                live_rows=prefill_rows,
                **common,
            )
        self._assert_csa_allocation_success(allocation_success)

    # -- sparse attention ------------------------------------------------------

    def _forward_sparse_gqa_decode(
        self,
        *,
        summary_cache: AscendStep4SparseSummaryCache,
        md: AscendStep4DSAMetadata,
        query: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        d = md.decode
        assert d is not None
        if d.query_len != 1:
            if self.num_speculative_tokens <= 0:
                raise NotImplementedError(
                    "Ascend Step4 DSA paged summary path received a "
                    "multi-token decode batch without MTP enabled "
                    f"(query_len={d.query_len})."
                )
            raise NotImplementedError(
                "Ascend Step4 DSA verify attention must be dispatched through"
                " _forward_sparse_gqa_verify (query_len="
                f"{d.query_len})."
            )
        n = md.num_decodes
        groups = self.num_kv_groups
        # Region scoring over the static selector capacity, the whole batch
        # in one kernel launch. Dead columns hold stale scores, but the
        # selector only reads columns below each row's live history length.
        logits = self._get_dsa_tensor_buffer_at_least(
            "sparse_decode_logits",
            (groups * n, self._max_regions),
            device=index_q.device,
            dtype=torch.float32,
        )
        indexer_logits_paged(
            index_q[:n],
            weights[:n],
            summary_cache.mean_cache.view(torch.float8_e4m3fn),
            d.block_table,
            torch.arange(n, device=index_q.device, dtype=torch.int32),
            out=logits,
            summaries_per_page=int(summary_cache.summaries_per_page),
        )
        seq_lens_i32 = d.seq_lens.to(torch.int32)
        packed, counts = decode_sparse_meta(
            logits,
            seq_lens_i32,
            d.block_table,
            topk=self.topk_regions,
            region_size=self.region_size,
            regions_per_page=self.regions_per_page,
        )
        out, _ = sparse_attention_decode(
            query,
            key_flat,
            value_flat,
            packed,
            counts,
            seq_lens_i32,
            num_kv_groups=groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
            block_regions=16,
        )
        output[:n] = out.to(output.dtype)

    def _forward_sparse_gqa_verify(
        self,
        *,
        summary_cache: AscendStep4SparseSummaryCache,
        md: AscendStep4DSAMetadata,
        query: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse attention for an MTP verify batch (K+1 rows per request).

        Follows the reference flow's flattened-decode design (its
        ``_flatten_decode_q_le4_metadata`` path): every verify row becomes a
        single-token decode request, runs its own indexer top-k selection,
        and the exact attention goes through ``sparse_attention_decode``.
        Each row's ``kv_seqlen`` is its own absolute position + 1, so the
        decode kernel's per-slot visible length
        ``clamp(kv_seqlen - start_token, 0, region_size)`` reproduces the
        per-row causal mask the prefill kernel used to express through
        per-region ``valid`` counts, while ``decode_sparse_meta`` keeps the
        row's current region out of top-k competition and force-appends it
        -- the same contract the single-token decode path guarantees.
        Everything is device-driven on static per-bucket shapes, so uniform
        verify batches stay ACL-graph capturable.
        """
        d = md.decode
        n = md.num_decodes
        k = int(d.query_len)
        total = n * k
        groups = self.num_kv_groups
        logits = self._get_dsa_tensor_buffer_at_least(
            "sparse_decode_logits",
            (groups * total, self._max_regions),
            device=index_q.device,
            dtype=torch.float32,
        )
        # Token -> request map: each request contributes K consecutive rows.
        token_to_req = (
            torch.arange(total, device=index_q.device, dtype=torch.int32) // k
        )
        indexer_logits_paged(
            index_q[:total],
            weights[:total],
            summary_cache.mean_cache.view(torch.float8_e4m3fn),
            d.block_table[:n],
            token_to_req,
            out=logits,
            summaries_per_page=int(summary_cache.summaries_per_page),
        )
        # Flatten the verify rows to single-token decode requests: row r of
        # request i carries its own context end, i.e. kv_seqlen = position +
        # 1. The decode kernel then bounds each selected region by
        # clamp(kv_seqlen - start_token, 0, region_size), which is the
        # per-row causal mask.
        row_kv_seqlens = positions[:total].to(torch.int32) + 1
        # Per-row block-table rows, gathered on device so the flatten stays
        # graph-capturable; this mirrors the reference's metadata-flatten
        # kernel expanding per-request rows to per-token rows.
        row_block_table = d.block_table[:n][token_to_req.to(torch.long)]
        packed, counts = decode_sparse_meta(
            logits,
            row_kv_seqlens,
            row_block_table,
            topk=self.topk_regions,
            region_size=self.region_size,
            regions_per_page=self.regions_per_page,
        )
        out, _ = sparse_attention_decode(
            query,
            key_flat,
            value_flat,
            packed,
            counts,
            row_kv_seqlens,
            num_kv_groups=groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
            block_regions=16,
        )
        output[:total] = out.to(output.dtype)

    def _score_regions(
        self,
        summary_cache: AscendStep4SparseSummaryCache,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        num_tokens: int,
        num_rows: int,
        seq_lens_host: list,
        context_lens_host: list,
    ) -> torch.Tensor:
        """Indexer scores for the prefill rows, against the paged summary.

        Same contract as the reference ``score_regions``: a
        [groups * total_q, width] fp32 tensor in group-major row order, width
        sized to the live history. The whole packed batch is scored in one
        kernel launch: each query token reads the fp8 summary pages through
        its own request's block-table row (widened in-register). Dead columns
        past a row's live history hold zeros or stale scores, which the
        top-k stage never reads because each row's candidate range is bounded
        by its own history length.
        """
        groups = self.num_kv_groups
        # ``seq_lens`` from the common attention metadata hold each request's
        # FULL history (past + this chunk's query tokens); the per-row query
        # length is the difference.
        query_lens_list = [
            int(seq_lens_host[row]) - int(context_lens_host[row])
            for row in range(num_rows)
        ]
        last_positions = [
            int(seq_lens_host[row]) - 1
            for row in range(num_rows)
            if query_lens_list[row] > 0
        ]
        width = max(
            (position // self.region_size for position in last_positions), default=0
        )
        logits = torch.zeros(
            (groups * num_tokens, max(width, 1)),
            device=index_q.device,
            dtype=torch.float32,
        )
        if num_tokens == 0:
            return logits
        mean_cache = summary_cache.mean_cache.view(torch.float8_e4m3fn)
        # Token -> block-table row map for the packed batch, built once on the
        # host (the query lengths already live there) and shipped in one copy.
        token_to_req = torch.repeat_interleave(
            torch.arange(num_rows, dtype=torch.int32),
            torch.tensor(query_lens_list, dtype=torch.int64),
        ).to(index_q.device)
        indexer_logits_paged(
            index_q[:num_tokens],
            weights[:num_tokens],
            mean_cache,
            block_table[:num_rows],
            token_to_req,
            out=logits,
            summaries_per_page=int(summary_cache.summaries_per_page),
        )
        return logits

    def _forward_sparse_gqa_prefill(
        self,
        *,
        summary_cache: AscendStep4SparseSummaryCache,
        md: AscendStep4DSAMetadata,
        query: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        token_offset: int,
    ) -> None:
        p = md.prefill
        assert p is not None
        num_reqs = md.num_prefills
        num_query_tokens = p.num_query_tokens

        logits = self._score_regions(
            summary_cache,
            index_q,
            weights,
            p.block_table,
            num_query_tokens,
            num_reqs,
            p.seq_lens_host,
            p.context_lens_host,
        )
        groups = self.num_kv_groups
        q_positions = positions[token_offset : token_offset + num_query_tokens].to(
            torch.int32
        )
        # Token -> request map straight from the device start-loc.
        requests = (
            torch.searchsorted(
                p.query_start_loc,
                torch.arange(
                    num_query_tokens, device=index_q.device, dtype=p.query_start_loc.dtype
                ),
                right=True,
            )
            - 1
        ).to(torch.int32)
        if groups > 1:
            q_positions = q_positions.repeat(groups).contiguous()
            requests = requests.repeat(groups).contiguous()
        packed, counts = prefill_sparse_meta(
            logits,
            q_positions,
            p.block_table,
            requests,
            topk=self.topk_regions,
            region_size=self.region_size,
            regions_per_page=self.regions_per_page,
        )
        out, _ = sparse_attention_prefill(
            query,
            key_flat,
            value_flat,
            packed,
            counts,
            num_kv_groups=groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
            block_regions=2,
        )
        output[token_offset : token_offset + num_query_tokens] = out.to(output.dtype)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output.zero_()
        md = attn_metadata[layer.layer_name]
        if not isinstance(md, AscendStep4DSAMetadata):
            return output.zero_()

        summary_cache = self._summary_cache
        if summary_cache is None:
            raise RuntimeError(
                "Step4 sparse summary cache must be bound during KV cache "
                "initialization before attention forward."
            )
        num_tokens = md.num_actual_tokens

        self._maybe_update_summary_cache(
            summary_cache=summary_cache,
            md=md,
            index_k=index_k,
            index_z=index_z,
        )

        if isinstance(kv_cache, (tuple, list)):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        key_flat = key_cache.view(-1, self.num_kv_heads, self.head_size)
        value_flat = value_cache.view(-1, self.num_kv_heads, self.head_size)

        if md.num_decodes > 0:
            n = md.num_decodes
            decode_query_len = int(md.decode.query_len) if md.decode else 1
            if decode_query_len > 1:
                # MTP verify batch: every decode request contributes K+1
                # consecutive rows (positions ctx..ctx+K).
                total = n * decode_query_len
                self._forward_sparse_gqa_verify(
                    summary_cache=summary_cache,
                    md=md,
                    query=query[:total].contiguous(),
                    key_flat=key_flat,
                    value_flat=value_flat,
                    index_q=index_q[:total],
                    weights=weights[:total],
                    positions=positions,
                    output=output,
                )
            else:
                self._forward_sparse_gqa_decode(
                    summary_cache=summary_cache,
                    md=md,
                    query=query[:n].contiguous(),
                    key_flat=key_flat,
                    value_flat=value_flat,
                    index_q=index_q[:n],
                    weights=weights[:n],
                    output=output,
                )

        if md.num_prefills > 0:
            start = md.num_decode_tokens
            self._forward_sparse_gqa_prefill(
                summary_cache=summary_cache,
                md=md,
                query=query[start:num_tokens].contiguous(),
                key_flat=key_flat,
                value_flat=value_flat,
                index_q=index_q[start:num_tokens],
                weights=weights[start:num_tokens],
                positions=positions,
                output=output,
                token_offset=start,
            )
        return output
