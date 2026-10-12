# SPDX-License-Identifier: Apache-2.0
"""Ascend Step-4 DSA (DeepSeek-style sparse attention) backend.

Structure mirrors the upstream CUDA implementation
(``vllm.models.step5.sparse_attention.Step5DSAAttentionImpl``): per-region
summaries live in a page-backed fp8 side cache whose lifetime follows the
owner KV blocks (``vllm_ascend.models.step5.summary_cache``), the per-step
token -> region layout is built from ``slot_mapping`` by the upstream Triton
layout builder, and ``forward`` is ``_maybe_update_summary_cache`` followed by
the sparse-GQA decode/prefill paths.

The CuTeDSL-only operators are replaced by the Triton kernels in
``vllm_ascend.models.step5.summary_kernels`` (region update). The indexer
region scoring, causal masking, TopK selection and ascending sort are fused
into the aclnn ``IndexerRegionScore`` operator
(``vllm_ascend.models.step5.indexer_region_score``); the exact sparse
attention runs in the aclnn ``GenericBlockSparseAttention`` operator
(``vllm_ascend.models.step5.generic_block_sparse_attention``), which
consumes the indexer's logical region ids directly (current-region append
and per-token causal truncation happen inside the operator), so the
physical-page pack kernels and the Triton attention kernels of
``dsa_kernels.py`` are no longer on the serving path. The operator's AICPU
scheduling metadata is generated once per step in the metadata builder --
outside the model graph, like the FA3 scheduler tiling -- and shared across
all layers, because it schedules from position-derived block counts that
are identical layer to layer.

Spec-decode verify batches (multi-token decode rows, ``query_len = K+1``)
follow the upstream ``Step5DSAMTP`` contract with an equivalent Ascend
mechanism: verify rows are *staged* into the active-slot tail state instead
of folded (staging is last-write-wins per (region, offset), so the next
round's rewrite of rejected positions is an exact rollback to the accepted
prefix), and completed regions are only materialized into the fp8 page
cache by the deferred flush once their last position is below the request's
committed context. The verify attention keeps each request as one
multi-token varlen request (``query_len = K+1``): the fused indexer's
per-token mask3 bound and the attention operator's right-aligned causal
truncation hold every row to its own position as causal bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.step5.sparse_summary_cache import Step5DSAScratchWorkspace
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

from vllm_ascend.models.step5.generic_block_sparse_attention import (
    gbsa_expected_counts,
    gbsa_sparse_attention,
    gbsa_sparse_attention_metadata,
)
from vllm_ascend.worker.step5_csa_slot_manager import (
    CSA_SLOTS_PER_REQUEST,
    CSAMetadata,
    CSASlotManager,
)
from vllm_ascend.models.step5.indexer_region_score import (
    indexer_score_topk,
)
from vllm_ascend.models.step5.summary_cache import AscendStep5SparseSummaryCache, CSAPrefillState
from vllm_ascend.models.step5.summary_kernels import (
    build_decode_summary_layout_step3p5,
    csa_compact_decode_update,
    csa_compact_prefill_update,
    csa_compact_prefill_update_default,
    csa_prefill_handoff,
    csa_deferred_flush,
    csa_verify_stage_update,
)

logger = init_logger(__name__)

_CSA_ALLOCATION_ERROR = (
    "Step5 DSA active-slot capacity is exhausted. Lower max_num_seqs or "
    "increase the CSA capacity policy."
)


def _is_graph_capturing() -> bool:
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


@dataclass
class AscendStep5DSARuntimeLayout:
    """Per-step token -> physical summary region mapping (see upstream)."""

    token_flat_slot: torch.Tensor
    token_positions: torch.Tensor
    reset_slots: torch.Tensor
    token_valid: torch.Tensor


class AscendStep5DSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_STEP4_DSA"

    @staticmethod
    def get_impl_cls():
        return AscendStep5DSAImpl

    @staticmethod
    def get_builder_cls():
        return AscendStep5DSAMetadataBuilder

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
class AscendStep5DSAPrefillMetadata:
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

    gbsa_metadata: torch.Tensor = None
    """Per-step AICPU GenericBlockSparseAttention scheduling tensor, built
    once in the builder from the query positions and shared across all
    layers (the scheduled counts are position-derived, layer-independent)."""

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    @property
    def num_query_tokens_padded(self) -> int:
        return int(self.query_start_loc[-1])


@dataclass
class AscendStep5DSADecodeMetadata:
    seq_lens: torch.Tensor
    """(num_decodes,) context length including the current token."""

    block_table: torch.Tensor
    """(num_decodes, max_pages) int32 rows for the decode requests."""

    query_len: int = 1
    """Uniform decode query length K (1 for plain decode, 1+num_spec for
    MTP verify batches)."""

    gbsa_metadata: torch.Tensor = None
    """Per-step AICPU scheduling tensor for the (possibly flattened verify)
    decode path, built once in the builder and shared across all layers."""


@dataclass
class AscendStep5DSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    """(num_reqs + 1,) int32 full-batch token start locations, including
    padding (the worker's persistent int32 start-loc buffer, so slices are
    zero-copy and directly usable as the fused indexer's cu_seq_lens)."""
    query_start_loc_int64: torch.Tensor
    """(num_reqs + 1,) int64 copy of ``query_start_loc``, refreshed in
    place into a persistent builder buffer each build: the aclnn
    GenericBlockSparseAttention operator requires ``cu_seqlens_q`` in
    int64 (the fused indexer consumes the int32 original)."""
    seq_lens: torch.Tensor
    """(num_reqs,) int32 full-batch sequence lengths, including padding
    rows (the worker's persistent int32 seq-len buffer)."""
    csa_prefill_plan: CSAMetadata | None = None
    prefill: AscendStep5DSAPrefillMetadata | None = None
    decode: AscendStep5DSADecodeMetadata | None = None


class AscendStep5DSAMetadataBuilder(
    AttentionMetadataBuilder[AscendStep5DSAMetadata]
):
    # Decode forward is fully tensor-driven (see AscendStep5DSAImpl
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
        self._csa_layer_names = layer_names
        self._csa_static_context = vllm_config.compilation_config.static_forward_context
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        # DSA selector geometry, read from the layer impls (all layers share
        # one config) -- the GBSA AICPU metadata is scheduled once per step
        # here in the builder, outside the model graph (FA3 tiling pattern).
        impl = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ].impl
        self.gbsa_num_heads = impl.num_heads
        self.gbsa_num_kv_groups = impl.num_kv_groups
        self.gbsa_head_size = impl.head_size
        self.gbsa_region_size = impl.region_size
        self.gbsa_topk = impl.topk_regions
        self.gbsa_block_size = kv_cache_spec.block_size
        self.capture_sizes = set(
            vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
        # Graph-replay metadata buffers: on capture-sized steps the AICPU
        # tensor must live at a stable address (baked into the captured
        # graph) and be refreshed in place each step -- the FA3
        # scheduler_buffers pattern.
        self.gbsa_meta_buffers: dict[tuple, torch.Tensor] = {}
        # Persistent int64 staging for query_start_loc: the aclnn
        # GenericBlockSparseAttention operator requires cu_seqlens_q in
        # int64. build() refreshes it in place so captured verify graphs
        # observe the current step's start locations at a stable address
        # (a fresh .to(int64) tensor per step would leave the graph reading
        # a stale allocation).
        self.query_start_loc_int64_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens + 1,
            dtype=torch.int64,
            device=device,
        )

    def _build_gbsa_metadata(
        self,
        positions: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        block_table: torch.Tensor,
        max_query_len: int,
        *,
        path: str,
        graph_shape: bool,
    ) -> torch.Tensor:
        """One AICPU scheduling pass per step for these query positions.

        On capture-sized shapes the fresh tensor is copied into a persistent
        buffer keyed by the shape signature, so a captured graph replays
        against a stable address whose content is refreshed every step.
        """
        counts = gbsa_expected_counts(
            positions,
            groups=self.gbsa_num_kv_groups,
            region_size=self.gbsa_region_size,
            topk=self.gbsa_topk,
        )
        metadata = gbsa_sparse_attention_metadata(
            counts,
            num_heads=self.gbsa_num_heads,
            num_kv_groups=self.gbsa_num_kv_groups,
            head_dim=self.gbsa_head_size,
            region_size=self.gbsa_region_size,
            topk=self.gbsa_topk,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            max_query_len=max_query_len,
            max_pages=int(block_table.shape[1]),
            page_size=self.gbsa_block_size,
        )
        if not graph_shape:
            return metadata
        buffer_key = (
            path,
            tuple(counts.shape),
            int(cu_seqlens_q.shape[0]) - 1,
            max_query_len,
        )
        buffer = self.gbsa_meta_buffers.get(buffer_key)
        if buffer is None:
            self.gbsa_meta_buffers[buffer_key] = metadata
            return metadata
        buffer.copy_(metadata)
        return buffer

    def build_csa_slot_plans(
        self, manager: CSASlotManager, req_ids: list[str | None],
        metadata: AscendStep5DSAMetadata, plan_prefill: bool,
        prompt_lens: list[int],
    ) -> None:
        sizes = {
            self._csa_static_context[name].impl.region_size
            for name in self._csa_layer_names
        }
        if len(sizes) != 1:
            raise ValueError("CSA slot plans require one region size per attention group")
        region_size = sizes.pop()
        num_decodes = metadata.num_decodes
        prefill = metadata.prefill
        if prefill is not None:
            prefill_ids = req_ids[num_decodes:num_decodes + metadata.num_prefills]
            if all(req_id is None for req_id in prefill_ids):
                metadata.csa_prefill_plan = manager.dummy_prefill(metadata.num_prefills)
            elif plan_prefill and all(req_id is not None for req_id in prefill_ids):
                metadata.csa_prefill_plan = manager.plan(
                    prefill_ids, prefill.context_lens_host, prefill.seq_lens_host, region_size,
                    prompt_lens,
                )
            else:
                raise ValueError("CSA prefill requires a complete runner prompt plan")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendStep5DSAMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_int64 = self.query_start_loc_int64_buffer[
            : query_start_loc.shape[0]
        ]
        query_start_loc_int64.copy_(query_start_loc)
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        # Capture-sized steps hand the graph a persistent metadata buffer;
        # all other shapes run eager on fresh tensors (FA3 tiling pattern).
        num_input_tokens = (
            getattr(common_attn_metadata, "num_input_tokens", 0)
            or common_attn_metadata.num_actual_tokens
        )
        graph_shape = num_input_tokens in self.capture_sizes

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        prefill_metadata: AscendStep5DSAPrefillMetadata | None = None
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
            prefill_metadata = AscendStep5DSAPrefillMetadata(
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
            # GBSA AICPU scheduling, once per step outside the model graph.
            # Per-token positions: context + in-chunk offset, straight from
            # the (rebased) start-loc; the scheduled counts depend on these
            # positions only, so every layer reuses this one tensor.
            qsl = prefill_metadata.query_start_loc.to(torch.int64)
            num_query_tokens = prefill_metadata.num_query_tokens
            token_idx = torch.arange(
                num_query_tokens, device=seq_lens.device, dtype=torch.int64
            )
            reqs = (torch.searchsorted(qsl, token_idx, right=True) - 1).to(
                torch.long
            )
            prefill_positions = (
                prefill_context_lens.to(torch.int64)[reqs]
                + token_idx
                - qsl[reqs]
            )
            prefill_metadata.gbsa_metadata = self._build_gbsa_metadata(
                prefill_positions,
                prefill_metadata.query_start_loc,
                prefill_seq_lens,
                block_table,
                prefill_metadata.max_query_len,
                path="prefill",
                graph_shape=graph_shape,
            )

        decode_metadata: AscendStep5DSADecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(
                num_decodes, num_decode_tokens, decode_query_len
            )
            gbsa_metadata = None
            if active_decodes > 0:
                decode_seq_lens = seq_lens[:active_decodes]
                if decode_query_len == 1:
                    # Plain decode: one query per request at its own position.
                    positions = decode_seq_lens.to(torch.int64) - 1
                    cu_seq_lens = torch.arange(
                        active_decodes + 1, device=seq_lens.device, dtype=torch.int64
                    )
                    seqused_kv = decode_seq_lens
                    max_query_len = 1
                else:
                    # MTP verify multi-token varlen: each request stays one
                    # query_len=K+1 request spanning kv positions
                    # ctx..ctx+K (ctx = seq_len - q_len). The per-token
                    # positions drive the scheduled counts; the
                    # request-level start-loc head (decode rows lead the
                    # batch) and the full seq lens anchor the right-aligned
                    # causal bound.
                    total = active_decodes * decode_query_len
                    row_base = (
                        decode_seq_lens.to(torch.int64) - decode_query_len
                    ).repeat_interleave(decode_query_len)
                    positions = row_base + (
                        torch.arange(total, device=seq_lens.device, dtype=torch.int64)
                        % decode_query_len
                    )
                    cu_seq_lens = query_start_loc[: active_decodes + 1]
                    seqused_kv = decode_seq_lens
                    max_query_len = decode_query_len
                gbsa_metadata = self._build_gbsa_metadata(
                    positions,
                    cu_seq_lens,
                    seqused_kv,
                    block_table,
                    max_query_len,
                    path="decode",
                    graph_shape=graph_shape,
                )
            decode_metadata = AscendStep5DSADecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                query_len=decode_query_len,
                gbsa_metadata=gbsa_metadata,
            )

        return AscendStep5DSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            query_start_loc=query_start_loc,
            query_start_loc_int64=query_start_loc_int64,
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


class AscendStep5DSAImpl(AttentionImplBase[AscendStep5DSAMetadata]):
    # Read by AscendStep5SparseSummaryCacheLayer when sizing/binding the
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
        prefill_mnd_slot_allocator: str = "default",
    ) -> None:
        if prefill_mnd_slot_allocator not in ("default", "runner"):
            raise ValueError("step5_prefill_mnd_slot_allocator must be 'default' or 'runner'")
        self.use_runner_prefill_slots = prefill_mnd_slot_allocator == "runner"
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
                "Step5 DSA requires block_size to be a multiple of the region "
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
                "Step5 DSA MTP requires num_speculative_tokens + 1 <= "
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
        self._summary_cache: AscendStep5SparseSummaryCache | None = None
        self._summary_cache_config = None
        self._dsa_scratch_workspace = Step5DSAScratchWorkspace()
        self._dsa_scratch_bound = False

    # -- upstream-compatible side-storage contract ---------------------------

    @property
    def sparse_region_block_size(self) -> int:
        return self.region_size

    def bind_scratch_workspace(self, workspace: Step5DSAScratchWorkspace) -> None:
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
                    "Step5 DSA scratch capacity was exceeded after bind: "
                    f"name={name!r}, requested_shape={shape}, "
                    f"required_elements={int(required)}."
                )
            if _is_graph_capturing():
                raise RuntimeError(
                    "Step5 DSA tensor scratch buffers must be initialized "
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
        prefill_capacity = CSA_SLOTS_PER_REQUEST * int(self.max_num_seqs) if self.use_runner_prefill_slots else 0
        prefill_bytes = prefill_capacity * (
            (2 * proxy_dim + 1) * torch.tensor([], dtype=torch.float32).element_size()
            + torch.tensor([], dtype=torch.int32).element_size()
            + region_block_size * torch.tensor([], dtype=torch.uint8).element_size()
        )
        return int(
            prefill_bytes + capacity * torch.tensor([], dtype=torch.int32).element_size()
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
        summary_cache: AscendStep5SparseSummaryCache,
        *,
        capacity: int | None = None,
    ) -> None:
        capacity = int(capacity or self._csa_active_region_capacity())
        device = summary_cache.mean_cache.device
        total_regions = int(summary_cache.num_pages) * int(
            summary_cache.summaries_per_page
        )
        proxy_dim = int(self.proxy_dim)
        summary_cache._step5_csa_active_region_ids = torch.full(
            (capacity,), -1, device=device, dtype=torch.int32
        )
        summary_cache._step5_csa_active_slot_by_region = torch.full(
            (total_regions,), -1, device=device, dtype=torch.int32
        )
        summary_cache._step5_csa_allocation_success = torch.ones(
            (1,), device=device, dtype=torch.int32
        )
        summary_cache._step5_csa_numerator_cache = torch.zeros(
            (capacity, 1, proxy_dim), device=device, dtype=torch.float32
        )
        summary_cache._step5_csa_denominator_cache = torch.zeros_like(
            summary_cache._step5_csa_numerator_cache
        )
        summary_cache._step5_csa_max_cache = torch.full(
            (capacity, 1), float("-inf"), device=device, dtype=torch.float32
        )
        summary_cache._step5_csa_active_token_k = torch.zeros(
            (capacity, self.region_size, 1, proxy_dim),
            device=device,
            dtype=self._decode_out_dtype,
        )
        summary_cache._step5_csa_active_token_z = torch.zeros_like(
            summary_cache._step5_csa_active_token_k
        )
        summary_cache._step5_csa_active_token_valid = torch.zeros(
            (capacity, self.region_size), device=device, dtype=torch.uint8
        )
        if self.use_runner_prefill_slots:
            summary_cache._step5_csa_prefill_state = CSAPrefillState.allocate(
                CSA_SLOTS_PER_REQUEST * int(self.max_num_seqs), total_regions,
                proxy_dim, self.region_size, device,
            )

    def bind_summary_cache(
        self,
        summary_cache: AscendStep5SparseSummaryCache,
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
        if not self.use_runner_prefill_slots:
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
        self._dsa_scratch_bound = True

    @staticmethod
    def _round_up(value: int, alignment: int) -> int:
        return (int(value) + alignment - 1) // alignment * alignment

    @staticmethod
    def _begin_csa_allocation_check(
        summary_cache: AscendStep5SparseSummaryCache,
    ) -> torch.Tensor:
        """Reset and return the graph-stable device allocation status."""
        success = summary_cache._step5_csa_allocation_success
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
        summary_cache: AscendStep5SparseSummaryCache,
        md: AscendStep5DSAMetadata,
        num_actual_tokens: int,
    ) -> AscendStep5DSARuntimeLayout:
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
        return AscendStep5DSARuntimeLayout(
            token_flat_slot=token_flat_slot,
            token_positions=token_positions,
            reset_slots=reset_slots,
            token_valid=token_valid,
        )

    def _maybe_update_summary_cache(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        md: AscendStep5DSAMetadata,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> None:
        """Fold this step's indexer K/Z tokens into the paged summary state.

        Pure decode retains the original stage/flush path. Prefill defaults
        to model-side allocation; runner mode uses independent state and
        hands off the unfinished prompt tail.
        MTP verify batches (uniform K+1 rows per
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
                summary_cache._step5_csa_active_region_ids,
                summary_cache._step5_csa_active_slot_by_region,
                allocation_success,
                summary_cache._step5_csa_numerator_cache,
                summary_cache._step5_csa_denominator_cache,
                summary_cache._step5_csa_max_cache,
                summary_cache._step5_csa_active_token_k,
                summary_cache._step5_csa_active_token_z,
                summary_cache._step5_csa_active_token_valid,
                layout.token_flat_slot,
                layout.reset_slots,
                layout.token_valid,
                layout.token_positions,
                index_k,
                index_z,
                live_rows=int(layout.token_flat_slot.numel()),
                **common,
            )
        elif not self.use_runner_prefill_slots:
            self._update_prefill_default(summary_cache, layout, index_k, index_z,
                                         allocation_success, mean_cache, common, 0, num_tokens)
        else:
            # Mixed ordinary decode/prefill uses the same split as verify:
            # decode keeps staged-token semantics and prefill consumes its plan.
            decode_rows = int(md.num_decode_tokens)
            if md.num_decodes:
                csa_compact_decode_update(
                    mean_cache,
                    summary_cache._step5_csa_active_region_ids,
                    summary_cache._step5_csa_active_slot_by_region,
                    allocation_success,
                    summary_cache._step5_csa_numerator_cache,
                    summary_cache._step5_csa_denominator_cache,
                    summary_cache._step5_csa_max_cache,
                    summary_cache._step5_csa_active_token_k,
                    summary_cache._step5_csa_active_token_z,
                    summary_cache._step5_csa_active_token_valid,
                    layout.token_flat_slot[:decode_rows], layout.reset_slots[:decode_rows],
                    layout.token_valid[:decode_rows], layout.token_positions[:decode_rows],
                    index_k[:decode_rows], index_z[:decode_rows],
                    live_rows=decode_rows,
                    **common,
                )
            num_requests = md.num_prefills
            prefill_rows = num_tokens - decode_rows
            prefill_state = summary_cache._step5_csa_prefill_state
            csa_compact_prefill_update(
                mean_cache,
                prefill_state.region_ids,
                prefill_state.slot_by_region,
                allocation_success,
                prefill_state.numerator,
                prefill_state.denominator,
                prefill_state.max_logits,
                prefill_state.staging_valid,
                layout.token_flat_slot[decode_rows:],
                layout.reset_slots[decode_rows:],
                layout.token_valid[decode_rows:],
                layout.token_positions[decode_rows:],
                index_k[decode_rows:],
                index_z[decode_rows:],
                query_start_loc=md.query_start_loc[md.num_decodes:md.num_decodes + num_requests + 1],
                seq_lens=md.seq_lens[md.num_decodes:md.num_decodes + num_requests],
                live_rows=prefill_rows,
                slot_plan=md.csa_prefill_plan,
                **common,
            )
            self._handoff_prefill_state(summary_cache, md.csa_prefill_plan, allocation_success)
        if md.num_decodes or not self.use_runner_prefill_slots:
            self._assert_csa_allocation_success(allocation_success)

    def _update_prefill_default(
        self, summary_cache: AscendStep5SparseSummaryCache,
        layout: AscendStep5DSARuntimeLayout, index_k: torch.Tensor, index_z: torch.Tensor,
        allocation_success: torch.Tensor, mean_cache: torch.Tensor,
        common: dict, start: int, num_tokens: int,
    ) -> None:
        padded_rows = self._round_up(num_tokens, 64)
        scratch_region_ids = self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_region_ids", (padded_rows,), device=index_k.device, dtype=torch.int32,
        )
        scratch_row_map = self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_row_map", (padded_rows * self.region_size,),
            device=index_k.device, dtype=torch.int32,
        )
        scratch_reset_map = self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_scratch_reset_map", (padded_rows,), device=index_k.device, dtype=torch.int32,
        )
        csa_compact_prefill_update_default(
            mean_cache, summary_cache._step5_csa_active_region_ids,
            summary_cache._step5_csa_active_slot_by_region, allocation_success,
            summary_cache._step5_csa_numerator_cache,
            summary_cache._step5_csa_denominator_cache, summary_cache._step5_csa_max_cache,
            summary_cache._step5_csa_active_token_valid,
            scratch_region_ids, scratch_row_map, scratch_reset_map,
            layout.token_flat_slot[start:], layout.reset_slots[start:],
            layout.token_valid[start:], layout.token_positions[start:],
            index_k[start:], index_z[start:], live_rows=num_tokens, **common,
        )

    def _handoff_prefill_state(
        self, summary_cache: AscendStep5SparseSummaryCache,
        metadata: CSAMetadata, allocation_success: torch.Tensor,
    ) -> None:
        if not metadata.needs_handoff:
            return
        state = summary_cache._step5_csa_prefill_state
        csa_prefill_handoff(
            state.region_ids, state.slot_by_region, state.numerator,
            state.denominator, state.max_logits,
            summary_cache._step5_csa_active_region_ids,
            summary_cache._step5_csa_active_slot_by_region,
            summary_cache._step5_csa_numerator_cache,
            summary_cache._step5_csa_denominator_cache,
            summary_cache._step5_csa_max_cache,
            summary_cache._step5_csa_active_token_valid,
            allocation_success, metadata,
            proxy_dim=int(self.proxy_dim), region_size=int(self.region_size),
        )
        self._assert_csa_allocation_success(allocation_success)

    def _update_summary_cache_verify(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        md: AscendStep5DSAMetadata,
        layout: AscendStep5DSARuntimeLayout,
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
        batch, the prefill tail uses its independent runner-owned state for
        its own rows; a request never contributes rows to both segments.
        """
        d = md.decode
        n = md.num_decodes
        query_len = int(d.query_len)
        # 1) Materialize fully-committed tail regions (releases their slots
        #    for reuse by this step's staging).
        csa_deferred_flush(
            mean_cache,
            summary_cache._step5_csa_active_region_ids,
            summary_cache._step5_csa_active_slot_by_region,
            summary_cache._step5_csa_numerator_cache,
            summary_cache._step5_csa_denominator_cache,
            summary_cache._step5_csa_max_cache,
            summary_cache._step5_csa_active_token_k,
            summary_cache._step5_csa_active_token_z,
            summary_cache._step5_csa_active_token_valid,
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
            summary_cache._step5_csa_active_region_ids,
            summary_cache._step5_csa_active_slot_by_region,
            allocation_success,
            summary_cache._step5_csa_numerator_cache,
            summary_cache._step5_csa_denominator_cache,
            summary_cache._step5_csa_max_cache,
            summary_cache._step5_csa_active_token_k,
            summary_cache._step5_csa_active_token_z,
            summary_cache._step5_csa_active_token_valid,
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
        # 3) Mixed batch: update the prefill tail using the configured allocator.
        #    (prefill requests own private pages; their
        #    regions never collide with the verify requests' regions).
        if md.num_prefills > 0:
            prefill_rows = int(md.num_actual_tokens) - nv
            if not self.use_runner_prefill_slots:
                self._update_prefill_default(summary_cache, layout, index_k, index_z,
                                             allocation_success, mean_cache, common, nv, prefill_rows)
            else:
                prefill_state = summary_cache._step5_csa_prefill_state
                csa_compact_prefill_update(
                    mean_cache,
                    prefill_state.region_ids,
                    prefill_state.slot_by_region,
                    allocation_success,
                    prefill_state.numerator,
                    prefill_state.denominator,
                    prefill_state.max_logits,
                    prefill_state.staging_valid,
                    layout.token_flat_slot[nv:],
                    layout.reset_slots[nv:],
                    layout.token_valid[nv:],
                    layout.token_positions[nv:],
                    index_k[nv:],
                    index_z[nv:],
                    # 边界仍是全批次行号；kernel 减去首项，与 [nv:] 的输入对齐。
                    query_start_loc=md.query_start_loc[n:n + md.num_prefills + 1],
                    seq_lens=md.seq_lens[n:n + md.num_prefills],
                    live_rows=prefill_rows,
                    slot_plan=md.csa_prefill_plan,
                    **common,
                )
                self._handoff_prefill_state(summary_cache, md.csa_prefill_plan, allocation_success)
        self._assert_csa_allocation_success(allocation_success)

    # -- sparse attention ------------------------------------------------------

    def _forward_sparse_gqa_decode(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        md: AscendStep5DSAMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        d = md.decode
        assert d is not None
        if d.query_len != 1:
            if self.num_speculative_tokens <= 0:
                raise NotImplementedError(
                    "Ascend Step5 DSA paged summary path received a "
                    "multi-token decode batch without MTP enabled "
                    f"(query_len={d.query_len})."
                )
            raise NotImplementedError(
                "Ascend Step5 DSA verify attention must be dispatched through"
                " _forward_sparse_gqa_verify (query_len="
                f"{d.query_len})."
            )
        n = md.num_decodes
        groups = self.num_kv_groups
        seq_lens_i32 = d.seq_lens.to(torch.int32)
        # Fused score + causal topk + ascending sort via aclnn; the exact
        # attention is the aclnn GenericBlockSparseAttention operator, which
        # appends the current region (floor(pos / region_size)) and applies
        # the per-token causal truncation itself. actual_kv_lengths is the
        # effective KV length including the current query (seq_len); the
        # query token's own 0-based position (seq_len - 1) feeds the
        # attention operator's current-region append.
        cu_seq_lens = torch.arange(
            n + 1, device=index_q.device, dtype=torch.int32
        )
        positions = seq_lens_i32 - 1
        actual_kv_lengths = seq_lens_i32
        _scores, ids = indexer_score_topk(
            index_q[:n],
            weights[:n],
            summary_cache.mean_cache.view(torch.float8_e4m3fn),
            d.block_table,
            cu_seq_lens,
            actual_kv_lengths,
            max_region_len=self._max_regions,
            topk=self.topk_regions,
            region_size=self.region_size,
        )
        out = gbsa_sparse_attention(
            query,
            key_cache,
            value_cache,
            ids,
            positions,
            d.block_table,
            cu_seq_lens,
            seq_lens_i32,
            d.gbsa_metadata,
            num_kv_groups=groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
        )
        output[:n] = out.to(output.dtype)

    def _forward_sparse_gqa_verify(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        md: AscendStep5DSAMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse attention for an MTP verify batch (K+1 rows per request).

        Each verify request stays one multi-token varlen request of
        ``query_len = K+1`` rows spanning kv positions ``ctx..ctx+K`` with
        ``ctx = seq_len - K - 1``. The exact attention goes through the
        aclnn ``GenericBlockSparseAttention`` operator: its right-aligned
        causal bound (``seqused_kv = seq_len``, ``q_len = K+1``) places
        query row ``i`` at kv position ``ctx + i`` and truncates it to its
        own position, and the operator appends each row's current region
        ``floor((ctx + i) / region_size)`` with causal truncation. The
        fused indexer's per-token mask3 bound ``floor((ctx + j) /
        region_size)`` keeps the row's current region out of top-k
        competition -- the same contract the single-token decode path
        guarantees, evaluated once per request instead of once per row.
        Everything is device-driven on static per-bucket shapes, so uniform
        verify batches stay ACL-graph capturable.
        """
        decode_md = md.decode
        num_decodes = md.num_decodes
        query_len = int(decode_md.query_len)
        total_tokens = num_decodes * query_len
        groups = self.num_kv_groups
        # Request-level varlen metadata: the committed context
        # ctx = seq_len - query_len anchors both operators. seqused_kv is
        # the full seq_len (right-aligned causal anchor); the indexer's
        # actual_kv_lengths is the same effective KV length (seq_len),
        # from which the operator restores the first query position
        # ctx = actual_kv_lengths - q_len and applies the mask3 bound
        # floor((ctx + j) / region_size). cu_seq_lens reuses the builder's
        # start-loc copies -- int32 for the fused indexer, int64 for the
        # GenericBlockSparseAttention operator's cu_seqlens_q contract --
        # zero-kernel views refreshed in place by the builder. Static
        # shapes, all device tensors: ACL-graph capturable.
        seq_lens = decode_md.seq_lens[:num_decodes]
        cu_seq_lens = md.query_start_loc[: num_decodes + 1]
        cu_seqlens_q = md.query_start_loc_int64[: num_decodes + 1]
        actual_kv_lengths = seq_lens
        _scores, ids = indexer_score_topk(
            index_q[:total_tokens],
            weights[:total_tokens],
            summary_cache.mean_cache.view(torch.float8_e4m3fn),
            decode_md.block_table[:num_decodes],
            cu_seq_lens,
            actual_kv_lengths,
            max_region_len=self._max_regions,
            topk=self.topk_regions,
            region_size=self.region_size,
        )
        out = gbsa_sparse_attention(
            query,
            key_cache,
            value_cache,
            ids,
            positions[:total_tokens],
            decode_md.block_table[:num_decodes],
            cu_seqlens_q,
            seq_lens,
            decode_md.gbsa_metadata,
            num_kv_groups=groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
        )
        output[:total_tokens] = out.to(output.dtype)

    def _forward_sparse_gqa_prefill(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        md: AscendStep5DSAMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
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

        # Live history width in logical regions (the furthest last-position
        # of any request in this chunk), bounding what the fused operator
        # needs to score.
        query_lens_list = [
            int(p.seq_lens_host[row]) - int(p.context_lens_host[row])
            for row in range(num_reqs)
        ]
        width = max(
            (
                (int(p.seq_lens_host[row]) - 1) // self.region_size
                for row in range(num_reqs)
                if query_lens_list[row] > 0
            ),
            default=0,
        )

        q_positions = positions[token_offset : token_offset + num_query_tokens].to(
            torch.int32
        )

        # Fused score + causal topk + ascending sort via aclnn. The
        # operator's mask3 bound floor((context_lens[n] + j)/region_size) is
        # each token's own position // region_size, so the current region
        # never competes for a slot; the attention operator appends it and
        # causally truncates it to the token's position.
        # cu_seq_lens from the device start-loc; actual_kv_lengths is the
        # effective KV length including this chunk (cached context plus
        # query tokens), from which the indexer restores the first query
        # position as seq_len - q_len.
        cu_seq_lens = torch.cat(
            [
                torch.zeros((1,), device=index_q.device, dtype=torch.int32),
                p.query_start_loc[1 : num_reqs + 1].to(torch.int32),
            ]
        )
        actual_kv_lengths = p.seq_lens[:num_reqs].to(torch.int32)
        _scores, ids = indexer_score_topk(
            index_q[:num_query_tokens],
            weights[:num_query_tokens],
            summary_cache.mean_cache.view(torch.float8_e4m3fn),
            p.block_table[:num_reqs],
            cu_seq_lens,
            actual_kv_lengths,
            max_region_len=width,
            topk=self.topk_regions,
            region_size=self.region_size,
        )
        out = gbsa_sparse_attention(
            query,
            key_cache,
            value_cache,
            ids,
            q_positions,
            p.block_table[:num_reqs],
            cu_seq_lens,
            p.seq_lens[:num_reqs].to(torch.int32),
            p.gbsa_metadata,
            num_kv_groups=self.num_kv_groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
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
        if not isinstance(md, AscendStep5DSAMetadata):
            return output.zero_()

        summary_cache = self._summary_cache
        if summary_cache is None:
            raise RuntimeError(
                "Step5 sparse summary cache must be bound during KV cache "
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
                    key_cache=key_cache,
                    value_cache=value_cache,
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
                    key_cache=key_cache,
                    value_cache=value_cache,
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
                key_cache=key_cache,
                value_cache=value_cache,
                index_q=index_q[start:num_tokens],
                weights=weights[start:num_tokens],
                positions=positions,
                output=output,
                token_offset=start,
            )
        return output
