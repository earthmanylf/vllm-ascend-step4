# SPDX-License-Identifier: Apache-2.0
"""Ascend Step5 DSA with SWF per-page state and persistent request journal.

The NPU summary producer follows the fixed SWF ordered FP32 update and
request-indexed rollback contract. Ascend IndexerRegionScore and GenericBlockSparseAttention operators
consume the materialized FP8 summaries; each row excludes future regions,
force-includes its current region and applies an exact token causal mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
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
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.models.step5._step5_compat import Step5DSAScratchWorkspace
from vllm_ascend.models.step5.generic_block_sparse_attention import (
    gbsa_expected_counts,
    gbsa_sparse_attention,
    gbsa_sparse_attention_metadata,
)
from vllm_ascend.models.step5.indexer_region_score import indexer_score_topk
from vllm_ascend.models.step5.summary_cache import AscendStep5SparseSummaryCache, AscendStep5SparseSummaryMetadata
from vllm_ascend.models.step5.summary_kernels import build_prefill_groups

logger = init_logger(__name__)


def _is_graph_capturing() -> bool:
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


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
    def get_supported_kernel_block_sizes(kv_cache_spec: AttentionSpec | None = None) -> list[int | MultipleOf]:
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

    gbsa_metadata: torch.Tensor | None = None
    """Per-step AICPU schedule shared by every DSA layer in this group."""
    cu_seqlens_q: torch.Tensor | None = None
    """Int64 cumulative query lengths consumed by the native attention op."""

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

    gbsa_metadata: torch.Tensor | None = None
    """Per-step AICPU schedule shared by every DSA layer in this group."""
    cu_seqlens_q: torch.Tensor | None = None
    """Int64 cumulative query lengths consumed by the native attention op."""


@dataclass
class AscendStep5DSAMetadata(AttentionMetadata):
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
    prefill: AscendStep5DSAPrefillMetadata | None = None
    decode: AscendStep5DSADecodeMetadata | None = None


class AscendStep5DSAMetadataBuilder(AttentionMetadataBuilder[AscendStep5DSAMetadata]):
    # Decode forward is fully tensor-driven (see AscendStep5DSAImpl
    # ._forward_sparse_gqa_decode): static shapes per capture bucket, all
    # data-dependent choices realized with device-side validity masks, so
    # uniform single-token decode batches are ACL-graph capturable under
    # FULL_DECODE_ONLY; mixed prefill+decode batches still run eager -- the
    # contract of UNIFORM_BATCH.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
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

        impl = vllm_config.compilation_config.static_forward_context[layer_names[0]].impl
        self.gbsa_num_heads = impl.num_heads
        self.gbsa_num_kv_groups = impl.num_kv_groups
        self.gbsa_head_size = impl.head_size
        self.gbsa_region_size = impl.region_size
        self.gbsa_topk = impl.topk_regions
        self.gbsa_block_size = kv_cache_spec.block_size
        # MRV2 rounds speculative graph descriptors to the uniform query
        # length, so raw config sizes do not identify the captured shapes.
        self.capture_sizes: set[int] = set()
        self.gbsa_meta_buffers: dict[tuple, torch.Tensor] = {}
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

    def _prepare_gbsa_metadata(self, md: AscendStep5DSAMetadata) -> None:
        """Schedule native attention outside the model, without touching journals."""
        query_start_loc_int64 = self.query_start_loc_int64_buffer[: md.query_start_loc.shape[0]]
        query_start_loc_int64.copy_(md.query_start_loc)
        graph_shape = md.num_actual_tokens in self.capture_sizes
        d = md.decode
        if d is not None and md.num_decodes:
            n = md.num_decodes
            q_len = int(d.query_len)
            total = n * q_len
            row = torch.arange(total, device=d.seq_lens.device, dtype=torch.int64)
            positions = (d.seq_lens.to(torch.int64) - q_len).repeat_interleave(q_len) + row % q_len
            d.cu_seqlens_q = query_start_loc_int64[: n + 1]
            d.gbsa_metadata = self._build_gbsa_metadata(
                positions,
                d.cu_seqlens_q,
                d.seq_lens,
                d.block_table,
                q_len,
                path="decode",
                graph_shape=graph_shape,
            )
        p = md.prefill
        if p is not None and md.num_prefills:
            p.cu_seqlens_q = p.query_start_loc.to(torch.int64)
            row = torch.arange(p.num_query_tokens, device=p.seq_lens.device, dtype=torch.int64)
            requests = torch.searchsorted(p.cu_seqlens_q, row, right=True) - 1
            positions = p.context_lens.to(torch.int64)[requests] + row - p.cu_seqlens_q[requests]
            p.gbsa_metadata = self._build_gbsa_metadata(
                positions,
                p.cu_seqlens_q,
                p.seq_lens,
                p.block_table,
                p.max_query_len,
                path="prefill",
                graph_shape=False,
            )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        **kwargs,
    ) -> AscendStep5DSAMetadata:
        # Register before build so warmup, capture and later runtime builds
        # all bind the same address for this actual capture descriptor.
        self.capture_sizes.add(common_attn_metadata.num_actual_tokens)
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendStep5DSAMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        # The upstream uniform splitter indexes the first query length.
        if common_attn_metadata.num_reqs == 0:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = 0, 0, 0, 0
        else:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )

        prefill_metadata: AscendStep5DSAPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(num_prefills, num_prefill_tokens, qsl_cpu, num_decodes)
            prefill_end = num_decodes + active_prefills
            prefill_seq_lens = seq_lens[num_decodes:prefill_end]
            prefill_query_lens = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[num_decodes:prefill_end]
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
                query_start_loc=query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens,
                max_query_len=int(prefill_query_lens.max()),
                num_query_tokens=int(prefill_query_lens.sum()),
                seq_lens_host=seq_lens_host_t.tolist(),
                context_lens_host=context_lens_host_t.tolist(),
            )

        decode_metadata: AscendStep5DSADecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(num_decodes, num_decode_tokens, decode_query_len)
            decode_metadata = AscendStep5DSADecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                query_len=decode_query_len,
            )

        metadata = AscendStep5DSAMetadata(
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
        self._prepare_gbsa_metadata(metadata)
        return metadata


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
                "Step5 DSA requires block_size to be a multiple of the region "
                f"size, got block_size={block_size}, region_size={region_size}."
            )
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.regions_per_page = block_size // region_size
        self.num_kv_groups = self.num_kv_heads
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.max_model_len = int(max_model_len)
        self.max_num_batched_tokens = max(1, int(max_num_batched_tokens))
        self._decode_out_dtype = decode_out_dtype or torch.bfloat16
        # Selector width: every region a max-length sequence can fill.
        self._max_regions = -(-self.max_model_len // self.region_size)
        self.summary_cache: AscendStep5SparseSummaryCache | None = None
        self.summary_layer_name: str | None = None
        # Injected by the model (mirrors upstream model.py): the ring that
        # carries each request's open-region rows and addresses the summary
        # updates. Created with the model so memory profiling counts it.
        self.csa_ring = None
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
        buffers = self._dsa_scratch_workspace.tensor_buffers_by_engine.setdefault(virtual_engine, {})
        required = 1
        for dim in shape:
            required *= int(dim)
        out = buffers.get(name)
        if out is None or out.device != device or out.dtype != dtype or int(out.numel()) < int(required):
            if self._dsa_scratch_workspace.allocations_locked or getattr(self, "_dsa_scratch_bound", False):
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

    def bind_summary_cache(
        self,
        summary_cache: AscendStep5SparseSummaryCache,
        *,
        initialize_runtime_state: bool = True,
    ) -> None:
        """Bind the TailRing summary cache before graph capture.

        The TailRing design keeps no per-page accumulators and no undo
        journal, so there is no runtime state to initialise and no prefill
        scratch workspace to reserve: the ring itself is allocated by the KV
        cache layer and the summary is written straight into ``mean``.
        """
        if initialize_runtime_state:
            self.summary_cache = summary_cache

    # -- SWF summary producer -------------------------------------------------

    def _maybe_update_summary_cache(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        summary_metadata: AscendStep5SparseSummaryMetadata,
        md: AscendStep5DSAMetadata,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Update summaries with token-local ring metadata for each batch part."""
        num_tokens = int(md.num_actual_tokens)
        if num_tokens <= 0:
            return
        key = index_k[:num_tokens].contiguous()
        z = index_z[:num_tokens].contiguous()
        slots = summary_metadata.slot_mapping[:num_tokens]
        ring = self.csa_ring
        if ring is None:
            raise RuntimeError("Step5 DSA requires the CSA tail ring to be bound before the summary update")
        decode_rows = int(md.num_decode_tokens) if md.decode is not None else 0
        if decode_rows:
            if md.decode.query_len > 1:
                summary_cache.update_verify(
                    key[:decode_rows],
                    z[:decode_rows],
                    slots[:decode_rows],
                    md.query_start_loc[: md.num_decodes + 1],
                    md.seq_lens[: md.num_decodes],
                    ring,
                )
            else:
                summary_cache.update_decode(
                    key[:decode_rows],
                    z[:decode_rows],
                    slots[:decode_rows],
                    positions[:decode_rows],
                    ring,
                )
        if md.num_prefills > 0:
            # Local QSL matches local key/z rows. Grouping is device-only and
            # uses true positions, including chunked partial-region tails.
            query_start = md.prefill.query_start_loc
            rows = num_tokens - decode_rows
            owners = self._summary_owners(query_start, rows)
            metadata = build_prefill_groups(
                slots[decode_rows:],
                positions[decode_rows:num_tokens],
                owners,
                query_start,
            )
            summary_cache.update_prefill(
                key[decode_rows:],
                z[decode_rows:],
                metadata,
                positions[decode_rows:num_tokens],
                ring,
                token_offset=decode_rows,
            )

    @staticmethod
    def _summary_owners(query_start: torch.Tensor, num_tokens: int) -> torch.Tensor:
        rows = torch.arange(num_tokens, device=query_start.device, dtype=query_start.dtype)
        return (torch.searchsorted(query_start, rows, right=True) - 1).to(torch.int32)

    # -- sparse attention ------------------------------------------------------

    def _forward_sparse_gqa_decode(
        self,
        *,
        summary_cache: AscendStep5SparseSummaryCache,
        summary_metadata: AscendStep5SparseSummaryMetadata,
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
        cu_seq_lens = md.query_start_loc[: n + 1]
        positions = seq_lens_i32 - 1
        actual_kv_lengths = seq_lens_i32
        _scores, ids = indexer_score_topk(
            index_q[:n],
            weights[:n],
            summary_cache.mean.view(torch.float8_e4m3fn),
            summary_metadata.block_table[:n],
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
            d.cu_seqlens_q,
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
        summary_metadata: AscendStep5SparseSummaryMetadata,
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
        The requests retain their original varlen boundaries for both operators.
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
        cu_seqlens_q = decode_md.cu_seqlens_q
        actual_kv_lengths = seq_lens
        _scores, ids = indexer_score_topk(
            index_q[:total_tokens],
            weights[:total_tokens],
            summary_cache.mean.view(torch.float8_e4m3fn),
            summary_metadata.block_table[:num_decodes],
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
        summary_metadata: AscendStep5SparseSummaryMetadata,
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
        query_lens_list = [int(p.seq_lens_host[row]) - int(p.context_lens_host[row]) for row in range(num_reqs)]
        width = max(
            (
                (int(p.seq_lens_host[row]) - 1) // self.region_size
                for row in range(num_reqs)
                if query_lens_list[row] > 0
            ),
            default=0,
        )

        q_positions = positions[token_offset : token_offset + num_query_tokens].to(torch.int32)

        # Fused score + causal topk + ascending sort via aclnn. The
        # operator's mask3 bound floor((context_lens[n] + j)/region_size) is
        # each token's own position // region_size, so the current region
        # never competes for a slot; the attention operator appends it and
        # causally truncates it to the token's position.
        # cu_seq_lens from the device start-loc; actual_kv_lengths is the
        # effective KV length including this chunk (cached context plus
        # query tokens), from which the indexer restores the first query
        # position as seq_len - q_len.
        cu_seq_lens = p.query_start_loc[: num_reqs + 1]
        actual_kv_lengths = p.seq_lens[:num_reqs].to(torch.int32)
        _scores, ids = indexer_score_topk(
            index_q[:num_query_tokens],
            weights[:num_query_tokens],
            summary_cache.mean.view(torch.float8_e4m3fn),
            summary_metadata.block_table[md.num_decodes : md.num_decodes + num_reqs],
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
            p.cu_seqlens_q,
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

        summary_cache = self.summary_cache
        if summary_cache is None:
            raise RuntimeError(
                "Step5 sparse summary cache must be bound during KV cache initialization before attention forward."
            )
        num_tokens = md.num_actual_tokens
        output[num_tokens:].zero_()
        if num_tokens == 0:
            return output

        summary_metadata = attn_metadata.get(self.summary_layer_name)
        if not isinstance(summary_metadata, AscendStep5SparseSummaryMetadata):
            raise RuntimeError(
                f"Step5 summary group metadata is missing or invalid for layer {self.summary_layer_name!r}."
            )

        self._maybe_update_summary_cache(
            summary_cache=summary_cache,
            summary_metadata=summary_metadata,
            md=md,
            index_k=index_k,
            index_z=index_z,
            positions=positions,
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
                    summary_metadata=summary_metadata,
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
                    summary_metadata=summary_metadata,
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
                summary_metadata=summary_metadata,
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
