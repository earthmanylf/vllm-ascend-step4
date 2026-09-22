"""Step4 geometry, as measured from the shipping checkpoint on 2026-08-11.

Every value here was read out of the checkpoint's safetensors headers (no tensor
loads) or out of ``step4/config.json``. Do not edit without re-measuring -- several
of these are the difference between a silently-wrong model and a correct one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step4Geometry:
    hidden_size: int = 4096

    # Main attention. q_proj [12288, 4096] = 64 * 192, k/v_proj [768, 4096] = 4 * 192.
    num_attention_heads: int = 64
    num_attention_groups: int = 4
    head_dim: int = 192

    # DSA indexer. sparse_indexer_q [4096, 4096] = 16 * 256;
    # sparse_indexer_k and _z are both [256, 4096] => one shared 256-dim head (MQA).
    proxy_dim: int = 256
    indexer_num_heads: int = 16
    indexer_num_k_heads: int = 1
    num_provider_groups: int = 4

    # Rotates the leading 32 of 256 dims, half-split NeoX. An internal doc claims 64;
    # that came from misreading a fused kernel arg whose unit is cos/sin pairs.
    indexer_rope_dim: int = 32

    topk: int = 512
    region_block_size: int = 8
    paged_block_size: int = 16

    sliding_window: int = 512

    num_hidden_layers: int = 92
    # Per-layer config lists are length 93; index 92 is the MTP layer. Never truncate.
    num_layer_entries: int = 93

    vocab_size: int = 128896
    moe_num_experts: int = 352
    moe_intermediate_size: int = 1536
    dense_intermediate_size: int = 13824

    @property
    def q_heads_per_group(self) -> int:
        return self.indexer_num_heads // self.num_provider_groups

    @property
    def indexer_weight_prescale(self) -> float:
        """Variance-preserving scale for the per-head sum in indexer scoring.

        Derived from geometry, not from the shard layout: the score sums over the
        heads inside one provider group, so the divisor is 4, never 16. Computing
        this from "all locally resident heads" yields 0.25 and silently changes
        which regions get selected.
        """
        return float(self.q_heads_per_group) ** -0.5

    @property
    def regions_per_page(self) -> int:
        return self.paged_block_size // self.region_block_size

    def __post_init__(self) -> None:
        assert self.indexer_num_heads % self.num_provider_groups == 0
        assert self.paged_block_size % self.region_block_size == 0
        assert self.num_attention_heads % self.num_attention_groups == 0
        assert self.num_layer_entries == self.num_hidden_layers + 1


STEP4 = Step4Geometry()

# The upstream unit-test fixtures run step3p5 geometry (8 provider groups, 32 indexer
# heads). Kept here so parity runs can be reproduced against the upstream tests.
STEP3P5 = Step4Geometry(
    num_attention_groups=8,
    indexer_num_heads=32,
    num_provider_groups=8,
)
