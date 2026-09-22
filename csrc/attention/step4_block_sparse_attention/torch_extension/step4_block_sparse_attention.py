# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple
import torch
import torch_npu  # noqa: F401
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class Step4BlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("step4_block_sparse_attention", category="attention")

    def sources(self):
        return ["csrc/attention/step4_block_sparse_attention.cpp"]

    def schema(self) -> str:
        return (
            "step4_block_sparse_attention("
            "Tensor query, Tensor key, Tensor value, "
            "Tensor packed_regions, Tensor region_counts, "
            "int num_heads, int num_key_value_heads, "
            "*, "
            "Tensor? kv_seqlens=None, "
            "int region_size=8, "
            "int page_size=128, "
            "float scale_value=0.0, "
            'str layout_q="TND", '
            'str layout_kv="TGD", '
            "int num_splits=1, "
            "bool return_softmax_lse=False"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            packed_regions: torch.Tensor,
            region_counts: torch.Tensor,
            num_heads: int,
            num_key_value_heads: int,
            *,
            kv_seqlens: Optional[torch.Tensor] = None,
            region_size: int = 8,
            page_size: int = 128,
            scale_value: float = 0.0,
            layout_q: str = "TND",
            layout_kv: str = "TGD",
            num_splits: int = 1,
            return_softmax_lse: bool = False,
        ):
            torch._check(query.dim() == 3, lambda: f"query should be 3-D TND [T, N1, D], got dim {query.dim()}")
            torch._check(query.dtype == torch.bfloat16 or query.dtype == torch.float16,
                        lambda: f"step4 only supports BF16/FP16 query, got {query.dtype}")
            torch._check(key.dtype == query.dtype, lambda: f"key should share query dtype, got {key.dtype}")
            torch._check(value.dtype == query.dtype, lambda: f"value should share query dtype, got {value.dtype}")
            torch._check(query.size(-1) <= 256,
                        lambda: f"embedding size (head dim) should not exceed 256, got {query.size(-1)}")
            packed_dtype = packed_regions.dtype
            is_decode = packed_dtype == torch.int64
            torch._check(packed_dtype == torch.int32 or packed_dtype == torch.int64,
                         lambda: f"packed_regions should be INT32 (Prefill) or INT64 (Decode), got {packed_dtype}")
            if is_decode:
                torch._check(kv_seqlens is not None,
                             lambda: "kv_seqlens is required in Decode mode (packed_regions is INT64).")
            torch._check(num_heads > 0, lambda: f"num_heads should be greater than 0, got {num_heads}")
            torch._check(num_key_value_heads > 0,
                        lambda: f"num_key_value_heads should be greater than 0, got {num_key_value_heads}")
            torch._check(num_splits == 1, lambda: f"num_splits > 1 is not supported yet, got {num_splits}")

            attn = torch.empty(query.shape, dtype=query.dtype, device="meta")
            if return_softmax_lse:
                # TND query [T, N1, D] → LSE [T, N1]
                lse = torch.empty((query.size(0), query.size(1)), dtype=torch.float32, device="meta")
            else:
                lse = torch.empty((0,), dtype=torch.float32, device="meta")
            return attn, lse


_op_builder = Step4BlockSparseAttentionOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def step4_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    *,
    kv_seqlens: Optional[torch.Tensor] = None,
    region_size: int = 8,
    page_size: int = 128,
    scale_value: float = 0.0,
    layout_q: str = "TND",
    layout_kv: str = "TGD",
    num_splits: int = 1,
    return_softmax_lse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Step4 Block Sparse Attention 前向计算，封装 aclnnStep4BlockSparseAttention。

    适用于 Ascend 950（arch35）的 block sparse attention 算子，支持 BF16/FP16。
    通过 packed_regions 描述稀疏模式，区分 Prefill 与 Decode 场景。

    Args:
        query (Tensor): 查询张量，TND 排布 ``[T, N1, D]``，dtype 为 BF16 或 FP16。
        key (Tensor): 键张量，TGD 排布 ``[T_kv, G, D]``，直接从 KV cache 读取，dtype 与 query 一致。
        value (Tensor): 值张量，TGD 排布 ``[T_kv, G, D]``，dtype 与 query 一致。
        packed_regions (Tensor): 稀疏区域描述，group-major ``[G * T, K]``，
            ``K = topk + 1``。INT32 表示 Prefill，INT64 表示 Decode。
            Prefill: ``phys_region | (valid_tokens << 24)``，``valid`` 1–255；
            无效只填 -1，不要把任意负数当 skip。
            Decode: ``start_token | (phys_region << 32)``，无效填 0；
            ``packed == 0`` 是 region-0 合法编码，不要当 skip。
        region_counts (Tensor): INT32 ``[G * T]``，每个 query token 实际使用的 region 数，
            以它为准，不要扫到 K。
        num_heads (int): query 头数 N1。
        num_key_value_heads (int): kv 头数 G。
        kv_seqlens (Tensor, optional): INT32 ``[B]``，当前 token 的绝对长度（非前缀和），
            Decode 模式必传，Prefill 传 None。默认 None。
            用于 ``valid = clamp(kv_seqlen - start, 0, region_size)``。
        region_size (int): 每个 region 的 token 数，默认 8。
        page_size (int): paged block（blocksize），默认 128，须整除 ``region_size``。
            TGD 扁平 cache 上地址等价于 ``phys * region_size``；PA + block_table 未支持。
        scale_value (float): 缩放系数，默认 0.0（表示 ``1/sqrt(D)``）。
        layout_q (str): query 数据排布，默认 "TND"。
        layout_kv (str): key/value 数据排布，默认 "TGD"。
        num_splits (int): split-KV 数，初版仅支持 1，默认 1。
        return_softmax_lse (bool): 是否返回 softmax LSE，默认 False。

    Returns:
        Tuple[Tensor, Tensor]:
            - attention_out: 注意力输出，shape 与 query 相同 ``[T, N1, D]``，dtype 与 query 一致。
            - softmax_lse: 当 ``return_softmax_lse=True`` 时为 ``[T, N1]`` 的 float32；
              否则为空 tensor ``[0]``。

    Example::

        # Prefill
        attn, lse = step4_block_sparse_attention(
            q, k, v, packed, counts,
            num_heads=64, num_key_value_heads=4,
            region_size=8, return_softmax_lse=True,
        )

        # Decode (kv_seqlens 必传)
        attn, _ = step4_block_sparse_attention(
            q_dec, k, v, packed_dec, counts_dec,
            num_heads=64, num_key_value_heads=4,
            kv_seqlens=seq_lens,
            region_size=8,
        )
    """
    return _op_builder.load().step4_block_sparse_attention(
        query,
        key,
        value,
        packed_regions,
        region_counts,
        num_heads,
        num_key_value_heads,
        kv_seqlens,
        region_size,
        page_size,
        scale_value,
        layout_q,
        layout_kv,
        num_splits,
        return_softmax_lse,
    )
