# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
    )
    def Step4BlockSparseAttention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_regions: Tensor,
        region_counts: Tensor,
        kv_seqlens: Optional[Tensor],
        *,
        num_heads: int,
        num_key_value_heads: int,
        region_size: int = 8,
        page_size: int = 128,
        scale_value: float = 0.0,
        layout_q: str = "TND",
        layout_kv: str = "TGD",
        num_splits: int = 1,
        return_softmax_lse: bool = False,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(Step4BlockSparseAttention)\n
        .INPUT(query, TensorType({DT_BF16}))\n
        .INPUT(key, TensorType({DT_BF16}))\n
        .INPUT(value, TensorType({DT_BF16}))\n
        .INPUT(packed_regions, TensorType({DT_INT32, DT_INT64}))\n
        .INPUT(region_counts, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(kv_seqlens, TensorType({DT_INT32}))\n
        .OUTPUT(attention_out, TensorType({DT_BF16}))\n
        .OPTIONAL_OUTPUT(softmax_lse, TensorType({DT_FLOAT}))\n
        .REQUIRED_ATTR(num_heads, Int)\n
        .REQUIRED_ATTR(num_key_value_heads, Int)\n
        .ATTR(region_size, Int, 8)\n
        .ATTR(page_size, Int, 128)\n
        .ATTR(scale_value, Float, 0.0)\n
        .ATTR(layout_q, String, "TND")\n
        .ATTR(layout_kv, String, "TGD")\n
        .ATTR(num_splits, Int, 1)\n
        .ATTR(return_softmax_lse, Int, 1)\n
        """
        # process inputs
        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "packed_regions": packed_regions,
            "region_counts": region_counts,
            "kv_seqlens": kv_seqlens,
        }

        # process attrs
        attrs = {
            "num_heads": attr.Int(num_heads),
            "num_key_value_heads": attr.Int(num_key_value_heads),
            "region_size": attr.Int(region_size),
            "page_size": attr.Int(page_size),
            "scale_value": attr.Float(scale_value),
            "layout_q": attr.Str(layout_q),
            "layout_kv": attr.Str(layout_kv),
            "num_splits": attr.Int(num_splits),
            "return_softmax_lse": attr.Int(1 if return_softmax_lse else 0),
        }

        # process outputs
        outputs = [
            "attention_out",
            "softmax_lse",
        ]

        return ge_op(
            op_type="Step4BlockSparseAttention",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("Step4BlockSparseAttention")
            .input("query", "DT_BF16")
            .input("key", "DT_BF16")
            .input("value", "DT_BF16")
            .input("packed_regions", "DT_INT32, DT_INT64")
            .input("region_counts", "DT_INT32")
            .optional_input("kv_seqlens", "DT_INT32")
            .required_attr("num_heads", attr.Int)
            .required_attr("num_key_value_heads", attr.Int)
            .attr("region_size", attr.Int(8))
            .attr("page_size", attr.Int(128))
            .attr("scale_value", attr.Float(0.0))
            .attr("layout_q", attr.Str("TND"))
            .attr("layout_kv", attr.Str("TGD"))
            .attr("num_splits", attr.Int(1))
            .attr("return_softmax_lse", attr.Int(1))
            .output("attention_out", "DT_BF16")
            .output("softmax_lse", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.step4_block_sparse_attention.default
    )
    def convert_step4_block_sparse_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_regions: Tensor,
        region_counts: Tensor,
        num_heads: int,
        num_key_value_heads: int,
        *,
        kv_seqlens: Optional[Tensor] = None,
        region_size: int = 8,
        page_size: int = 128,
        scale_value: float = 0.0,
        layout_q: str = "TND",
        layout_kv: str = "TGD",
        num_splits: int = 1,
        return_softmax_lse: bool = False,
        meta_outputs: TensorSpec = None,
    ):
        """step4_block_sparse_attention GE converter."""
        return Step4BlockSparseAttention(
            query=query,
            key=key,
            value=value,
            packed_regions=packed_regions,
            region_counts=region_counts,
            kv_seqlens=kv_seqlens,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            region_size=region_size,
            page_size=page_size,
            scale_value=scale_value,
            layout_q=layout_q,
            layout_kv=layout_kv,
            num_splits=num_splits,
            return_softmax_lse=return_softmax_lse,
        )
