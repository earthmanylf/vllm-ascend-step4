"""Fused head-wise attention gate: sigmoid + broadcast multiply in one kernel.

Replaces the eager post-attention chain on Ascend -- the
``step5_materialize_gate_input`` clones (an inductor-only guard) plus
``attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()`` -- with a
single Triton kernel. The gate may be a strided column view into the
fused qkv+gate GEMM output (SWA layers with ``fuse_qkv_gate``) or into
the fused qkv+indexer output (DSA layers): the row stride is a runtime
kernel argument, so no materializing copy is needed. The sigmoid and the
multiply run in fp32, which is at least as accurate as the bf16 eager
path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _gated_scale_kernel(
    attn_ptr,
    gate_ptr,
    out_ptr,
    stride_attn_token,
    stride_gate_token,
    stride_out_token,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    g = tl.sigmoid(
        tl.load(gate_ptr + token * stride_gate_token + head).to(tl.float32)
    )
    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < HEAD_DIM
    x = tl.load(
        attn_ptr + token * stride_attn_token + head * HEAD_DIM + dims,
        mask=dim_ok,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + token * stride_out_token + head * HEAD_DIM + dims,
        (x * g).to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )


def _gated_scale(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    num_heads = gate.shape[-1]
    head_dim = attn_output.shape[-1] // num_heads
    out = torch.empty_like(attn_output)
    _gated_scale_kernel[(attn_output.shape[0], num_heads)](
        attn_output,
        gate,
        out,
        attn_output.stride(0),
        gate.stride(0),
        out.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out


def _gated_scale_fake(
    attn_output: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(attn_output)


direct_register_custom_op(
    op_name="ascend_step5_gated_scale",
    op_func=_gated_scale,
    mutates_args=[],
    fake_impl=_gated_scale_fake,
)


def gated_scale(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply the head-wise sigmoid gate: ``out[t] = attn[t] * sigmoid(gate[t])``.

    ``attn_output`` is ``(tokens, num_heads * head_dim)``; ``gate`` is
    ``(tokens, num_heads)`` and may be a strided column view (row stride
    wider than ``num_heads``) -- no contiguous copy is performed.
    """
    return torch.ops.vllm.ascend_step5_gated_scale(attn_output, gate)
