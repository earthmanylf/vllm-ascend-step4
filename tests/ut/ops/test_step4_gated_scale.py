"""Unit tests for the Step-4 fused head-wise gate scale Triton kernel.

Tests compare the fused kernel output against the eager path it replaces:
``attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()``. The gate is
supplied both contiguous and as a strided column view of a wider packed
buffer (matching the views the model feeds it: the qkvg split on SWA
layers and the split_indexer slice on DSA layers), so the kernel's
stride handling is covered.

Run on NPU:
    pytest -sv tests/ut/ops/test_step4_gated_scale.py
"""

import pytest
import torch

from vllm_ascend.models.step5.gated_scale import gated_scale


def _reference(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    num_tokens = attn_output.shape[0]
    num_heads = gate.shape[-1]
    head_dim = attn_output.shape[-1] // num_heads
    return (
        attn_output.view(num_tokens, num_heads, head_dim)
        * gate.unsqueeze(-1).sigmoid()
    ).view(num_tokens, -1)


@pytest.mark.parametrize("num_tokens", [1, 8, 96, 337])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_gated_scale_contiguous(num_tokens: int, num_heads: int, head_dim: int):
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    torch.manual_seed(123)
    attn_output = torch.randn(
        num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(num_tokens, num_heads, device=device, dtype=torch.bfloat16)

    out = gated_scale(attn_output, gate)
    ref = _reference(attn_output.clone(), gate.clone())

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("head_dim", [192])
def test_gated_scale_strided_gate(num_heads: int, head_dim: int):
    """Gate as a strided column view of a packed [qkv | gate] buffer."""
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    torch.manual_seed(123)
    num_tokens = 96
    kv_width = 2 * 192
    row_width = num_heads * head_dim + kv_width + num_heads
    packed = torch.randn(
        num_tokens, row_width, device=device, dtype=torch.bfloat16
    )
    attn_output = packed[..., : num_heads * head_dim].contiguous()
    gate = packed[..., -num_heads:]
    assert gate.stride(0) == row_width

    out = gated_scale(attn_output, gate)
    ref = _reference(attn_output.clone(), gate.contiguous())

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
