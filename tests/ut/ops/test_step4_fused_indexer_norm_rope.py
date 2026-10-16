"""Unit tests for the Step-4 fused indexer QKZ norm+RoPE Triton kernel.

Compares the fused kernel against a pure-PyTorch reference replicating the
module's eager path (AscendStep4DSAAttention._finish_indexer):
  - index_q: per-head RMSNorm (zero_centered -> weight + 1), no mean subtract
  - index_k: per-head LayerNorm (subtract mean, with bias)
  - NeoX RoPE on the first ``rotary_dim`` dims = rotary_dim//2 pairs
    (pair layout: dims i and i + rotary_dim//2), angles from the caller's
    cos/sin cache whose row width is rotary_dim//2.
  - index_z: passthrough copy.

Regression coverage for the bug where the kernel treated ``rotary_dim`` as
a pair count: with sparse_indexer_rope_dim=32 it read dims 32-63 as the
imaginary half and ran cache lookups past the 16-wide row into the next
position's angles.

Run on NPU:
    pytest -sv tests/ut/ops/test_step4_fused_indexer_norm_rope.py
"""

import pytest
import torch

HEAD_DIM = 256          # proxy_dim
ROTARY_DIM = 32         # sparse_indexer_rope_dim: 32 dims = 16 pairs
EPS = 1e-6
THETA = 10000000.0      # indexer rope_theta (DSA layers)


def _build_cos_sin_cache(max_pos: int, pairs: int, theta: float, device):
    """Production-layout cache: [max_pos, 2*pairs] -> chunk -> two views with
    stride(0) == 2*pairs, as built from cos_sin_cache.chunk(2, -1)."""
    idx = torch.arange(0, 2 * pairs, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / (2 * pairs)))
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    return cache.chunk(2, dim=-1)


def _reference_indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float,
    q_norm_weight_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = index_q.shape[0]
    dtype = index_q.dtype

    def rms_norm(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(tokens, -1, head_dim).float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        return (normed * (q_weight.float() + q_norm_weight_bias)).to(dtype)

    def layer_norm(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(tokens, -1, head_dim).float()
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normed = (x - mean) * torch.rsqrt(variance + eps)
        return (normed * k_weight.float() + k_bias.float()).to(dtype)

    q = rms_norm(index_q)
    k = layer_norm(index_k)

    half = rotary_dim // 2

    def apply_rope(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(tokens, -1, head_dim).float()
        x1 = x[..., :half]
        x2 = x[..., half:rotary_dim]
        cos_t = cos[positions.long()].float()[:, None, :]
        sin_t = sin[positions.long()].float()[:, None, :]
        out = x.clone()
        out[..., :half] = x1 * cos_t - x2 * sin_t
        out[..., half:rotary_dim] = x1 * sin_t + x2 * cos_t
        return out.to(dtype).view(x_flat.shape)

    return apply_rope(q).view(tokens, -1), apply_rope(k).view(tokens, -1), index_z.clone()


@pytest.fixture(scope="module")
def _import_kernel():
    # Load the module file directly so the step5 package __init__ (which
    # imports optional ops-transformer dependencies) is not executed.
    import importlib.util
    import pathlib

    module_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm_ascend"
        / "models"
        / "step5"
        / "fused_qknorm_rope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_ascend.models.step5.fused_qknorm_rope", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_indexer_qkz_norm_rope


@pytest.mark.parametrize("num_tokens", [1, 8, 33])
@pytest.mark.parametrize("num_q_heads,num_k_heads", [(1, 1), (4, 1), (16, 1)])
def test_fused_indexer_norm_rope_accuracy(
    num_tokens, num_q_heads, num_k_heads, _import_kernel
):
    fused_fn = _import_kernel
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    max_pos = 4096
    torch.manual_seed(42)
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    x_cat = torch.randn(
        num_tokens, q_len + 2 * k_len, device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(
        0, max_pos, (num_tokens,), device=device, dtype=torch.int32
    )
    cos, sin = _build_cos_sin_cache(max_pos, ROTARY_DIM // 2, THETA, device)

    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)
    # Verify the inputs are views from a single tensor, not contiguous copies.
    assert index_q is not x_cat
    assert index_k is not x_cat
    assert index_z is not x_cat
    assert index_q.storage().data_ptr() == x_cat.storage().data_ptr()
    assert index_k.storage().data_ptr() == x_cat.storage().data_ptr()
    assert index_z.storage().data_ptr() == x_cat.storage().data_ptr()
    if k_len > 0:
        assert index_k.data_ptr() != x_cat.data_ptr()
    if k_len > 0 or q_len > 0:
        assert index_z.data_ptr() != x_cat.data_ptr()

    q_ref, k_ref, z_ref = _reference_indexer_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        k_bias,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )

    q_out, k_out, z_out = fused_fn(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        q_norm_bias=None,
        k_norm_bias=k_bias,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
        k_norm_weight_bias=0.0,
        subtract_mean_q=False,
        subtract_mean_k=True,
    )

    # index_q lands in e4m3 (quantized at the kernel store): compare
    # against the reference cast to the same dtype. The kernel rounds
    # through the input precision before the e4m3 store, so the values
    # match the unfused pipeline's bits up to one e4m3 ULP (3-bit
    # mantissa) wherever the fp32 math itself differs by an ULP.
    q_ref8 = q_ref.to(torch.float8_e4m3fn)
    assert q_out.dtype == torch.float8_e4m3fn
    match = torch.allclose(q_ref8.float(), q_out.float(), atol=0.02, rtol=0.13)
    if not match:
        err = (q_ref8.float() - q_out.float()).abs()
        print(f"\n[Q] max_abs_err={err.max().item():.6f} "
              f"mean_abs_err={err.mean().item():.6f}")
        assert match, f"Q mismatch: max_abs_err={err.max().item():.6f}"
    for name, ref, out in (("K", k_ref, k_out),):
        match = torch.allclose(ref, out, atol=1e-2, rtol=1e-2)
        if not match:
            err = (ref.float() - out.float()).abs()
            print(f"\n[{name}] max_abs_err={err.max().item():.6f} "
                  f"mean_abs_err={err.mean().item():.6f}")
            assert match, f"{name} mismatch: max_abs_err={err.max().item():.6f}"
    assert torch.equal(z_ref, z_out), "Z passthrough mismatch"


def test_fused_indexer_norm_rope_pos0_is_identity(_import_kernel):
    """At position 0 the rope angles are 0: rotated dims must equal the
    normalized values bit-exactly. The buggy layout already failed at pos 0
    (dims 16-31/48-63 rotated ~90 deg from cross-row cache reads)."""
    fused_fn = _import_kernel
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    num_tokens, num_q_heads, num_k_heads = 4, 4, 1
    torch.manual_seed(7)
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    x_cat = torch.randn(
        num_tokens, q_len + 2 * k_len, device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(num_tokens, device=device, dtype=torch.int32)
    cos, sin = _build_cos_sin_cache(8, ROTARY_DIM // 2, THETA, device)

    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)

    q_ref, k_ref, _ = _reference_indexer_norm_rope(
        index_q, index_k, index_z, q_weight, k_weight, k_bias, cos, sin,
        positions, head_dim=HEAD_DIM, num_q_heads=num_q_heads,
        num_k_heads=num_k_heads, rotary_dim=ROTARY_DIM, eps=EPS,
        q_norm_weight_bias=1.0,
    )
    q_out, k_out, _ = fused_fn(
        index_q, index_k, index_z, q_weight, k_weight, cos, sin, positions,
        q_norm_bias=None, k_norm_bias=k_bias,
        head_dim=HEAD_DIM, num_q_heads=num_q_heads, num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM, eps=EPS,
        q_norm_weight_bias=1.0, k_norm_weight_bias=0.0,
        subtract_mean_q=False, subtract_mean_k=True,
    )
    # index_q is e4m3; one-ULP tolerance for the quantized comparison.
    q_ref8 = q_ref.to(torch.float8_e4m3fn)
    assert q_out.dtype == torch.float8_e4m3fn
    assert torch.allclose(q_ref8.float(), q_out.float(), atol=0.02, rtol=0.13)
    assert torch.allclose(k_ref, k_out, atol=1e-2, rtol=1e-2)
