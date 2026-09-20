"""Unit tests for the Step-4 fused QKNorm+RoPE Triton kernel.

Tests compare the fused kernel output against a pure-PyTorch reference that
replicates the eager path: per-head RMSNorm (zero_centered) + NeoX RoPE.

RoPE angles are supplied through a cos/sin cache built with a configurable
theta. The theta=1e7 parametrization is a regression test for the bug where
the kernel ignored the caller-provided cache and recomputed angles from a
hardcoded theta=1e4 (every DSA layer uses rope_theta=1e7).

Run on NPU:
    pytest -sv tests/ut/ops/test_step4_fused_qknorm_rope.py
"""

import pytest
import torch

HEAD_DIM = 192
EPS = 1e-5
THETA_SLIDING = 10000.0
THETA_DSA = 10000000.0


def _build_cos_sin_cache(
    max_pos: int, rotary_pairs: int, theta: float, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a [max_pos, 2*rotary_pairs] cos_sin_cache and chunk it, matching
    the production layout (vLLM RotaryEmbedding.cos_sin_cache.chunk(2, -1)),
    so the kernel sees the same strided views as in the model."""
    span = 2 * rotary_pairs
    idx = torch.arange(0, span, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / span))
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [max_pos, rotary_pairs]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    return cache.chunk(2, dim=-1)  # cos, sin: [max_pos, rotary_pairs] views


def _reference_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    norm_weight_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference: split → per-head RMSNorm → NeoX RoPE from the
    given cos/sin cache (same tensors passed to the kernel)."""
    total_tokens = qkv.shape[0]
    q_dim = num_q_heads * HEAD_DIM
    kv_dim = num_kv_heads * HEAD_DIM

    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

    # --- RMSNorm (zero_centered variant) ---
    def rms_norm(x_flat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(total_tokens, -1, HEAD_DIM).float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        scale = weight.float() + norm_weight_bias
        return (normed * scale).to(qkv.dtype).view(x_flat.shape)

    q = rms_norm(q, q_weight)
    k = rms_norm(k, k_weight)

    # --- NeoX RoPE (partial, only first rotary_dim dims) ---
    def apply_rope(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(total_tokens, -1, HEAD_DIM).float()
        half = rotary_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:rotary_dim]
        cos_t = cos[positions.long()].float()[:, None, :]  # [tokens, 1, half]
        sin_t = sin[positions.long()].float()[:, None, :]
        rotated1 = x1 * cos_t - x2 * sin_t
        rotated2 = x1 * sin_t + x2 * cos_t
        x_out = x.clone()
        x_out[..., :half] = rotated1
        x_out[..., half:rotary_dim] = rotated2
        return x_out.to(qkv.dtype).view(x_flat.shape)

    q = apply_rope(q)
    k = apply_rope(k)
    return q, k, v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _import_kernel():
    from vllm_ascend.models.step4.fused_qknorm_rope import fused_qknorm_rope
    return fused_qknorm_rope


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [THETA_SLIDING, THETA_DSA])
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32, 48])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (16, 2), (32, 4)])
@pytest.mark.parametrize("norm_weight_bias", [1.0, 0.0])
def test_fused_qknorm_rope_accuracy(
    num_tokens,
    num_q_heads,
    num_kv_heads,
    norm_weight_bias,
    theta,
    _import_kernel,
):
    """Compare fused kernel output against PyTorch reference."""
    fused_fn = _import_kernel
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    rotary_dim = 128
    rotary_pairs = rotary_dim // 2
    max_pos = 4096

    torch.manual_seed(123)
    q_dim = num_q_heads * HEAD_DIM
    kv_dim = num_kv_heads * HEAD_DIM
    qkv = torch.randn(num_tokens, q_dim + 2 * kv_dim, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)
    cos, sin = _build_cos_sin_cache(max_pos, rotary_pairs, theta, device)

    # Reference
    q_ref, k_ref, v_ref = _reference_qknorm_rope(
        qkv.clone(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_dim=rotary_dim,
        eps=EPS,
        norm_weight_bias=norm_weight_bias,
    )

    # Fused kernel
    q_out, k_out, v_out = fused_fn(
        qkv.clone(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=EPS,
        norm_weight_bias=norm_weight_bias,
    )

    # --- Q comparison ---
    q_atol = 1e-2
    q_rtol = 1e-2
    q_match = torch.allclose(q_ref, q_out, atol=q_atol, rtol=q_rtol)
    if not q_match:
        q_abs_err = (q_ref.float() - q_out.float()).abs()
        print(f"\n[Q] max_abs_err={q_abs_err.max().item():.6f}  "
              f"mean_abs_err={q_abs_err.mean().item():.6f}")
    assert q_match, (
        f"Q output mismatch: max_abs_err={q_abs_err.max().item():.6f}"
    )

    # --- K comparison ---
    k_match = torch.allclose(k_ref, k_out, atol=q_atol, rtol=q_rtol)
    if not k_match:
        k_abs_err = (k_ref.float() - k_out.float()).abs()
        print(f"\n[K] max_abs_err={k_abs_err.max().item():.6f}  "
              f"mean_abs_err={k_abs_err.mean().item():.6f}")
    assert k_match, (
        f"K output mismatch: max_abs_err={k_abs_err.max().item():.6f}"
    )

    # --- V comparison (should be exact copy) ---
    v_ref_full = qkv[:, q_dim + kv_dim:].clone()
    assert torch.equal(v_ref_full, v_out), "V output mismatch (should be exact copy)"


def test_fused_qknorm_rope_zero_tokens(_import_kernel):
    """Zero-token input should return empty tensors without error."""
    fused_fn = _import_kernel
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    num_q_heads, num_kv_heads = 8, 1
    rotary_pairs = 64
    q_dim = num_q_heads * HEAD_DIM
    kv_dim = num_kv_heads * HEAD_DIM

    qkv = torch.empty(0, q_dim + 2 * kv_dim, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.empty(0, dtype=torch.int32, device=device)
    cos = torch.empty(0, 128, device=device, dtype=torch.bfloat16)
    sin = torch.empty(0, 128, device=device, dtype=torch.bfloat16)

    q_out, k_out, v_out = fused_fn(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=EPS,
        norm_weight_bias=1.0,
    )

    assert q_out.shape == (0, q_dim)
    assert k_out.shape == (0, kv_dim)
    assert v_out.shape == (0, kv_dim)


def test_fused_qknorm_rope_rope_only_region(_import_kernel):
    """Verify that non-RoPE dims (rotary_dim..HEAD_DIM) pass through norm-only."""
    fused_fn = _import_kernel
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    num_tokens, num_q_heads, num_kv_heads = 16, 8, 1
    rotary_dim = 128
    rotary_pairs = rotary_dim // 2
    max_pos = 4096
    q_dim = num_q_heads * HEAD_DIM
    kv_dim = num_kv_heads * HEAD_DIM

    torch.manual_seed(777)
    qkv = torch.randn(num_tokens, q_dim + 2 * kv_dim, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int32)
    cos, sin = _build_cos_sin_cache(max_pos, rotary_pairs, THETA_SLIDING, device)

    q_ref, k_ref, v_ref = _reference_qknorm_rope(
        qkv.clone(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_dim=rotary_dim,
        eps=EPS,
        norm_weight_bias=1.0,
    )

    q_out, k_out, v_out = fused_fn(
        qkv.clone(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=EPS,
        norm_weight_bias=1.0,
    )

    # Check non-RoPE region separately (should be just RMSNorm output, no rotation)
    q_ref_view = q_ref.view(num_tokens, num_q_heads, HEAD_DIM)
    q_out_view = q_out.view(num_tokens, num_q_heads, HEAD_DIM)
    non_rope_ref = q_ref_view[..., rotary_dim:]
    non_rope_out = q_out_view[..., rotary_dim:]
    assert torch.allclose(non_rope_ref, non_rope_out, atol=1e-2, rtol=1e-2), (
        f"Non-RoPE region mismatch: max_err={((non_rope_ref - non_rope_out).abs().max()).item():.6f}"
    )


def test_fused_qknorm_rope_max_abs_error_report(_import_kernel):
    """Diagnostic test: print detailed error breakdown for debugging."""
    fused_fn = _import_kernel
    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("NPU required for Triton kernel")

    num_tokens, num_q_heads, num_kv_heads = 32, 16, 2
    rotary_dim = 128
    rotary_pairs = rotary_dim // 2
    max_pos = 4096
    q_dim = num_q_heads * HEAD_DIM
    kv_dim = num_kv_heads * HEAD_DIM

    torch.manual_seed(0)
    qkv = torch.randn(num_tokens, q_dim + 2 * kv_dim, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)
    cos, sin = _build_cos_sin_cache(max_pos, rotary_pairs, THETA_DSA, device)

    q_ref, k_ref, _ = _reference_qknorm_rope(
        qkv.clone(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_dim=rotary_dim,
        eps=EPS,
        norm_weight_bias=1.0,
    )

    q_out, k_out, _ = fused_fn(
        qkv.clone(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rotary_pairs=rotary_pairs,
        eps=EPS,
        norm_weight_bias=1.0,
    )

    q_err = (q_ref.float() - q_out.float()).abs()
    k_err = (k_ref.float() - k_out.float()).abs()
    print(f"\n{'='*60}")
    print(f"Q: max_abs_err={q_err.max().item():.8f}  mean_abs_err={q_err.mean().item():.8f}")
    print(f"K: max_abs_err={k_err.max().item():.8f}  mean_abs_err={k_err.mean().item():.8f}")

    # Per-head-dim error breakdown for Q
    q_ref_h = q_ref.view(num_tokens, num_q_heads, HEAD_DIM)
    q_out_h = q_out.view(num_tokens, num_q_heads, HEAD_DIM)
    per_dim_err = (q_ref_h - q_out_h).abs().mean(dim=(0, 1))
    print(f"\nQ per-dim error (first 16 dims): {per_dim_err[:16].tolist()}")
    print(f"Q per-dim error (dims 60-70):   {per_dim_err[60:70].tolist()}")
    print(f"Q per-dim error (dims 128-144): {per_dim_err[128:144].tolist()}")
    print(f"{'='*60}")
