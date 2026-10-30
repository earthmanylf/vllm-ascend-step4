# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests executing the real Step5 class and complete patch module."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_step5_rms_norm.py"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def norm_patch(monkeypatch):
    # Locate the source without importing vLLM's dependency-heavy package.
    source = next(
        (
            Path(root) / "vllm/models/step5/nvidia/layernorm.py"
            for root in sys.path
            if (Path(root) / "vllm/models/step5/nvidia/layernorm.py").is_file()
        ),
        None,
    )
    assert source is not None, "Put the upstream vLLM source checkout on PYTHONPATH"
    upstream = load_source("step5_layernorm_reference", source)
    original = upstream.Step5RMSNorm.forward
    monkeypatch.setitem(sys.modules, "vllm.models.step5.nvidia.layernorm", upstream)
    npu = ModuleType("torch_npu")

    def rms_norm(x, scale, epsilon):
        assert x.dtype == scale.dtype == torch.float32
        result = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * scale
        return result, None

    npu.npu_rms_norm = Mock(side_effect=rms_norm)
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    assert PATCH_PATH.is_file(), "Step5 RMSNorm Ascend patch module is missing"
    patch = load_source("step5_rms_norm_candidate", PATCH_PATH)
    return upstream, original, patch, npu


def run_contract(
    norm_patch,
    monkeypatch,
    *,
    npu,
    dtype,
    weight_dtype,
    zero_centered,
    residual_dtype=None,
    out_dtype=None,
    buffer=False,
):
    upstream, original, patch, fused = norm_patch
    if npu:
        monkeypatch.setattr(torch.Tensor, "is_npu", property(lambda self: True), raising=False)
    torch.manual_seed(42)
    norm = upstream.Step5RMSNorm(32, zero_centered=zero_centered, dtype=weight_dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32, dtype=weight_dtype) * 0.1)
    x = torch.randn(2, 3, 32, dtype=dtype)
    residual = None if residual_dtype is None else torch.randn(x.shape, dtype=residual_dtype)
    before_x = x.clone()
    before_residual = None if residual is None else residual.clone()
    output = torch.empty_like(x, dtype=out_dtype or dtype) if buffer else None
    expected = original(norm, x, residual, out_dtype=out_dtype)
    actual = norm(x, residual, output=output, out_dtype=out_dtype)
    if residual is not None:
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        assert actual[1] is not residual and actual[1] is not x
        torch.testing.assert_close(residual, before_residual, rtol=0, atol=0)
        actual, expected = actual[0], expected[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    if buffer:
        assert actual is output
    assert fused.npu_rms_norm.call_count == int(npu)
    if npu:
        scale = fused.npu_rms_norm.call_args.args[1]
        expected_scale = norm.weight.float() + (1.0 if zero_centered else 0.0)
        torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("npu", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("out_dtype", [None, torch.float32, torch.bfloat16])
def test_numerics_dtype_and_output(norm_patch, monkeypatch, npu, dtype, zero_centered, out_dtype):
    run_contract(
        norm_patch,
        monkeypatch,
        npu=npu,
        dtype=dtype,
        weight_dtype=dtype,
        zero_centered=zero_centered,
        out_dtype=out_dtype,
        buffer=True,
    )


@pytest.mark.parametrize("npu", [False, True])
@pytest.mark.parametrize("residual_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("out_dtype", [None, torch.bfloat16])
def test_residual_promotion_no_mutation(norm_patch, monkeypatch, npu, residual_dtype, zero_centered, out_dtype):
    run_contract(
        norm_patch,
        monkeypatch,
        npu=npu,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        zero_centered=zero_centered,
        residual_dtype=residual_dtype,
        out_dtype=out_dtype,
    )


@pytest.mark.parametrize("npu", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"fp16_out": True},
        {"residual": torch.ones(2, 32), "fp16_out": True},
        {"residual": torch.ones(2, 32), "output": torch.empty(2, 32)},
        {"output": torch.empty(2, 31)},
        {"output": torch.empty(2, 32, dtype=torch.bfloat16)},
        {"output": torch.empty(2, 32), "out_dtype": torch.bfloat16},
    ],
)
def test_invalid_options(norm_patch, monkeypatch, npu, options):
    upstream, original, patch, fused = norm_patch
    if npu:
        monkeypatch.setattr(torch.Tensor, "is_npu", property(lambda self: True), raising=False)
    norm = upstream.Step5RMSNorm(32)
    x = torch.ones(2, 32)
    with pytest.raises(ValueError) as expected:
        original(norm, x, **options)
    with pytest.raises(ValueError, match=str(expected.value).replace(".", r"\.")):
        norm(x, **options)
    fused.npu_rms_norm.assert_not_called()


def test_cpu_delegates_unchanged(norm_patch, monkeypatch):
    upstream, original, patch, fused = norm_patch
    sentinel = object()
    fallback = Mock(return_value=sentinel)
    monkeypatch.setattr(patch, "_ORIGINAL_FORWARD", fallback)
    norm = upstream.Step5RMSNorm(32)
    x = torch.ones(2, 32)
    assert norm(x, out_dtype=torch.bfloat16) is sentinel
    fallback.assert_called_once_with(norm, x, None, None, False, out_dtype=torch.bfloat16)
    fused.npu_rms_norm.assert_not_called()


def test_noncontiguous_output_identity(norm_patch, monkeypatch):
    upstream, original, patch, fused = norm_patch
    monkeypatch.setattr(torch.Tensor, "is_npu", property(lambda self: True), raising=False)
    norm = upstream.Step5RMSNorm(32, zero_centered=True)
    x = torch.randn(2, 32)
    output = torch.empty(32, 2).t()
    assert not output.is_contiguous()
    expected = original(norm, x)
    assert norm(x, output=output) is output
    torch.testing.assert_close(output, expected)


def test_unsupported_dtype_falls_back(norm_patch, monkeypatch):
    upstream, original, patch, fused = norm_patch
    monkeypatch.setattr(torch.Tensor, "is_npu", property(lambda self: True), raising=False)
    norm = upstream.Step5RMSNorm(32, dtype=torch.float64)
    x = torch.randn(2, 32, dtype=torch.float64)
    torch.testing.assert_close(norm(x), original(norm, x), rtol=0, atol=0)
    fused.npu_rms_norm.assert_not_called()


def test_patch_module_exists():
    assert PATCH_PATH.is_file(), "Step5 RMSNorm Ascend patch module is missing"


def test_worker_registers_patch():
    worker = PATCH_PATH.parent / "__init__.py"
    assert "import vllm_ascend.patch.worker.patch_step5_rms_norm" in worker.read_text()
