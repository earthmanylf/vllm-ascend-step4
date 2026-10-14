# SPDX-License-Identifier: Apache-2.0
"""Sparse KV offload must not provision device packages during worker imports."""

import ast
import builtins
import subprocess
import sys
import types
import typing
from pathlib import Path

import pytest


@pytest.fixture
def load_backend():
    path = (
        Path(__file__).resolve().parents[4]
        / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py"
    )
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_offload_backend")
    namespace = {"typing": typing}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[fn.name]


def test_cold_module_import_does_not_load_memfabric():
    # Execute the real module in a fresh interpreter so an earlier SDK import
    # cannot hide the regression behind Python's module cache.
    code = """
import importlib.abc
import sys
attempts = []
class RejectMemfabric(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'memfabric_hybrid' or fullname.startswith('memfabric_hybrid.'):
            attempts.append(fullname)
            raise AssertionError('Sparse offload imported its SDK during module import')
sys.meta_path.insert(0, RejectMemfabric())
import vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager
assert not attempts, attempts
assert 'memfabric_hybrid' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_enabled_offload_resolves_the_real_backend_api(load_backend, monkeypatch):
    backend = object()
    sdk = types.ModuleType("memfabric_hybrid")
    sdk.offload = backend
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", sdk)
    assert load_backend() is backend


def test_missing_optional_sdk_has_actionable_error(load_backend, monkeypatch):
    original = builtins.__import__

    def missing_sdk(name, *args, **kwargs):
        if name == "memfabric_hybrid":
            raise ModuleNotFoundError("SDK missing", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sdk)
    with pytest.raises(RuntimeError, match="requires memfabric_hybrid"):
        load_backend()


def test_broken_sdk_dependency_is_not_suppressed(load_backend, monkeypatch):
    original = builtins.__import__

    def broken_sdk(name, *args, **kwargs):
        if name == "memfabric_hybrid":
            raise ModuleNotFoundError("SDK dependency missing", name="vendor_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_sdk)
    with pytest.raises(ModuleNotFoundError) as error:
        load_backend()
    assert error.value.name == "vendor_dependency"
