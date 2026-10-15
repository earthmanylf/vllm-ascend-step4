# SPDX-License-Identifier: Apache-2.0
"""Profile/bind shared DSA scratch without an accelerator or attention run."""

import ast
import math
import types
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/step5"


class Buffer:
    def __init__(self, shape, *, device="cpu", dtype="f32", storage=None):
        self.shape, self.device, self.dtype = tuple(shape), device, dtype
        self.storage = object() if storage is None else storage

    def numel(self):
        return math.prod(self.shape)

    def __getitem__(self, index):
        count = len(range(self.shape[0])[index])
        return Buffer((count, *self.shape[1:]), device=self.device, dtype=self.dtype, storage=self.storage)

    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Buffer(shape, device=self.device, dtype=self.dtype, storage=self.storage)

    def data_ptr(self):
        return id(self.storage)


class Workspace:
    def __init__(self):
        self.tensor_buffers_by_engine = {}
        self.memory_profiled = self.allocations_locked = False

    def get_order_token(self, *args, **kwargs):
        return None


class Cache:
    def __init__(self, pages, block_size, device):
        self.mean_cache = Buffer((pages, block_size // 8, 1, 256), device=device)
        self.config = object()


def load_class(filename, name, methods, namespace):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), filename, "exec"), namespace)
    return namespace[name]


class TestSummaryScratch(unittest.TestCase):
    def test_profile_without_attention_reserves_shared_metadata_and_state_before_lock(self):
        allocations = []

        def empty(shape, **kwargs):
            out = Buffer(shape, **kwargs)
            allocations.append(out)
            return out

        namespace = {
            "torch": types.SimpleNamespace(empty=empty, int64="i64", float32="f32"),
            "Step5DSAScratchWorkspace": Workspace,
            "AscendStep5SparseSummaryCache": Cache,
            "_is_graph_capturing": lambda: False,
        }
        impl_type = load_class(
            "dsa_attention.py",
            "AscendStep5DSAImpl",
            {
                "bind_scratch_workspace",
                "_get_dsa_tensor_buffer_at_least",
                "bind_summary_cache",
                "_summary_metadata_buffer",
            },
            namespace,
        )
        impl_type._dsa_virtual_engine = lambda self: 0
        layer_type = load_class(
            "summary_cache.py",
            "AscendStep5SparseSummaryCacheLayer",
            {
                "_side_storage_peers",
                "_ensure_shared_scratch_workspace",
                "prepare_kv_cache_side_storage_memory_profile",
            },
            namespace,
        )
        layers = []
        for index, rows in enumerate((16, 192)):
            impl = impl_type()
            impl.max_num_batched_tokens = rows
            impl.num_speculative_tokens = 1
            impl.num_kv_groups, impl.max_num_seqs, impl._max_regions = 1, index + 1, 8
            impl._summary_cache = None
            impl._dsa_scratch_workspace = Workspace()
            impl._dsa_scratch_bound = False
            layer = layer_type()
            layer.layer_name = f"layer{index}"
            layer._target_impl, layer._kv_cache_block_size = impl, 16
            layer._dsa_scratch_workspace = None
            layers.append(layer)
        context = {"first": layers[0], "second": layers[1], "first_alias": layers[0]}
        # Exercise the real profile hook while attention itself is never run.
        layers[0].prepare_kv_cache_side_storage_memory_profile(context, "cpu")
        workspace = layers[0]._dsa_scratch_workspace
        self.assertIs(workspace, layers[1]._dsa_scratch_workspace)
        self.assertTrue(workspace.memory_profiled and workspace.allocations_locked)
        self.assertTrue(layers[0]._owns_dsa_scratch_workspace)
        self.assertFalse(layers[1]._owns_dsa_scratch_workspace)
        buffers = workspace.tensor_buffers_by_engine[0]
        self.assertEqual(buffers["summary_prefill_groups"].numel(), 192 * 12)
        self.assertEqual(buffers["summary_prefill_state"].numel(), 192 * 513)
        self.assertEqual(buffers["summary_prefill_groups"].dtype, "i64")
        self.assertEqual(buffers["summary_prefill_state"].dtype, "f32")
        scratch_bytes = buffers["summary_prefill_groups"].numel() * 8 + buffers["summary_prefill_state"].numel() * 4
        self.assertEqual(scratch_bytes, 412416)
        pointers = {name: value.data_ptr() for name, value in buffers.items()}
        allocation_count = len(allocations)
        self.assertTrue(all(layer._target_impl._summary_cache is None for layer in layers))
        # A repeated aliased profile call and real binds must reuse allocations.
        layers[1].prepare_kv_cache_side_storage_memory_profile(context, "cpu")
        caches = [Cache(4, 16, "cpu"), Cache(4, 16, "cpu")]
        for layer, cache in zip(layers, caches):
            layer._target_impl.bind_summary_cache(cache)
            self.assertIs(layer._target_impl._summary_cache, cache)
            self.assertEqual(cache.prefill_state_scratch.data_ptr(), pointers["summary_prefill_state"])
        self.assertEqual(len(allocations), allocation_count)
        self.assertEqual(pointers, {name: value.data_ptr() for name, value in buffers.items()})
        self.assertNotEqual(caches[0].mean_cache.data_ptr(), caches[1].mean_cache.data_ptr())
        self.assertEqual(caches[0].prefill_state_scratch.shape, (16, 513))
        self.assertEqual(caches[1].prefill_state_scratch.shape, (192, 513))
        metadata = layers[1]._target_impl._summary_metadata_buffer(191, "cpu")
        self.assertEqual(metadata.shape, (191, 12))
        self.assertEqual(metadata.data_ptr(), pointers["summary_prefill_groups"])
        with self.assertRaisesRegex(RuntimeError, "capacity was exceeded"):
            layers[1]._target_impl._summary_metadata_buffer(193, "cpu")
        with self.assertRaisesRegex(RuntimeError, "capacity was exceeded"):
            layers[1]._target_impl._get_dsa_tensor_buffer_at_least(
                "summary_prefill_groups",
                (1, 12),
                device="cpu",
                dtype="f32",
            )
        self.assertEqual(len(allocations), allocation_count)


if __name__ == "__main__":
    unittest.main()
