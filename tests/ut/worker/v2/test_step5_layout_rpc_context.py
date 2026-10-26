"""Layout-RPC context regression without importing the NPU worker."""

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


class LayoutRPCContextTests(unittest.TestCase):
    def worker_type(self):
        tree = ast.parse((ROOT / "vllm_ascend/worker/worker.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
        methods = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_supported_kv_cache_layouts"
        ]
        self.assertEqual(len(methods), 1, "NPU layout RPC must establish its supplied config context")
        state = SimpleNamespace(config=None, raise_error=False)

        @contextmanager
        def current(config):
            before = state.config
            state.config = config
            try:
                yield
            finally:
                state.config = before

        class Base:
            def get_supported_kv_cache_layouts(self):
                self_test.assertIs(state.config, self.vllm_config)
                if state.raise_error:
                    raise RuntimeError("backend failed")
                return ["LBNHC"] if state.config.active_pd else ["BLNHC"]

        self_test = self
        wrapper = ast.ClassDef(
            name="Worker", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
        )
        namespace = dict(Base=Base, set_current_vllm_config=current)
        code = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        exec(compile(code, "source-layout-rpc", "exec"), namespace)
        return namespace["Worker"], state

    def test_local_and_inactive_select_blnhc(self):
        worker_type, state = self.worker_type()
        for active in (False, False):
            worker = worker_type()
            worker.vllm_config = SimpleNamespace(active_pd=active)
            self.assertEqual(worker.get_supported_kv_cache_layouts(), ["BLNHC"])
            self.assertIsNone(state.config)

    def test_active_pd_keeps_compact_layout(self):
        worker_type, state = self.worker_type()
        worker = worker_type()
        worker.vllm_config = SimpleNamespace(active_pd=True)
        self.assertEqual(worker.get_supported_kv_cache_layouts(), ["LBNHC"])
        self.assertIsNone(state.config)

    def test_context_restored_on_backend_error(self):
        worker_type, state = self.worker_type()
        worker = worker_type()
        worker.vllm_config = SimpleNamespace(active_pd=False)
        sentinel = object()
        state.config, state.raise_error = sentinel, True
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            worker.get_supported_kv_cache_layouts()
        self.assertIs(state.config, sentinel)
