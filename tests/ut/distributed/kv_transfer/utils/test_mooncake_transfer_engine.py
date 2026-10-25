# SPDX-License-Identifier: Apache-2.0
"""Source-isolated CPU checks: never load Mooncake or CANN native libraries."""

import builtins
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py"


@pytest.fixture
def loader():
    events = []
    handle = SimpleNamespace()
    envs = SimpleNamespace(ASCEND_HOME_PATH=None)
    package = ModuleType("vllm_ascend")
    package.envs = envs
    engine = Mock()
    engine.initialize.return_value = 0

    def construct():
        events.append("construct")
        return engine

    factory = Mock(side_effect=construct)
    native = Mock(side_effect=lambda *args, **kwargs: events.append("load") or handle)
    original_import = builtins.__import__
    import_error = None

    def import_module(name, *args, **kwargs):
        if name == "mooncake.engine":
            events.append("import")
            if state.import_error is not None:
                raise state.import_error
            return SimpleNamespace(TransferEngine=factory)
        return original_import(name, *args, **kwargs)

    state = SimpleNamespace(
        events=events,
        handle=handle,
        envs=envs,
        engine=engine,
        factory=factory,
        native=native,
        import_error=import_error,
    )
    with (
        patch.dict("sys.modules", {"vllm_ascend": package}),
        patch("ctypes.CDLL", native),
        patch("builtins.__import__", side_effect=import_module),
    ):
        namespace = {"__name__": "isolated_mooncake_transfer_engine"}
        exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), namespace)
        state.global_te = namespace["global_te"]
        yield state


def test_import_and_construction_do_not_load_native_libraries(loader):
    assert loader.events == []
    assert loader.global_te.transfer_engine is None


@pytest.mark.parametrize("ascend_home", [None, "/configured/cann"])
def test_cann_load_precedes_import_and_engine_is_cached(loader, ascend_home):
    loader.envs.ASCEND_HOME_PATH = ascend_home

    def initialize(*args):
        loader.events.append("initialize")
        assert loader.global_te.transfer_engine is None
        assert loader.global_te._cann_library is loader.handle
        return 0

    loader.engine.initialize.side_effect = initialize
    assert loader.global_te.get_transfer_engine("host", None) is loader.engine
    assert loader.events == ["load", "import", "construct", "initialize"]
    loader.native.assert_called_once_with("libllm_datadist.so", mode=ctypes.RTLD_GLOBAL)
    loader.engine.initialize.assert_called_once_with("host", "P2PHANDSHAKE", "ascend", "")
    assert loader.global_te._cann_library is loader.handle
    assert loader.global_te.get_transfer_engine("ignored", "ignored") is loader.engine
    assert len(loader.events) == 4


@pytest.mark.parametrize("subdirectory", ["lib64", "aarch64-linux/lib64", "x86_64-linux/lib64"])
def test_cann_fallback_uses_configured_toolkit_path(loader, tmp_path, subdirectory):
    loader.envs.ASCEND_HOME_PATH = str(tmp_path)
    library = tmp_path / subdirectory / "libllm_datadist.so"
    library.parent.mkdir(parents=True)
    library.touch()

    def load(path, *, mode):
        assert mode == ctypes.RTLD_GLOBAL
        if path == str(library):
            return loader.handle
        raise OSError("not found")

    loader.native.side_effect = load
    assert loader.global_te.get_transfer_engine("host", "device") is loader.engine
    assert loader.native.call_args.args == (str(library),)
    loader.engine.initialize.assert_called_once_with("host", "P2PHANDSHAKE", "ascend", "device")
    assert loader.global_te._cann_library is loader.handle


def test_missing_cann_fails_before_import_and_can_retry(loader):
    loader.native.side_effect = OSError("not found")
    with pytest.raises(RuntimeError, match="libllm_datadist.so.*ASCEND_HOME_PATH") as failure:
        loader.global_te.get_transfer_engine("host", None)
    assert isinstance(failure.value.__cause__, OSError)
    assert loader.events == []
    assert loader.global_te.transfer_engine is None
    assert loader.global_te._cann_library is None
    loader.native.side_effect = None
    loader.native.return_value = loader.handle
    assert loader.global_te.get_transfer_engine("host", None) is loader.engine


def test_missing_configured_cann_reports_attempts_and_loader_error(loader, tmp_path):
    loader.envs.ASCEND_HOME_PATH = str(tmp_path)
    loader.native.side_effect = OSError("dependency unavailable")
    with pytest.raises(RuntimeError, match="dependency unavailable") as failure:
        loader.global_te.get_transfer_engine("host", None)
    assert str(tmp_path / "lib64/libllm_datadist.so") in str(failure.value)
    assert loader.native.call_count == 4
    assert loader.events == []
    assert loader.global_te._cann_library is None


@pytest.mark.parametrize("stage", ["import", "construct", "initialize_exception", "initialize_return"])
def test_failure_keeps_cann_handle_and_retries_without_publishing(loader, stage):
    if stage == "import":
        loader.import_error = ImportError("missing mooncake")
        expected = ImportError
    elif stage == "construct":
        loader.factory.side_effect = RuntimeError("construction failed")
        expected = RuntimeError
    elif stage == "initialize_exception":
        loader.engine.initialize.side_effect = RuntimeError("initialization failed")
        expected = RuntimeError
    else:
        loader.engine.initialize.return_value = -1
        expected = RuntimeError
    with pytest.raises(expected):
        loader.global_te.get_transfer_engine("host", None)
    assert loader.global_te.transfer_engine is None
    assert loader.global_te._cann_library is loader.handle
    loader.import_error = None
    loader.factory.side_effect = None
    loader.factory.return_value = loader.engine
    loader.engine.initialize.side_effect = None
    loader.engine.initialize.return_value = 0
    assert loader.global_te.get_transfer_engine("host", None) is loader.engine
    loader.native.assert_called_once()


def test_concurrent_calls_publish_only_one_initialized_engine(loader):
    callers = 8
    barrier = threading.Barrier(callers)
    entered = threading.Event()
    release = threading.Event()

    def initialize(*args):
        entered.set()
        assert release.wait(timeout=5)
        return 0

    def request_engine():
        barrier.wait(timeout=5)
        return loader.global_te.get_transfer_engine("host", None)

    loader.engine.initialize.side_effect = initialize
    with ThreadPoolExecutor(max_workers=callers) as pool:
        futures = [pool.submit(request_engine) for _ in range(callers)]
        try:
            assert entered.wait(timeout=5)
            assert loader.global_te.transfer_engine is None
            assert all(not future.done() for future in futures)
        finally:
            release.set()
        assert all(future.result(timeout=5) is loader.engine for future in futures)
    loader.native.assert_called_once()
    loader.factory.assert_called_once()
    loader.engine.initialize.assert_called_once()
    assert loader.events.count("import") == 1


def test_register_buffer_semantics_are_unchanged(loader):
    with pytest.raises(AssertionError, match="must be initialized"):
        loader.global_te.register_buffer([10], [20])
    loader.global_te.get_transfer_engine("host", None)
    loader.engine.register_memory.side_effect = [-1, 0, 0]
    with pytest.raises(RuntimeError, match="registration failed"):
        loader.global_te.register_buffer([10], [20])
    assert not loader.global_te.is_register_buffer
    loader.global_te.register_buffer([10, 30], [20, 40])
    assert loader.global_te.is_register_buffer
    loader.global_te.register_buffer([50], [60])
    assert loader.engine.register_memory.call_count == 3
