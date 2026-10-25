import ctypes
import threading
from pathlib import Path

from vllm_ascend import envs

_CANN_LIBRARY_NAME = "libllm_datadist.so"
_CANN_LIBRARY_DIRS = ("lib64", "aarch64-linux/lib64", "x86_64-linux/lib64")


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self._cann_library = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def _load_cann_library(self):
        if self._cann_library is not None:
            return
        candidates = [_CANN_LIBRARY_NAME]
        ascend_home = envs.ASCEND_HOME_PATH
        if ascend_home:
            candidates.extend(
                str(Path(ascend_home) / directory / _CANN_LIBRARY_NAME) for directory in _CANN_LIBRARY_DIRS
            )
        for candidate in candidates:
            try:
                # Retain on the process singleton even if import/initialization
                # fails: unloading CANN while the extension is alive is unsafe.
                self._cann_library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError as error:
                load_error = error
        raise RuntimeError(
            f"Cannot load CANN {_CANN_LIBRARY_NAME} before importing Mooncake. "
            "Make it available to the OS loader or set ASCEND_HOME_PATH to the CANN toolkit. "
            f"Tried: {', '.join(candidates)}. Last loader error: {load_error}"
        ) from load_error

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    self._load_cann_library()
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
                    self.transfer_engine = transfer_engine
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True


global_te = GlobalTE()
