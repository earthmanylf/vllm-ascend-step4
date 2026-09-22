# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


@pytest.fixture(scope="session")
def npu_device():
    device_id = int(os.environ.get("TEST_DEVICE_ID", "5"))
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")
