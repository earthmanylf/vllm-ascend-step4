"""Snapshot real runtime package identities; no device use unless requested."""

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

names = [
    "vllm",
    "vllm-ascend",
    "torch",
    "torch-npu",
    "torchvision",
    "torchaudio",
    "triton",
    "triton-ascend",
    "flash-linear-attention-npu-a5",
    "flash-attn-npu",
    "cann-ops-transformer",
    "fastokens",
    "setuptools",
    "setuptools-rust",
]
result = {
    "python": sys.version,
    "machine": platform.machine(),
    "packages": {},
    "environment": {
        k: os.environ.get(k)
        for k in ["ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]
    },
}
for name in names:
    try:
        d = md.distribution(name)
        result["packages"][name] = {"version": d.version, "requires": d.requires, "root": str(d.locate_file(""))}
    except md.PackageNotFoundError:
        result["packages"][name] = None
result["source_identities"] = {}
for name in ("vllm", "vllm-ascend"):
    path = Path("/opt/upgrade/src") / name / "SOURCE-IDENTITY.json"
    result["source_identities"][name] = json.loads(path.read_text()) if path.exists() else None
p = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
result["pip_check"] = {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
print(json.dumps(result, indent=2))
