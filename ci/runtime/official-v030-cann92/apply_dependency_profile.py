"""Audited dependency alignment for TailRing on the retained torch/CANN stack."""

import json
from pathlib import Path

root = Path("/opt/upgrade/src")
ledger = []
replacements = {
    "vllm": {"torch == 2.14.0": "torch == 2.10.0"},
    "vllm-ascend": {
        "triton-ascend==3.2.2": "triton-ascend==3.6.0+dev20261013182112",
        "fastapi<0.124.0": "fastapi>=0.133.0,<0.137.0",
    },
}
for repo, changes in replacements.items():
    for name in ("requirements.txt", "pyproject.toml"):
        path = root / repo / name
        if not path.exists():
            continue
        text = path.read_text()
        for old, new in changes.items():
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                ledger.append({"repo": repo, "file": name, "old": old, "new": new, "count": count})
        path.write_text(text)
if not any(x["repo"] == "vllm" for x in ledger):
    raise RuntimeError("Unexpected TailRing vLLM build dependency profile")
Path("/opt/upgrade/evidence/dependency-profile-patch.json").write_text(json.dumps(ledger, indent=2) + "\n")
Path("/opt/upgrade/constraints-tailring.txt").write_text(
    "torch==2.10.0+cpu\n"
    "torch-npu==2.10.0.post4\n"
    "torchvision==0.25.0\n"
    "torchaudio==2.10.0\n"
    "triton-ascend==3.6.0+dev20261013182112\n"
    "numpy<2.0\n"
    "compressed-tensors==0.17.0\n"
)
