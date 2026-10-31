"""Audited dependency alignment for TailRing on the retained torch/CANN stack."""

import json
from pathlib import Path

root = Path("/opt/upgrade/src")
ledger = []
replacements = {
    "vllm": {
        "torch == 2.14.0": "torch == 2.10.0",
        "opencv-python-headless >= 4.13.0": "# opencv-python-headless installed separately",
        # The [image] extra only adds opencv-python-headless>=4.0.0, whose 4.13
        # metadata demands NumPy 2. Drop the extra and install the audited
        # OpenCV build explicitly instead.
        "mistral_common[image] >= 1.11.6": "mistral_common >= 1.11.6",
    },
    "vllm-ascend": {
        "triton-ascend==3.2.2": "triton-ascend==3.6.0+dev20261013182112",
        "fastapi<0.124.0": "fastapi>=0.133.0,<0.137.0",
    },
}
for repo, changes in replacements.items():
    for name in ("requirements.txt", "pyproject.toml", "requirements/common.txt"):
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
if not any(x["new"] == "# opencv-python-headless installed separately" for x in ledger):
    raise RuntimeError("vLLM OpenCV requirement was not moved to the staged install")
if not any(x["new"] == "mistral_common >= 1.11.6" for x in ledger):
    raise RuntimeError("mistral_common image extra was not detached from the NumPy pin")
Path("/opt/upgrade/evidence/dependency-profile-patch.json").write_text(json.dumps(ledger, indent=2) + "\n")
Path("/opt/upgrade/constraints-tailring.txt").write_text(
    "torch==2.10.0+cpu\n"
    "torch-npu==2.10.0.post4\n"
    "torchvision==0.25.0+cpu\n"
    "torchaudio==2.10.0+cpu\n"
    "triton-ascend==3.6.0+dev20261013182112\n"
    "numpy==1.26.4\n"
    "compressed-tensors==0.17.0\n"
    "opencv-python-headless==4.13.0.90\n"
)
