"""Validate actual runtime identity and write an image-only build receipt."""

import argparse
import datetime
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
for name in ("ci-dir", "context", "output"):
    parser.add_argument("--" + name, required=True, type=Path)
parser.add_argument("--image", required=True)
args = parser.parse_args()
lock = json.loads((args.ci_dir / "pins/official-v030-cann92.json").read_text())
image = json.loads((args.context / "built-image-inspect.json").read_text())[0]
runtime = json.loads((args.context / "runtime-identity.json").read_text())
for name, expected in lock["package_versions"].items():
    if runtime["packages"][name]["version"] != expected:
        raise ValueError(f"Wrong package version: {name}")
    if runtime["source_identities"][name]["commit"] != lock["source_commits"][name]:
        raise ValueError(f"Wrong source identity: {name}")
if runtime["machine"] != "aarch64":
    raise ValueError("Wrong image architecture")
for name, expected in (
    ("torch", "2.10.0+cpu"),
    ("torch-npu", "2.10.0.post4"),
    ("triton-ascend", "3.6.0+dev20261013182112"),
):
    if runtime["packages"][name]["version"] != expected:
        raise ValueError(f"Wrong retained native dependency: {name}")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "image_ref": args.image,
            "image_config_digest": image["Id"],
            "profile": lock,
            "runtime": runtime,
            "context": json.loads((args.context / "context-manifest.json").read_text()),
            "device_validation": "pending separate four-node DP8 TP4 EP32 acceptance",
        },
        indent=2,
    )
    + "\n"
)
print(args.output)
