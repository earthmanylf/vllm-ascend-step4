# SPDX-License-Identifier: Apache-2.0
"""Apply the retained 950 native source patch, failing closed on source drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def apply(source: Path, bundle: Path = BUNDLE) -> dict:
    source = source.resolve(strict=True)
    identity = json.loads((bundle / "source-identity.json").read_text())
    patch = bundle / "native.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != identity["patch_sha256"]:
        raise ValueError("Native patch hash mismatch")
    for entry in identity["changes"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid native source path")
        target = source / relative
        if not target.resolve().is_relative_to(source):
            raise ValueError("Native source symlink escapes source root")
        if entry["before_sha256"] is None:
            if target.exists():
                raise ValueError(f"Unexpected base file: {relative}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != entry["before_sha256"]:
            raise ValueError(f"Native base source mismatch: {relative}")
    command = ["patch", "-p1", "--batch", "--forward", "-i", str(patch.resolve())]
    subprocess.run(command + ["--dry-run"], cwd=source, check=True)
    subprocess.run(command, cwd=source, check=True)
    for entry in identity["changes"]:
        target = source / entry["path"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["after_sha256"]:
            raise ValueError(f"Patched native source mismatch: {entry['path']}")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    identity = apply(args.source)
    print(json.dumps({"source_verified": True, "files": len(identity["changes"]), "abi_version": 1}))


if __name__ == "__main__":
    main()
