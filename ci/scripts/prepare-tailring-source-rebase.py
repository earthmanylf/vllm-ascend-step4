"""Prepare immutable checksum-locked source snapshots; no daemon or device use."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-dir", required=True, type=Path)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ci, inputs = args.ci_dir.resolve(strict=True), args.inputs_dir.resolve(strict=True)
    lock = json.loads((ci / "pins/official-v030-cann92.json").read_text())
    recipe = ci / "docker/Dockerfile.tailring-source-rebase"
    if sha(recipe) != lock["source_rebase_recipe_sha256"]:
        raise ValueError("Source rebase Dockerfile checksum mismatch")
    names = {"vllm.tar.gz", "vllm-ascend.tar.gz", "ascend-catlass.tar.gz", lock["uv"]["file"]}
    selected = [row for row in lock["inputs"] if row["file"] in names]
    if {row["file"] for row in selected} != names:
        raise ValueError("Source rebase input lock is incomplete")
    checked, missing = [], []
    for row in selected:
        path = inputs / row["file"]
        if not path.is_file():
            missing.append(row["file"])
            continue
        if not path.resolve().is_relative_to(inputs):
            raise ValueError(f"Input symlink escapes root: {path}")
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Input checksum/size mismatch: {path}")
        checked.append(row)
    helpers = {}
    for name in ("apply_dependency_profile.py", "probe_environment.py"):
        path = ci / "runtime/official-v030-cann92" / name
        if sha(path) != lock["runtime_sha256"][name]:
            raise ValueError(f"Helper checksum mismatch: {path}")
        helpers[name] = path
    report = {
        "profile": lock["cache_profile"],
        "source_commits": lock["source_commits"],
        "verified_inputs": checked,
        "missing_inputs": missing,
        "all_inputs_available": not missing,
        "recipe_sha256": sha(recipe),
    }
    if args.check_only:
        print(json.dumps(report, indent=2))
        return
    if missing:
        raise ValueError("Missing source rebase inputs: " + ", ".join(missing))
    if args.output is None:
        raise ValueError("--output is required")
    output = args.output.absolute()
    if output != output.resolve() or output.exists():
        raise ValueError("Context must be a new canonical directory")
    for protected in (ci.parent, inputs):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Context must not overlap source or inputs")
    if shutil.disk_usage(output.parent).free < sum(row["bytes"] for row in selected) + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk space for source context")
    output.mkdir()
    for row in selected:
        shutil.copyfile(inputs / row["file"], output / row["file"])
        if sha(output / row["file"]) != row["sha256"]:
            raise ValueError(f"Input changed during copy: {row['file']}")
    for name, path in helpers.items():
        shutil.copyfile(path, output / name)
        if sha(output / name) != lock["runtime_sha256"][name]:
            raise ValueError(f"Helper changed during copy: {name}")
    shutil.copyfile(recipe, output / "Dockerfile")
    (output / ".dockerignore").write_text(
        "*\n!Dockerfile\n!*.tar.gz\n!" + lock["uv"]["file"] + "\n!apply_dependency_profile.py\n!probe_environment.py\n"
    )
    (output / "context-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
