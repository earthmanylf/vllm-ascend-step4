"""Prepare immutable checksum-locked source snapshots; no daemon or device use."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(root, name, allow_missing=False):
    path = root / name
    try:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"File symlink escapes root: {path}")
    except RuntimeError as error:
        raise ValueError(f"Invalid file symlink: {path}") from error
    if not path.exists() and not path.is_symlink() and allow_missing:
        return None
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path


def validate_input_row(row):
    if not isinstance(row, dict):
        raise ValueError("Invalid input lock row: expected an object")
    name = row.get("file")
    if not isinstance(name, str) or not all(
        part not in (".", "..") and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in name.split("/")
    ):
        raise ValueError(f"Unsafe input lock path: {name!r}")
    if (
        type(row.get("bytes")) is not int
        or row["bytes"] <= 0
        or not isinstance(row.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    ):
        raise ValueError(f"Invalid input lock size/SHA256: {name}")


def copy_verified(source, destination, expected, label, size=None):
    shutil.copyfile(source, destination)
    if (size is not None and destination.stat().st_size != size) or sha(destination) != expected:
        raise ValueError(f"{label} changed during copy: {destination.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-dir", required=True, type=Path)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ci, inputs = args.ci_dir.resolve(strict=True), args.inputs_dir.resolve(strict=True)
    if not ci.is_dir() or not inputs.is_dir():
        raise ValueError("--ci-dir and --inputs-dir must be directories")
    pins = checked_file(ci, "pins/official-v030-cann92.json")
    pins_bytes = pins.read_bytes()
    pins_sha256 = hashlib.sha256(pins_bytes).hexdigest()
    lock = json.loads(pins_bytes)
    recipe = checked_file(ci, "docker/Dockerfile.tailring-source-rebase")
    if sha(recipe) != lock["source_rebase_recipe_sha256"]:
        raise ValueError("Source rebase Dockerfile checksum mismatch")
    by_name = {}
    for row in lock["inputs"]:
        validate_input_row(row)
        if row["file"] in by_name:
            raise ValueError(f"Duplicate input lock: {row['file']}")
        by_name[row["file"]] = row
    uv = lock["uv"]
    validate_input_row(uv)
    if "/" in uv["file"] or not uv["file"].endswith(".whl"):
        raise ValueError("Source rebase uv input must be a wheel basename")
    names = {"vllm.tar.gz", "vllm-ascend.tar.gz", "ascend-catlass.tar.gz", uv["file"]}
    selected = [row for row in lock["inputs"] if row["file"] in names]
    if {row["file"] for row in selected} != names:
        raise ValueError("Source rebase input lock is incomplete")
    if any(by_name[uv["file"]][key] != uv[key] for key in ("bytes", "sha256")):
        raise ValueError("Source rebase uv identity does not match input lock")
    checked, missing, lfs_pointers = [], [], []
    for row in selected:
        path = checked_file(inputs, row["file"], allow_missing=True)
        if path is None:
            missing.append(row["file"])
            continue
        if path.stat().st_size < 1024:
            contents = path.read_bytes()
            if contents.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                lines = contents.splitlines()
                if (
                    lines.count(f"oid sha256:{row['sha256']}".encode()) != 1
                    or lines.count(f"size {row['bytes']}".encode()) != 1
                    or sum(line.startswith(b"oid ") for line in lines) != 1
                    or sum(line.startswith(b"size ") for line in lines) != 1
                ):
                    raise ValueError(f"LFS pointer does not match input lock: {path}")
                missing.append(row["file"])
                lfs_pointers.append(row["file"])
                continue
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Input checksum/size mismatch: {path}")
        checked.append(row)
    helpers = {}
    for name in ("apply_dependency_profile.py", "probe_environment.py"):
        path = checked_file(ci, "runtime/official-v030-cann92/" + name)
        if sha(path) != lock["runtime_sha256"][name]:
            raise ValueError(f"Helper checksum mismatch: {path}")
        helpers[name] = path
    report = {
        "profile": lock["cache_profile"],
        "source_branch": lock["source_branch"],
        "source_commits": lock["source_commits"],
        "source_snapshots": lock["source_snapshots"],
        "package_versions": lock["package_versions"],
        "base_image": lock["source_rebase_base_image"],
        "base_config_digest": lock["source_rebase_base_config_digest"],
        "uv": uv,
        "pins_file": "official-v030-cann92.json",
        "pins_sha256": pins_sha256,
        "runtime_sha256": {name: lock["runtime_sha256"][name] for name in helpers},
        "verified_inputs": checked,
        "missing_inputs": missing,
        "unhydrated_lfs_inputs": lfs_pointers,
        "all_inputs_available": not missing,
        "recipe_sha256": lock["source_rebase_recipe_sha256"],
        "device_or_build_validation": "not run",
    }
    if args.check_only:
        print(json.dumps(report, indent=2))
        return
    if missing:
        raise ValueError(
            "Missing source rebase inputs: " + ", ".join(missing) + "; git lfs pull hydrates shared cache objects"
        )
    if args.output is None:
        raise ValueError("--output is required")
    output = args.output.expanduser().absolute()
    if output != output.resolve() or output.exists() or output.is_symlink():
        raise ValueError("Context must be a new canonical directory")
    for protected in (ci.parent, inputs):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Context must not overlap source or inputs")
    if not output.parent.is_dir():
        raise ValueError(f"Context parent does not exist or is not a directory: {output.parent}")
    if shutil.disk_usage(output.parent).free < sum(row["bytes"] for row in selected) + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk space for source context")
    output.mkdir()
    for row in selected:
        copy_verified(checked_file(inputs, row["file"]), output / row["file"], row["sha256"], "Input", row["bytes"])
    for name in helpers:
        path = ci / "runtime/official-v030-cann92" / name
        copy_verified(
            checked_file(ci, str(path.relative_to(ci))), output / name, lock["runtime_sha256"][name], "Helper"
        )
    copy_verified(
        checked_file(ci, str(recipe.relative_to(ci))),
        output / "Dockerfile",
        lock["source_rebase_recipe_sha256"],
        "Recipe",
    )
    copy_verified(
        checked_file(ci, str(pins.relative_to(ci))), output / report["pins_file"], pins_sha256, "Pins", len(pins_bytes)
    )
    dockerignore = (
        "*\n!Dockerfile\n!*.tar.gz\n!" + lock["uv"]["file"] + "\n!apply_dependency_profile.py\n!probe_environment.py\n"
    )
    (output / ".dockerignore").write_text(dockerignore)
    expected_files = {row["file"]: row["sha256"] for row in selected}
    expected_files.update(report["runtime_sha256"])
    expected_files.update(
        {
            "Dockerfile": report["recipe_sha256"],
            report["pins_file"]: pins_sha256,
            ".dockerignore": hashlib.sha256(dockerignore.encode()).hexdigest(),
        }
    )
    context_files = {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}
    if context_files != expected_files or any(
        (output / row["file"]).stat().st_size != row["bytes"] for row in selected
    ):
        raise ValueError("Context changed during preparation; refusing manifest")
    # The manifest cannot hash itself; its hash is returned in the CLI report.
    report.update(
        {
            "context": str(output),
            "context_files": context_files,
        }
    )
    manifest = output / "context-manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    report["context_manifest_sha256"] = sha(manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"TailRing source context refused: {error}")
