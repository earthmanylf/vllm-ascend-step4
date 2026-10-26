"""Repack a locked Python-only revision; preserve native provenance and payload."""

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import stat
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

OLD_VERSION = "0.19.1rc2.dev2976+g624cc9d19a"
VERSION = "0.19.1rc2.dev2977+g143f3649d8"
SOURCE_COMMIT = "143f3649d8181edae635091c5fe809eb51ff992c"
NATIVE_COMMIT = "624cc9d19a74dcb8c685c4c7c5b3b4829dbfbab3"
SOURCE_BRANCH = "step5-preview-tailring-npu"
MODULE = "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py"
MODULE_SHA256 = "51f3da4a032aeee7cf44b579df58165a0096ac4e85741c8938aacf7db67f084e"
NATIVE_FILE_COUNT = 1114
NATIVE_EXTENSION = "vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
TAG = "cp312-cp312-linux_aarch64"
OLD_INFO = f"vllm_ascend-{OLD_VERSION}.dist-info"
NEW_INFO = f"vllm_ascend-{VERSION}.dist-info"
AUDIT = "vllm_ascend/_native_revision_audit.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def locked_file(root, name):
    path = root / name
    need(path.resolve().is_relative_to(root), f"Source file escapes root: {name}")
    need(path.is_file(), f"Source file missing or not regular: {name}")
    return path.read_bytes()


def native_hashes(files):
    return {
        name: sha(data)
        for name, data in files.items()
        if name.endswith(".so") or name.startswith("vllm_ascend/_cann_ops_custom/")
    }


def verify_record(files, info):
    name = info + "/RECORD"
    need(name in files, "Wheel RECORD missing")
    rows = list(csv.reader(io.StringIO(files[name].decode("utf-8"))))
    recorded = set()
    for row in rows:
        need(len(row) == 3 and row[0] not in recorded, "Invalid or duplicate wheel RECORD row")
        member, digest, size = row
        recorded.add(member)
        need(member in files, f"RECORD names missing file: {member}")
        if member == name:
            need(digest == size == "", "RECORD must not hash itself")
        else:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(files[member]).digest()).rstrip(b"=").decode()
            need(
                digest == "sha256=" + encoded and size == str(len(files[member])),
                f"Wheel RECORD checksum/size mismatch: {member}",
            )
    need(recorded == set(files), "Wheel RECORD does not cover every file")


def read_wheel(data):
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        seen = set()
        for member in wheel.infolist():
            name = member.filename
            need(name not in seen, f"Duplicate wheel member: {name}")
            seen.add(name)
            parts = name.rstrip("/").split("/")
            need(
                "\\" not in name
                and all(part not in ("", ".", "..") for part in parts)
                and not any(ord(char) < 32 for char in name),
                f"Unsafe wheel member: {name!r}",
            )
            kind = stat.S_IFMT(member.external_attr >> 16)
            need(
                kind in (0, stat.S_IFREG, stat.S_IFDIR) and (kind != stat.S_IFDIR or member.is_dir()),
                f"Non-regular wheel member: {name}",
            )
            need(parts[0] in ("vllm_ascend", OLD_INFO), f"Unexpected wheel root: {name}")
            if member.is_dir():
                continue
            need(
                name not in (OLD_INFO + "/RECORD.jws", OLD_INFO + "/RECORD.p7s"),
                "Signed base wheel requires a separate signature policy",
            )
            files[name] = wheel.read(member)
    need(AUDIT not in files, "Base wheel already contains a revision audit")
    for required in (
        MODULE,
        "vllm_ascend/_version.py",
        NATIVE_EXTENSION,
        OLD_INFO + "/METADATA",
        OLD_INFO + "/WHEEL",
    ):
        need(required in files, f"Required base wheel file missing: {required}")
    verify_record(files, OLD_INFO)
    metadata = BytesParser().parsebytes(files[OLD_INFO + "/METADATA"])
    names = metadata.get_all("Name", [])
    need(
        len(names) == 1 and names[0] in ("vllm_ascend", "vllm-ascend"),
        "Unexpected wheel distribution name",
    )
    need(metadata.get_all("Version") == [OLD_VERSION], "Unexpected wheel metadata version")
    wheel_metadata = BytesParser().parsebytes(files[OLD_INFO + "/WHEEL"])
    need(wheel_metadata.get_all("Wheel-Version") == ["1.0"], "Unexpected wheel format version")
    need(wheel_metadata.get_all("Tag") == [TAG], "Unexpected wheel ABI/platform tag")
    need(wheel_metadata.get_all("Root-Is-Purelib") == ["false"], "Base wheel must not be pure Python")
    # Inspect only literal version assignments; never execute base wheel code.
    versions = []
    for node in ast.parse(files["vllm_ascend/_version.py"]).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("version", "__version__") for target in node.targets
        ):
            versions.append(ast.literal_eval(node.value))
    need(versions and all(value == OLD_VERSION for value in versions), "Unexpected base _version.py identity")
    native = native_hashes(files)
    need(len(native) == NATIVE_FILE_COUNT, f"Expected {NATIVE_FILE_COUNT} native files, got {len(native)}")
    return files, native


def repack(wheel_path, source, output, base_wheel_sha256):
    need(bool(re.fullmatch(r"[0-9a-f]{64}", base_wheel_sha256)), "Expected a lowercase base wheel SHA256 lock")
    need(wheel_path.name == f"vllm_ascend-{OLD_VERSION}-{TAG}.whl", "Unexpected base wheel filename/version/tag")
    need(wheel_path.is_file() and not wheel_path.is_symlink(), "Base wheel must be a regular non-symlink file")
    # Hash and parse the same byte snapshot, not two separate reads.
    base = wheel_path.read_bytes()
    need(sha(base) == base_wheel_sha256, "Base wheel SHA256 does not match lock")
    source = source.resolve(strict=True)
    need(source.is_dir(), "Source must be a directory")
    identity = json.loads(locked_file(source, "SOURCE-IDENTITY.json"))
    need(
        isinstance(identity, dict)
        and identity.get("commit") == SOURCE_COMMIT
        and identity.get("branch") == SOURCE_BRANCH,
        "Unexpected source revision/branch",
    )
    module = locked_file(source, MODULE)
    need(sha(module) == MODULE_SHA256, "MTP source module SHA256 does not match lock")
    files, native = read_wheel(base)
    files[MODULE] = module
    files["vllm_ascend/_version.py"] = (
        "# Generated for the audited Python-only revision.\n"
        f"__version__ = version = {VERSION!r}\n"
        "__version_tuple__ = version_tuple = (0, 19, 1, 'rc2', 'dev2977', 'g143f3649d8')\n"
        f"__commit_id__ = commit_id = {SOURCE_COMMIT!r}\n"
    ).encode()
    renamed = {}
    for name, data in files.items():
        if name == OLD_INFO + "/RECORD":
            continue
        destination = NEW_INFO + name[len(OLD_INFO) :] if name.startswith(OLD_INFO + "/") else name
        if name == OLD_INFO + "/METADATA":
            old_line = f"Version: {OLD_VERSION}\n".encode()
            need(data.count(old_line) == 1, "Unexpected wheel metadata version encoding")
            data = data.replace(old_line, f"Version: {VERSION}\n".encode(), 1)
        renamed[destination] = data
    need(native == native_hashes(renamed), "Native bytes changed while repacking")
    ledger = {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "native_source_commit": NATIVE_COMMIT,
        "native_compatibility": "git diff restricted to one MTP Python module and tests",
        "distribution_version": VERSION,
        "base_wheel_sha256": base_wheel_sha256,
        "changed_module": MODULE,
        "module_sha256": sha(module),
        "native_files": native,
        "native_file_count": len(native),
        "policy": "No native file, dependencies, entry points or model weights changed",
    }
    renamed[AUDIT] = json.dumps(ledger, indent=2).encode()
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(renamed.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((NEW_INFO + "/RECORD", "", ""))
    renamed[NEW_INFO + "/RECORD"] = record.getvalue().encode()
    verify_record(renamed, NEW_INFO)
    output = output.expanduser().absolute()
    need(output == output.resolve(), "Output directory must be canonical (no symlink parents)")
    need(not output.exists() or output.is_dir(), "Output must be a directory")
    need(
        not output.is_relative_to(source)
        and not source.is_relative_to(output)
        and not output.is_relative_to(wheel_path.resolve().parent),
        "Output must not overlap source or base wheel directory",
    )
    destination = output / f"vllm_ascend-{VERSION}-{TAG}.whl"
    audit_path = output / "ascend-wheel-revision-audit.json"
    need(not destination.exists() and not destination.is_symlink(), "Refusing to overwrite wheel")
    need(not audit_path.exists() and not audit_path.is_symlink(), "Refusing to overwrite revision audit")
    output.mkdir(parents=True, exist_ok=True)
    wheel_created = False
    audit_created = False
    try:
        with destination.open("xb") as stream:
            wheel_created = True
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
                for name, data in sorted(renamed.items()):
                    wheel.writestr(name, data)
        with zipfile.ZipFile(destination) as wheel:
            written = {name: wheel.read(name) for name in wheel.namelist()}
        need(written == renamed and native_hashes(written) == native, "Output wheel changed while writing")
        verify_record(written, NEW_INFO)
        ledger["wheel"] = destination.name
        ledger["wheel_sha256"] = sha(destination.read_bytes())
        with audit_path.open("x") as stream:
            audit_created = True
            stream.write(json.dumps(ledger, indent=2) + "\n")
    except Exception:
        if wheel_created:
            destination.unlink(missing_ok=True)
        if audit_created:
            audit_path.unlink(missing_ok=True)
        raise
    return destination, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-wheel-sha256", required=True, help="Pinned SHA256 of the audited 624 base wheel")
    args = parser.parse_args()
    destination, ledger = repack(args.wheel, args.source, args.output, args.base_wheel_sha256)
    print(destination, ledger["wheel_sha256"], "native files preserved:", ledger["native_file_count"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile) as error:
        sys.exit(f"Ascend revision repack refused: {error}")
