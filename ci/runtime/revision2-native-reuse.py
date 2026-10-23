"""Install committed Python-only revision while preserving verified native bytes."""

import hashlib
import importlib.metadata as md
import json
import shutil
from pathlib import Path

source = Path("/opt/upgrade/src/vllm-ascend")
installed = Path(md.distribution("vllm-ascend").locate_file("vllm_ascend"))
revision = json.loads((source / "SOURCE-IDENTITY.json").read_text())
if revision["commit"] != "143f3649d8181edae635091c5fe809eb51ff992c":
    raise RuntimeError("Unexpected Python revision")
native_before = {
    str(path.relative_to(installed)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in installed.rglob("*")
    if path.is_file() and (path.suffix == ".so" or "_cann_ops_custom" in path.parts)
}
allowed = [
    "worker/v2/spec_decode/multi_module_mtp/speculator.py",
]
for relative in allowed:
    shutil.copyfile(source / "vllm_ascend" / relative, installed / relative)
native_after = {
    str(path.relative_to(installed)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in installed.rglob("*")
    if path.is_file() and (path.suffix == ".so" or "_cann_ops_custom" in path.parts)
}
if native_before != native_after:
    raise RuntimeError("Native files changed during Python-only install")
ledger = {
    "source_commit": revision["commit"],
    "native_build_commit": "624cc9d19a74dcb8c685c4c7c5b3b4829dbfbab3",
    "native_compatible_diff": "Only MTP Python implementation and tests differ",
    "installed_python_files": {
        relative: hashlib.sha256((installed / relative).read_bytes()).hexdigest() for relative in allowed
    },
    "native_file_sha256": native_after,
    "distribution_version": md.version("vllm-ascend"),
    "distribution_version_note": "Preserved compiled wheel metadata; source revision is explicitly recorded separately",
}
Path("/opt/upgrade/evidence/mtp-builder-revision2.json").write_text(json.dumps(ledger, indent=2) + "\n")
print("PYTHON_REVISION_INSTALLED_NATIVE_BYTES_UNCHANGED", len(native_after))
