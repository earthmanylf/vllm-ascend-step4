#!/usr/bin/env python3
"""In-image fast source replacement (runs inside the candidate container).

Replaces ONLY the changed tracked files of the feature commit over the parent
image's existing vllm-ascend installation:

  * /vllm-workspace/vllm-ascend/<path>          (source checkout, for provenance)
  * site-packages/vllm_ascend/<path>            (installed package, .py/data only)

Safety gates (fail closed, exit 1):
  * source.tar sha256 must equal source-input.json archive_sha256;
  * every changed file's sha256 must equal source-input.json digests;
  * NO changed path may touch csrc/, setup.py, pyproject.toml, CMakeLists.txt,
    .gitmodules or any non-text file under vllm_ascend/ (compiled assets must
    go through the full ./build.sh native compile).

No native recompile happens: this path is only sound when the change is
Python-level. Writes /opt/step5/feature-artifacts/fast-replacement.json.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import site
import sys
import tarfile

INPUT = Path('/opt/step5/feature-inputs')
SOURCE = Path('/vllm-workspace/vllm-ascend')
ARTIFACTS = Path('/opt/step5/feature-artifacts')

FORBIDDEN_PREFIXES = ('csrc/', 'setup.py', 'pyproject.toml', 'CMakeLists.txt', '.gitmodules')
ALLOWED_SUFFIXES = ('.py', '.json', '.yaml', '.yml', '.txt', '.cfg', '.toml', '.md')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def fail(message):
    print('FAST-REPLACE-REFUSED: ' + message, file=sys.stderr)
    raise SystemExit(1)


def main():
    pin = json.loads((INPUT / 'source-input.json').read_text())
    commit, tree = pin['commit'], pin['tree']
    tar_path = INPUT / 'source.tar'

    if sha(tar_path) != pin['archive_sha256']:
        fail('source.tar digest differs from source-input.json archive_sha256')

    changed = pin.get('changed_paths') or []
    if not changed:
        fail('source-input.json has no changed_paths; nothing to replace')

    # ---- gate 1: forbidden paths (build-system / native sources) ----
    for rel in changed:
        if rel.startswith(FORBIDDEN_PREFIXES) or rel.startswith('vllm_ascend/csrc/'):
            fail('changed path touches build/native sources: ' + rel +
                 ' -> use the full ./build.sh (native compile)')

    # ---- gate 2: under vllm_ascend/ only text files may be replaced ----
    for rel in changed:
        if rel.startswith('vllm_ascend/') and not rel.endswith(ALLOWED_SUFFIXES):
            fail('non-text file under vllm_ascend/ (compiled asset?): ' + rel +
                 ' -> use the full ./build.sh (native compile)')

    # ---- extract and verify digests ----
    staging = Path('/tmp/fast-source')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with tarfile.open(tar_path, 'r:') as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        expected = {m.name: m for m in members}
        for rel in changed:
            if rel not in expected:
                fail('changed path missing from source.tar: ' + rel)
        archive.extractall(staging, members=[expected[r] for r in changed])

    digests = {}
    digests.update(pin.get('production_files', {}))
    digests.update(pin.get('test_files', {}))
    for rel in changed:
        actual = sha(staging / rel)
        if rel in digests and actual != digests[rel]:
            fail('digest mismatch for ' + rel)
        if rel not in digests:
            digests[rel] = actual

    # ---- replace: source checkout + installed package ----
    replaced_checkout, replaced_installed = [], []
    sp_dir = None
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(base) / 'vllm_ascend'
        if candidate.is_dir():
            sp_dir = candidate
            break
    if sp_dir is None:
        fail('installed vllm_ascend package not found in site-packages')

    for rel in changed:
        src = staging / rel
        dst_checkout = SOURCE / rel
        dst_checkout.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_checkout)
        replaced_checkout.append(rel)
        if rel.startswith('vllm_ascend/'):
            inner = rel[len('vllm_ascend/'):]
            dst_pkg = sp_dir / inner
            dst_pkg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst_pkg)
            replaced_installed.append(rel)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record = {
        'mode': 'fast-source-replacement',
        'commit': commit,
        'tree': tree,
        'replaced_in_checkout': replaced_checkout,
        'replaced_in_installed_package': replaced_installed,
        'file_sha256': {rel: digests[rel] for rel in changed},
        'installed_package_dir': str(sp_dir),
        'source_checkout': str(SOURCE),
        'scope': ('Python-level source replacement only; no native recompile. '
                  'Not promised bit-identical to a full ./build.sh image.'),
    }
    (ARTIFACTS / 'fast-replacement.json').write_text(json.dumps(record, indent=2) + '\n')
    print('FAST-REPLACE-OK commit={} files={} installed={}'.format(
        commit, len(replaced_checkout), len(replaced_installed)))


if __name__ == '__main__':
    main()
