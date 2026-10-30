#!/usr/bin/env python3
"""Provision the third-party sources this CI needs, then report what is ready.

Everything heavy lives here but is not committed: the op repos are 300 MB+ each,
their build-time dependencies another 250 MB, and the CANN payload 4.8 GB.
Instead this directory holds a manifest pinning each source, and this script
materialises them.

Three ways to provide a source, in the order the script tries:

  skip    already present and every pinned ref resolves -> nothing to do
  copy    --from <dir> copies an existing checkout and verifies its refs, which
          is how an offline host gets these in (e.g. cloned on a laptop, then
          scp'd)
  clone   git clone from the pinned repo (needs network to gitcode.com)

The refs are verified either way. A directory missing a pinned ref is reported
rather than silently used: these ops decide the numerics of the build, and two
of them (opbase, ops-tensor) are checked out in place by the op repos' own cmake,
so "close enough" is not a thing here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / 'manifest.json'


def git(repo, *args, check=True):
    return subprocess.run(['git', '-C', str(repo), *args],
                          capture_output=True, text=True, check=check)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def has_ref(path: pathlib.Path, ref: str) -> bool:
    return git(path, 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}',
               check=False).returncode == 0


def missing_refs(path: pathlib.Path, refs) -> list:
    return [ref for ref in refs if not has_ref(path, ref)]


def submodule_refs(path: pathlib.Path, refs):
    """Gitlink commits each pinned ref records for its submodules.

    ops-tensor pins a different include/tensor_api commit per parent ref, so the
    submodule repo must carry all of them, not just the one checked out now.
    """
    out = {}
    for ref in refs:
        listed = git(path, 'ls-tree', '-r', ref, check=False)
        for line in listed.stdout.splitlines():
            meta, _, sub = line.partition('\t')
            if meta.startswith('160000'):
                out[(ref, sub)] = meta.split()[2]
    return out


def provision_git(entry, local_from, dry_run):
    name = entry['name']
    target = ROOT / name
    refs = entry.get('refs') or [entry['commit']]
    branch = entry.get('branch')

    if (target / '.git').exists():
        absent = missing_refs(target, refs)
        if not absent:
            return name, 'ready', f'{len(refs)} ref(s) present: {", ".join(refs)}'
        return name, 'wrong-refs', (
            f'present but missing {", ".join(absent)}; delete it and re-run, '
            'or pass --from with a checkout that carries them')

    if dry_run:
        where = f'clone {entry["repo"]}' + (f' @ {branch}' if branch else '')
        return name, 'would-provision', f'{where}, then need {", ".join(refs)}'

    source = (local_from or {}).get(name)
    if source:
        source = pathlib.Path(source)
        if not source.is_dir():
            return name, 'failed', f'--from {name} is not a directory: {source}'
        if not (source / '.git').exists():
            return name, 'failed', (
                f'--from {name} has no .git, so its refs cannot be verified; '
                'clone it instead of exporting a tarball')
        absent = missing_refs(source, refs)
        if absent:
            return name, 'failed', (
                f'--from {name} is missing {", ".join(absent)}')
        shutil.copytree(source, target, symlinks=True)
        return name, 'copied', f'{len(refs)} ref(s) verified from {source}'

    clone = ['git', 'clone', entry['repo'], str(target)]
    if branch:
        clone[2:2] = ['--branch', branch]
    done = subprocess.run(clone, capture_output=True, text=True)
    if done.returncode != 0:
        last = done.stderr.strip().splitlines()[-1] if done.stderr else ''
        return name, 'failed', (
            'clone failed (gitcode.com unreachable?); pass '
            f'--from {name}=/path/to/existing/checkout\n    {last}')
    absent = missing_refs(target, refs)
    if absent:
        # A pinned commit can be newer than the branch tip on a shallow mirror,
        # so try once more per ref before giving up.
        for ref in absent:
            subprocess.run(['git', '-C', str(target), 'fetch', 'origin', ref],
                           capture_output=True, text=True)
        absent = missing_refs(target, refs)
    if absent:
        return name, 'wrong-refs', f'cloned but missing {", ".join(absent)}'
    return name, 'cloned', f'{len(refs)} ref(s)'


def provision_payload(entry, local_from, dry_run):
    name = entry['name']
    target = ROOT / name / entry['file']
    pinned = entry['sha256']

    if target.is_file():
        size = target.stat().st_size
        if size == entry['bytes']:
            if sha256(target) != pinned:
                return name, 'corrupt', 'sha256 does not match the manifest'
            return name, 'ready', f'{size:,} bytes, sha256 verified'
        if size < 1024:
            return name, 'placeholder', (
                f'only {size} bytes: an LFS pointer, not the payload. '
                'Fetch with ci/scripts/fetch-cann92-payload.py, or pass '
                f'--from {name}=/path/to/{entry["file"]}')
        return name, 'wrong-size', f'{size:,} bytes, expected {entry["bytes"]:,}'

    if dry_run:
        return name, 'would-provision', (
            f'needs {entry["bytes"]:,} bytes from {entry["source_image"]}')

    source = (local_from or {}).get(name)
    if source:
        source = pathlib.Path(source)
        if source.is_dir():
            source = source / entry['file']
        if not source.is_file():
            return name, 'failed', f'--from {name} not found: {source}'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.stat().st_size == entry['bytes'] and sha256(target) == pinned:
            return name, 'copied', f'{target.stat().st_size:,} bytes verified'
        return name, 'corrupt', 'copied file does not match the pinned digest'

    return name, 'missing', (
        'not present. Run ci/scripts/fetch-cann92-payload.py, or pass '
        f'--from {name}=/path/to/{entry["file"]}. Not needed at all when '
        'building with --reuse-base-assets.')


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='local_from', action='append', default=[],
                        metavar='NAME=PATH',
                        help='copy this entry from a local path instead of cloning')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only', action='append', default=[],
                        help='provision just these entries')
    parser.add_argument('--submodules', action='store_true',
                        help='also verify every submodule commit the pinned refs '
                             'record (implies nothing is materialised, only checked)')
    args = parser.parse_args()

    local_from = {}
    for item in args.local_from:
        if '=' not in item:
            raise SystemExit(f'--from expects NAME=PATH, got {item!r}')
        key, value = item.split('=', 1)
        local_from[key] = value

    manifest = json.loads(MANIFEST.read_text())
    entries = [e for e in manifest['entries']
               if not args.only or e['name'] in args.only]

    rows, worst = [], 0
    for entry in entries:
        if entry['kind'] == 'git':
            name, state, detail = provision_git(entry, local_from, args.dry_run)
        else:
            name, state, detail = provision_payload(entry, local_from, args.dry_run)
        ok = state in ('ready', 'cloned', 'copied', 'would-provision')
        rows.append((name, state, detail, ok))
        worst = max(worst, 0 if ok else 1)

    if args.submodules:
        for entry in entries:
            if entry['kind'] != 'git' or not entry.get('submodules'):
                continue
            target = ROOT / entry['name']
            if not (target / '.git').exists():
                continue
            for (ref, sub), commit in submodule_refs(
                    target, entry.get('refs') or [entry['commit']]).items():
                subrepo = target / sub
                ok = (subrepo / '.git').exists() and has_ref(subrepo, commit)
                rows.append((f'{entry["name"]}:{sub}', 'submodule',
                             f'{ref} -> {commit[:8]} {"present" if ok else "MISSING"}',
                             ok))
                worst = max(worst, 0 if ok else 1)

    width = max(len(r[0]) for r in rows)
    for name, state, detail, ok in rows:
        print(f'{"OK  " if ok else "FAIL"} {name:<{width}}  {state:<17} {detail}')

    blocked = [r[0] for r in rows if not r[3]]
    if blocked:
        print()
        print('Blocked: ' + ', '.join(blocked))
        print('Nothing is committed here on purpose: these trees are large and the')
        print('CANN payload is 4.8 GB. Provision them, then re-run this script.')
    return worst


if __name__ == '__main__':
    raise SystemExit(main())
