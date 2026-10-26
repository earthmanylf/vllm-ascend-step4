#!/usr/bin/env python3
"""Copy already-available, verified session archives into the versioned Git LFS cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check(path, row):
    if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
        raise ValueError(f'Cache input size/SHA256 mismatch: {path}')


def is_pointer(path, row):
    if path.is_file() and path.stat().st_size < 1024:
        raw = path.read_bytes()
        if not raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            return False
        text = raw.decode()
        return (text.startswith('version https://git-lfs.github.com/spec/v1\n')
                and f"oid sha256:{row['sha256']}\n" in text
                and f"size {row['bytes']}\n" in text)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path,
                        help='Already-downloaded session archive root (read-only)')
    parser.add_argument('--ci-dir', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    ci = args.ci_dir.resolve(strict=True)
    source = args.source.resolve(strict=True)
    lock = json.loads((ci / 'pins/official-v030-cann92.json').read_text())
    cache = ci / 'cache' / lock['cache_profile']
    cache.mkdir(parents=True, exist_ok=True)
    # Never seed through a symlink into another tree.
    if not cache.resolve().is_relative_to(ci / 'cache'):
        raise ValueError('Cache profile directory escapes ci/cache')
    rows, missing = [], []
    for row in lock['inputs']:
        target = cache / row['file']
        if target.is_symlink():
            raise ValueError(f'Refusing cache symlink: {target}')
        pointer = is_pointer(target, row)
        if target.exists() and not pointer:
            check(target, row)
            rows.append({'file': row['file'], 'action': 'cache-hit', 'bytes': row['bytes']})
            continue
        candidates = [source / row['file']]
        if row.get('legacy_file'):
            candidates.append(source / row['legacy_file'])
        available = next((p for p in candidates if p.is_file()), None)
        if available is None:
            missing.append(row['file'])
            continue
        if not available.resolve().is_relative_to(source):
            raise ValueError(f'Source archive escapes supplied root: {available}')
        check(available, row)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(cache):
            raise ValueError(f'Cache archive parent escapes profile: {target}')
        if shutil.disk_usage(target.parent).free < row['bytes'] + 64 * 1024 * 1024:
            raise ValueError(f'Insufficient disk space: {target}')
        temporary = target.with_name(target.name + '.tmp')
        with temporary.open('xb') as destination, available.open('rb') as stream:
            shutil.copyfileobj(stream, destination, 8 * 1024 * 1024)
        check(temporary, row)
        if pointer:
            if not is_pointer(target, row):
                raise ValueError(f'LFS pointer changed during cache seed: {target}')
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        rows.append({'file': row['file'], 'action': 'copied', 'bytes': row['bytes']})
    report = {'cache': str(cache), 'cached': rows, 'missing': missing,
              'all_inputs_available': not missing, 'git_commit_or_lfs_upload': 'not performed'}
    (ci / 'cache/cache-status.local.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f'Shared cache refused: {error}')
