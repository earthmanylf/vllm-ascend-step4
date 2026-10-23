#!/usr/bin/env python3
"""Prepare the exact session recipe and checksum-locked archives; never archive live HEAD."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ci-dir', required=True, type=Path)
    parser.add_argument('--inputs-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    need(args.check_only or args.output is not None, '--output or --check-only required')
    ci = args.ci_dir.resolve(strict=True)
    inputs = args.inputs_dir.resolve(strict=True)
    lock = json.loads((ci / 'pins/official-v030-cann92.json').read_text())
    recipe = ci / 'docker/Dockerfile'
    need(digest(recipe) == lock['recipe_sha256'], 'Dockerfile differs from locked recipe (version-readable CANN filename)')
    for name in ('Dockerfile.fast', 'Dockerfile.official-v030-cann92'):
        need(digest(ci / 'docker' / name) == lock['recipe_sha256'], f'{name} recipe mismatch')
    runtime = ci / 'runtime/official-v030-cann92'
    for name, expected in lock['runtime_sha256'].items():
        need(digest(runtime / name) == expected, f'Runtime helper hash mismatch: {name}')
    missing = []
    lfs_pointers = []
    checked = []
    for row in lock['inputs']:
        path = inputs / row['file']
        if not path.is_file():
            missing.append(row['file'])
            continue
        resolved = path.resolve(strict=True)
        need(inputs in resolved.parents, f'Input escapes archive root: {path}')
        if path.stat().st_size < 1024 and path.read_bytes().startswith(
                b'version https://git-lfs.github.com/spec/v1\n'):
            text = path.read_text()
            need(f"oid sha256:{row['sha256']}\n" in text and f"size {row['bytes']}\n" in text,
                 f'LFS pointer does not match the input lock: {path}')
            missing.append(row['file'])
            lfs_pointers.append(row['file'])
            continue
        need(path.stat().st_size == row['bytes'], f'Input size mismatch: {path}')
        need(digest(path) == row['sha256'], f'Input SHA256 mismatch: {path}')
        checked.append(row)
    report = {'session_id': lock['session_id'], 'recipe_sha256': lock['recipe_sha256'],
              'inputs_dir': str(inputs), 'verified_inputs': checked, 'missing_inputs': missing,
              'unhydrated_lfs_inputs': lfs_pointers,
              'all_inputs_available': not missing, 'device_or_build_validation': 'not run'}
    if args.check_only:
        print(json.dumps(report, indent=2))
        return
    need(not missing, 'Missing inputs: ' + ', '.join(missing) +
         '; git lfs pull obtains shared cache objects; fetch-cann92-payload.py '
         'can obtain the formal CANN layer from its version-readable official image tag')
    out = args.output.expanduser().absolute()
    need(out == out.resolve(), 'Context path must be canonical (no symlink parents)')
    repo, delivery = ci.parent, ci.parent.parent
    for protected in (repo, delivery, inputs):
        need(out != protected and protected not in out.parents and out not in protected.parents,
             f'Context must be outside repository/delivery/input root: {protected}')
    need(not out.exists() and not out.is_symlink(), f'Context already exists: {out}')
    need(out.parent.is_dir(), f'Context parent does not exist: {out.parent}')
    total = sum(row['bytes'] for row in lock['inputs'])
    need(shutil.disk_usage(out.parent).free >= total + 64 * 1024 * 1024,
         f'Insufficient context disk space; need {total + 64 * 1024 * 1024} bytes')
    out.mkdir()
    for row in lock['inputs']:
        destination = out / row['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / row['file'], destination)
        need(digest(destination) == row['sha256'], f'Input changed during copy: {row["file"]}')
    shutil.copyfile(recipe, out / 'Dockerfile')
    shutil.copyfile(ci / 'docker/Dockerfile.dockerignore', out / '.dockerignore')
    for name, expected in lock['runtime_sha256'].items():
        shutil.copyfile(runtime / name, out / name)
        need(digest(out / name) == expected, f'Runtime helper changed during copy: {name}')
    need(digest(out / 'Dockerfile') == lock['recipe_sha256'], 'Recipe changed during copy')
    report.update({'context': str(out), 'context_files': {
        str(path.relative_to(out)): digest(path)
        for path in sorted(out.rglob('*')) if path.is_file()}})
    (out / 'context-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f'Official session context refused: {error}')
