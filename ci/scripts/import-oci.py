#!/usr/bin/env python3
"""Verify the sealed dual-image OCI before importing it into an existing containerd namespace.

Generalization of the delivery's build/import-oci.py:
  * delivery root and OCI path are arguments (works from any checkout);
  * the OCI is the UNIQUE *.oci.tar under <delivery>/images (deterministic selection);
  * expected size/sha256 are read from the delivery final-manifest.json
    (fallback: SHA256SUMS line for the same relative path);
  * prints the exact ctr command; only --apply executes it. Never starts a daemon.

Weakening vs the delivery original (documented trade-off): the original also gated on
final-manifest seal status and cross-checked the OCI against
reports/experiments/.../candidate-archive.json (image manifest/config digests). This CI
version verifies file-level sha256/bytes against the delivery seal only; the image-level
digest cross-check remains available in the delivery original.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def expected_from_manifest(delivery, oci):
    manifest = json.loads((delivery / 'final-manifest.json').read_text())
    rel = os.path.relpath(os.path.realpath(oci), os.path.realpath(delivery))
    for row in manifest['files']:
        if row['path'] == rel:
            return row['sha256'], row['bytes']
    raise SystemExit('OCI not listed in final-manifest.json: ' + rel)


def expected_from_sums(sums_file, oci):
    rel = os.path.relpath(os.path.realpath(oci), os.path.realpath(sums_file.parent))
    for line in sums_file.read_text().splitlines():
        digest, _, path = line.partition('  ')
        if path == rel:
            return digest, None
    raise SystemExit('OCI not listed in SHA256SUMS: ' + rel)


def locate_oci(delivery, explicit):
    if explicit is not None:
        oci = explicit.absolute()
        assert oci.is_file(), 'OCI file not found: ' + str(oci)
        return oci
    matches = sorted((delivery / 'images').glob('*.oci.tar'))
    assert len(matches) == 1, 'expected exactly one *.oci.tar under images/, found {}'.format(len(matches))
    return matches[0].absolute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--oci', type=Path, default=None,
                        help='default: the unique *.oci.tar under <delivery>/images')
    parser.add_argument('--sums', type=Path, default=None,
                        help='fallback checksum file (default: <delivery>/SHA256SUMS)')
    parser.add_argument('--ctr', type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--platform', default='linux/arm64')
    parser.add_argument('--sudo', action='store_true',
                        help='prefix the ctr command with sudo -n (e.g. 950DT build host)')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    delivery = args.delivery.absolute()
    oci = locate_oci(delivery, args.oci)
    assert oci == oci.resolve() and oci.is_file(), 'canonical OCI file required'
    # ctr 校验：--sudo 时由 root 侧解析（950DT 上 /opt/basemind 对普通用户不可 stat），
    # 否则要求绝对路径的可见可执行文件。
    assert args.ctr.is_absolute() and str(args.ctr), 'ctr must be an absolute path'
    if args.sudo:
        import subprocess
        subprocess.run(['sudo', '-n', 'test', '-f', str(args.ctr)], check=True)
    else:
        assert args.ctr.is_file(), 'ctr binary must be a visible regular file'
    assert args.namespace and not args.namespace.startswith('-'), 'bad namespace'

    try:
        expected_sha, expected_bytes = expected_from_manifest(delivery, oci)
    except (FileNotFoundError, SystemExit):
        sums = args.sums or (delivery / 'SHA256SUMS')
        expected_sha, expected_bytes = expected_from_sums(sums, oci)

    actual_sha = sha256(oci)
    actual_bytes = oci.stat().st_size
    assert actual_sha == expected_sha, 'OCI sha256 mismatch: {} != {}'.format(actual_sha, expected_sha)
    if expected_bytes is not None:
        assert actual_bytes == expected_bytes, 'OCI size mismatch: {} != {}'.format(actual_bytes, expected_bytes)

    command = [str(args.ctr), '-n', args.namespace, 'images', 'import', '--platform', args.platform, str(oci)]
    if args.sudo:
        command = ['sudo', '-n'] + command
    print(json.dumps({'pass': True, 'apply': args.apply, 'oci': str(oci),
                      'sha256': actual_sha, 'bytes': actual_bytes, 'command': command,
                      'scope': 'Only imports into the explicitly supplied existing runtime; never starts a daemon'},
                     indent=2), flush=True)
    if args.apply:
        import subprocess
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
