#!/usr/bin/env python3
"""Fetch formal CANN by a readable official image tag; internally verify locked ARM64 content."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.parse
import urllib.request

SHA = '6068329c27a345701d5cb7c1f76b7264f2c3320df9cfe2f8673ff37186f45c8f'
SIZE = 4806522202
IMAGE = 'quay.io/ascend/cann:9.2.0-950-openeuler24.03-py3.12-devel'
TAG = IMAGE.split(':', 1)[1]
ARM64_MANIFEST = 'sha256:811ca9bb37475ba1a8db6cd1f52731651c13f5860a4cfb58773a32e603e02f57'
FILENAME = 'cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz'
MEDIA_TYPES = ', '.join([
    'application/vnd.oci.image.index.v1+json',
    'application/vnd.docker.distribution.manifest.list.v2+json',
    'application/vnd.oci.image.manifest.v1+json',
    'application/vnd.docker.distribution.manifest.v2+json',
])


def get_manifest(reference, token):
    request = urllib.request.Request(
        'https://quay.io/v2/ascend/cann/manifests/' + reference,
        headers={'Authorization': 'Bearer ' + token, 'Accept': MEDIA_TYPES})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    return json.loads(raw), 'sha256:' + hashlib.sha256(raw).hexdigest()


def resolve_image():
    """Start with the named official tag, then refuse a changed ARM64 manifest/layer."""
    token_url = 'https://quay.io/v2/auth?' + urllib.parse.urlencode({
        'service': 'quay.io', 'scope': 'repository:ascend/cann:pull'})
    with urllib.request.urlopen(token_url, timeout=60) as response:
        auth = json.load(response)
    token = auth.get('token') or auth.get('access_token')
    if not token:
        raise ValueError('Registry did not return a scoped pull token')
    index, index_digest = get_manifest(TAG, token)
    candidates = [item for item in index.get('manifests', [])
                  if item.get('platform', {}).get('architecture') == 'arm64'
                  and item.get('platform', {}).get('os') == 'linux']
    if len(candidates) != 1 or candidates[0]['digest'] != ARM64_MANIFEST:
        raise ValueError('Named CANN release tag no longer matches the locked ARM64 manifest')
    manifest, actual = get_manifest(ARM64_MANIFEST, token)
    if actual != ARM64_MANIFEST:
        raise ValueError('ARM64 manifest SHA256 mismatch')
    layers = [item for item in manifest.get('layers', [])
              if item.get('digest') == 'sha256:' + SHA and item.get('size') == SIZE]
    if len(layers) != 1:
        raise ValueError('Named CANN release does not contain the locked Ascend payload')
    return token, {'image': IMAGE, 'platform': 'linux/arm64', 'tag_index_digest': index_digest,
                   'arm64_manifest': ARM64_MANIFEST, 'filename': FILENAME,
                   'payload_sha256': SHA, 'payload_bytes': SIZE}


def file_digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def valid_lfs_pointer(path):
    if path.is_file() and path.stat().st_size < 1024:
        raw = path.read_bytes()
        if not raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            return False
        text = raw.decode()
        return (text.startswith('version https://git-lfs.github.com/spec/v1\n')
                and f'oid sha256:{SHA}\n' in text and f'size {SIZE}\n' in text)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_cache = Path(__file__).resolve().parents[1] / 'cache/official-v030rc1-cann9.2.0-317c7fb99e'
    parser.add_argument('--image', default=IMAGE, choices=[IMAGE],
                        help='Version-readable official CANN image (locked to the session release)')
    parser.add_argument('--output', type=Path,
                        default=default_cache / 'cann92-release' / FILENAME)
    parser.add_argument('--resolve-only', action='store_true',
                        help='Verify the named tag and ARM64 manifest; do not download the 4.8GB layer')
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if args.resolve_only:
        _, identity = resolve_image()
        print(json.dumps(identity, indent=2))
        return
    if output.is_symlink():
        sys.exit('Refusing symlink output: ' + str(output))
    pointer = valid_lfs_pointer(output)
    if output.exists() and not pointer:
        if output.stat().st_size == SIZE and file_digest(output) == SHA:
            print(json.dumps({'image': IMAGE, 'file': str(output), 'cache_hit': True,
                              'bytes': SIZE, 'sha256': SHA}, indent=2))
            return
        sys.exit('Refusing to overwrite invalid existing cache: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < SIZE + 64 * 1024 * 1024:
        sys.exit('Insufficient disk space for 4.8GB formal CANN layer')
    token, identity = resolve_image()
    request = urllib.request.Request(
        'https://quay.io/v2/ascend/cann/blobs/sha256:' + SHA,
        headers={'Authorization': 'Bearer ' + token})
    temporary = output.with_name(output.name + '.part')
    sha, count = hashlib.sha256(), 0
    # Retain a failed .part for diagnosis; never publish it as a build input.
    with temporary.open('xb') as target, urllib.request.urlopen(request, timeout=120) as response:
        if urllib.parse.urlparse(response.url).scheme != 'https':
            sys.exit('Refusing non-TLS blob redirect')
        while block := response.read(8 * 1024 * 1024):
            target.write(block)
            sha.update(block)
            count += len(block)
    if count != SIZE or sha.hexdigest() != SHA:
        sys.exit(f'Payload verification failed: bytes={count}, sha256={sha.hexdigest()}')
    if pointer:
        if not valid_lfs_pointer(output):
            sys.exit('LFS pointer changed during download; refusing replacement')
        os.replace(temporary, output)
    else:
        # Exclusive publication: do not overwrite an input created by another process.
        os.link(temporary, output)
        temporary.unlink()
    print(json.dumps({**identity, 'file': str(output), 'cache_hit': False}, indent=2))


if __name__ == '__main__':
    main()
