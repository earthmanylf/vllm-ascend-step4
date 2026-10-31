#!/usr/bin/env python3
"""Prepare the recipe and its locked inputs, archiving vllm-ascend from the current HEAD.

Everything except vllm-ascend.tar.gz stays checksum-locked: the CANN payload, the
vLLM wheel source, the Catlass/ops assets and the uv wheel are all fixed. The
vllm-ascend tree is generated from a git ref (default HEAD) so a build always
carries the source that is checked out, and the derived version travels with the
context instead of being hard-coded in the Dockerfile.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_ascend_version import git, release_line, version_for  # noqa: E402


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


# Inputs produced at build time instead of being pinned as archives: vllm-ascend
# is exported from the current ref, and vllm is cloned from git by the Dockerfile.
# Their rows stay in the pin only to record the last snapshot.
SOURCE_INPUT = 'vllm-ascend.tar.gz'
GENERATED_INPUTS = ('vllm-ascend.tar.gz', 'vllm.tar.gz')
# Inputs the base image can provide instead. Only consulted with
# --reuse-base-assets; otherwise they are required as before.
CANN_BASE_PROVIDED_INPUTS = (
    'cann92-release/cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz',
    'uv-0.9.5-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl',
)


def archive_ref(repo, ref, out):
    """Export `ref` as a tar, the way the session build did.

    A temporary bare repo is used so the delivered checkout is never touched, and
    export attributes are overridden so the archive carries the whole tree.
    """
    commit = git(repo, 'rev-parse', f'{ref}^{{commit}}').strip()
    with tempfile.TemporaryDirectory(prefix='vllm-ascend-archive-') as temporary:
        bare = Path(temporary) / 'source.git'
        subprocess.run(['git', 'init', '--bare', '-q', str(bare)], check=True)
        subprocess.run(['git', '-C', str(bare), 'fetch', '--no-tags', str(repo),
                        '+' + commit + ':refs/heads/ci-build'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(bare), 'symbolic-ref', 'HEAD', 'refs/heads/ci-build'],
                       check=True)
        (bare / 'info' / 'attributes').write_text('* -export-ignore -export-subst\n')
        with Path(out).open('xb') as stream:
            subprocess.run(['git', '-C', str(bare), 'archive', '--format=tar', 'HEAD'],
                           stdout=stream, check=True)
    # Describe from the real checkout: the temporary repo is fetched with
    # --no-tags, so only the original knows the tags. `--dirty` cannot take a
    # commit-ish, so the working-tree state is appended separately, and only
    # when the ref actually is HEAD.
    described = subprocess.run(['git', '-C', str(repo), 'describe', '--tags', '--long', commit],
                               capture_output=True, text=True)
    describe = described.stdout.strip() if described.returncode == 0 else ''
    if describe:
        head = git(repo, 'rev-parse', 'HEAD').strip()
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'],
                               capture_output=True, text=True).stdout.strip()
        if commit == head and dirty:
            describe += '-dirty'
    return commit, describe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ci-dir', required=True, type=Path)
    parser.add_argument('--inputs-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--source-ref', default='HEAD',
                        help='git ref for vllm-ascend.tar.gz (default: HEAD)')
    parser.add_argument('--with-union-ops', action='store_true',
                        help='require the union vendor trees and wheel under '
                             'ci/third-party/prebuilt/ and stage them into the context')
    parser.add_argument('--reuse-base-assets', action='store_true',
                        help='the base image already carries CANN 9.2.0, the vendored '
                             'third-party sources and the compiled custom ops; stage '
                             'empty archives so those steps are skipped instead of '
                             'requiring the 4.8 GB payload and three LFS inputs')
    parser.add_argument('--vllm-base-version', default=None,
                        help='vllm series to anchor the image version on '
                             '(default: pin.vllm_base_version)')
    parser.add_argument('--vllm-commit', default=None,
                        help='vllm commit the build clones (default: pin.vllm_source.commit)')
    parser.add_argument('--base-version', default=None,
                        help='vllm-ascend series to anchor the version on '
                             '(default: vllm_ascend_base_version from the pin)')
    args = parser.parse_args()
    need(args.check_only or args.output is not None, '--output or --check-only required')
    need(not args.reuse_base_assets, '--reuse-base-assets is invalid for the CANN-only recipe; native sources and ops must be built')
    ci = args.ci_dir.resolve(strict=True)
    inputs = args.inputs_dir.resolve(strict=True)
    lock = json.loads((ci / 'pins/official-v030-cann92.json').read_text())
    all_inputs = lock['inputs'] + lock.get('build_wheels', [])
    recipe = ci / 'docker/Dockerfile'
    need(digest(recipe) == lock['recipe_sha256'], 'Dockerfile differs from locked recipe (version-readable CANN filename)')
    # Dockerfile.fast is the python-overlay recipe: a different recipe on purpose,
    # so it is not compared against the native recipe hash.
    runtime = ci / 'runtime/official-v030-cann92'
    for name, expected in lock['runtime_sha256'].items():
        need(digest(runtime / name) == expected, f'Runtime helper hash mismatch: {name}')
    missing = []
    lfs_pointers = []
    checked = []
    for row in all_inputs:
        if row['file'] in GENERATED_INPUTS or row['file'] in CANN_BASE_PROVIDED_INPUTS:
            continue
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
              'source_ref': args.source_ref, 'source_commit': '',
              'inputs_dir': str(inputs), 'verified_inputs': checked, 'missing_inputs': missing,
              'unhydrated_lfs_inputs': lfs_pointers,
              'reuse_base_assets': False, 'cann_supplied_by_base': True,
              'all_inputs_available': not missing, 'device_or_build_validation': 'not run'}
    if args.check_only:
        print(json.dumps(report, indent=2))
        return
    need(not missing, 'Missing inputs: ' + ', '.join(missing) +
         '; git lfs pull obtains the required source archives; CANN itself is supplied by the base image')
    out = args.output.expanduser().absolute()
    need(out == out.resolve(), 'Context path must be canonical (no symlink parents)')
    repo, delivery = ci.parent, ci.parent.parent
    for protected in (repo, delivery, inputs):
        need(out != protected and protected not in out.parents and out not in protected.parents,
             f'Context must be outside repository/delivery/input root: {protected}')
    need(not out.exists() and not out.is_symlink(), f'Context already exists: {out}')
    need(out.parent.is_dir(), f'Context parent does not exist: {out.parent}')
    # Inputs are complete, so the build will proceed: it now needs a git
    # checkout to export the source, and a clean one so the image can be traced.
    repo = ci.parent
    try:
        source_commit = git(repo, 'rev-parse', f'{args.source_ref}^{{commit}}').strip()
    except subprocess.CalledProcessError:
        raise ValueError(
            f'Cannot resolve source ref {args.source_ref} in {repo}; '
            'the build needs a git checkout to export vllm-ascend')
    if args.source_ref in ('HEAD', '@') and subprocess.run(
            ['git', '-C', str(repo), 'status', '--porcelain'],
            capture_output=True, text=True).stdout.strip():
        raise ValueError(
            'Working tree is dirty; commit or stash before building so the image '
            'version can be traced to a commit')
    total = sum(row['bytes'] for row in checked)
    need(shutil.disk_usage(out.parent).free >= total + 64 * 1024 * 1024,
         f'Insufficient context disk space; need {total + 64 * 1024 * 1024} bytes')
    out.mkdir()
    for row in all_inputs:
        if row['file'] in GENERATED_INPUTS or row['file'] in CANN_BASE_PROVIDED_INPUTS:
            continue
        destination = out / row['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / row['file'], destination)
        need(digest(destination) == row['sha256'], f'Input changed during copy: {row["file"]}')

    # vllm-ascend: export the requested ref, then derive the version the image
    # should report. Both travel with the context so the Dockerfile stays generic.
    source_tar = out / SOURCE_INPUT
    commit, describe = archive_ref(repo, args.source_ref, source_tar)
    need(commit == source_commit, 'Source commit changed while archiving')
    base_version = args.base_version or lock.get('vllm_ascend_base_version')
    need(bool(base_version), 'No vllm_ascend_base_version in the pin; pass --base-version')
    version, derivation = version_for(describe, base_version, commit)
    need(release_line(version) == release_line(base_version),
         f'Derived version {version} left the {base_version} series')
    # vllm carries its own version into the image. Its fork's tags stop at
    # v0.26.1rc0, so an unanchored derivation labels this 0.30 image as 0.26.x;
    # anchor on the pinned series the same way vllm-ascend does. The commit is
    # the one the build clones, so the label always names what is inside.
    vllm_base_version = args.vllm_base_version or lock.get('vllm_base_version')
    need(bool(vllm_base_version), 'No vllm_base_version in the pin; pass --vllm-base-version')
    vllm_commit = args.vllm_commit or (lock.get('vllm_source') or {}).get('commit') or ''
    need(bool(vllm_commit), 'No vllm commit in the pin; pass --vllm-commit')
    vllm_version, vllm_derivation = version_for('', vllm_base_version, vllm_commit)
    need(release_line(vllm_version) == release_line(vllm_base_version),
         f'Derived vllm version {vllm_version} left the {vllm_base_version} series')
    report['vllm_version'] = vllm_version
    report['vllm_version_derivation'] = vllm_derivation
    (out / 'source-version.json').write_text(json.dumps({
        'ref': args.source_ref,
        'commit': commit,
        'describe': describe,
        'version': version,
        'base_version': base_version,
        'derivation': derivation,
        'dirty': describe.endswith('-dirty'),
        'input_sha256': digest(source_tar),
        'input_bytes': source_tar.stat().st_size,
        'vllm_version': vllm_version,
        'vllm_base_version': vllm_base_version,
        'vllm_commit': vllm_commit,
    }, indent=2) + '\n')
    need(not describe.endswith('-dirty'),
         f'Ref {args.source_ref} resolved to a dirty tree ({describe})')
    report.update({'source_commit': commit, 'source_describe': describe,
                   'source_version': version, 'source_version_base': base_version,
                   'source_version_derivation': derivation,
                   'generated_inputs': {SOURCE_INPUT: digest(source_tar)}})
    if args.with_union_ops:
        prebuilt = ci / 'third-party' / 'prebuilt'
        need(prebuilt.is_dir(), '--with-union-ops needs ci/third-party/prebuilt')
        # A vendor tree is only usable if op_api/lib survived the flattening;
        # a nested vendors/<name>/vendors/<name> would install nothing.
        for vendor in ('union_test_transformer', 'custom_nn_nn'):
            need((prebuilt / vendor / 'op_api' / 'lib').is_dir(),
                 f'{vendor} must be flattened: expected '
                 f'{prebuilt / vendor / "op_api" / "lib"}')
        shutil.copytree(prebuilt, out / 'third-party' / 'prebuilt')
        staged = sorted(p.name for p in (out / 'third-party' / 'prebuilt').iterdir())
        report['union_ops_staged'] = staged
        print('union ops staged: ' + ', '.join(staged))
    else:
        # Docker COPY is evaluated even when the later installer is disabled.
        # Keep its optional input present without shipping any union vendor.
        prebuilt = out / 'third-party' / 'prebuilt'
        prebuilt.mkdir(parents=True, exist_ok=True)
        (prebuilt / '.keep').write_text('Union vendors disabled for this build.\n')
        report['union_ops_staged'] = []

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
