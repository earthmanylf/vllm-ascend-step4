#!/usr/bin/env python3
"""Prepare a python-overlay build context from the current HEAD.

Produces a context containing the overlay Dockerfile, a `git archive` of
vllm_ascend at the requested ref, the sync helper, and the identity the image
should report. Refuses to continue when the result could not be trusted:

  * uncommitted changes (the image must be traceable to a commit)
  * csrc/ changed relative to the base image's recorded source commit, because
    the base image's compiled extension would then be stale
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vllm_ascend_version import git, version_for  # noqa: E402

# Label the native recipe stamps on every image it builds.
SOURCE_LABEL = 'org.stepfun.step5.source_base'


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_label(docker: str, image: str, label: str) -> str:
    out = subprocess.run(
        [docker, 'image', 'inspect', image, '--format',
         '{{index .Config.Labels "%s"}}' % label],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f'Cannot inspect base image {image}: {out.stderr.strip()}')
    value = out.stdout.strip()
    return '' if value == '<no value>' else value


def archive_ref(repo: pathlib.Path, ref: str, out: pathlib.Path) -> str:
    """Export `ref` (vllm_ascend subtree only) as a tar; returns the commit."""
    commit = git(repo, 'rev-parse', f'{ref}^{{commit}}').strip()
    with tempfile.TemporaryDirectory(prefix='vllm-ascend-overlay-') as temporary:
        bare = pathlib.Path(temporary) / 'source.git'
        subprocess.run(['git', 'init', '--bare', '-q', str(bare)], check=True)
        subprocess.run(['git', '-C', str(bare), 'fetch', '--no-tags', str(repo),
                        '+' + commit + ':refs/heads/ci-build'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(bare), 'symbolic-ref', 'HEAD', 'refs/heads/ci-build'],
                       check=True)
        # Tracked files only, exactly as committed; export attributes ignored.
        (bare / 'info' / 'attributes').write_text('* -export-ignore -export-subst\n')
        with out.open('xb') as stream:
            subprocess.run(['git', '-C', str(bare), 'archive', '--format=tar',
                            'HEAD', 'vllm_ascend'], stdout=stream, check=True)
    return commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ci-dir', required=True, type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--source-ref', default='HEAD')
    parser.add_argument('--base-image', required=True)
    parser.add_argument('--docker', default='docker')
    parser.add_argument('--base-version', default=None)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    ci = args.ci_dir.resolve(strict=True)
    repo = ci.parent
    lock = json.loads((ci / 'pins/official-v030-cann92.json').read_text())
    base_version = args.base_version or lock.get('vllm_ascend_base_version')
    if not base_version:
        raise SystemExit('No vllm_ascend_base_version in the pin; pass --base-version')

    commit = git(repo, 'rev-parse', f'{args.source_ref}^{{commit}}').strip()
    dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'],
                           capture_output=True, text=True).stdout.strip()
    if dirty and args.source_ref in ('HEAD', '@'):
        raise SystemExit('Working tree is dirty; commit or stash first so the image '
                         'can be traced to a commit')

    # Refuse an overlay image as the base: labels are inherited, so it would
    # otherwise look like a valid native base and stack builds on each other.
    base_mode = image_label(args.docker, args.base_image, 'step5.build.mode')
    if base_mode == 'python-overlay':
        raise SystemExit(
            f'{args.base_image} is itself a python-overlay build. Layering overlays '
            'accumulates defects from earlier builds; point --base-image at a native '
            'image, or use --full.')

    base_commit = image_label(args.docker, args.base_image, SOURCE_LABEL)
    if not base_commit:
        raise SystemExit(
            f'Base image {args.base_image} carries no {SOURCE_LABEL} label, so the '
            'compiled extension it ships cannot be tied to a commit. Pick an image '
            'built by this CI, or use --full for a native rebuild.')
    try:
        git(repo, 'cat-file', '-e', f'{base_commit}^{{commit}}')
    except subprocess.CalledProcessError:
        raise SystemExit(f'Base image source commit {base_commit} is not in this '
                         'repository; cannot verify the compiled extension')

    describe = subprocess.run(
        ['git', '-C', str(repo), 'describe', '--tags', '--long', commit],
        capture_output=True, text=True).stdout.strip()
    version, derivation = version_for(describe, base_version)

    csrc_changed = git(repo, 'diff', '--name-only', base_commit, commit, '--', 'csrc').strip()
    report = {
        'build_mode': 'python-overlay',
        'source_ref': args.source_ref,
        'source_commit': commit,
        'source_describe': describe,
        'version': version,
        'base_version': base_version,
        'version_derivation': derivation,
        'base_image': args.base_image,
        'base_source_commit': base_commit,
        'csrc_changed_since_base': csrc_changed.split('\n') if csrc_changed else [],
    }
    if args.check_only:
        print(json.dumps(report, indent=2))
        return 0 if not csrc_changed else 1
    if csrc_changed:
        raise SystemExit(
            'csrc/ differs between the base image and this commit:\n  '
            + csrc_changed.replace('\n', '\n  ')
            + '\nThe base image ships a compiled vllm_ascend_C that would no longer '
              'match. Use --full for a native rebuild.')

    out = args.output.expanduser().absolute()
    if out.exists():
        raise SystemExit(f'Context already exists: {out}')
    out.mkdir(parents=True)

    archive = out / 'vllm_ascend-src.tar'
    got = archive_ref(repo, args.source_ref, archive)
    if got != commit:
        raise SystemExit('Source commit changed while archiving')

    # Modules present in the base image's source but gone at this commit. Derived
    # from git rather than by scanning the installed tree, so build-generated
    # modules (_version.py, _build_info.py) and vendored custom ops survive.
    deleted = git(repo, 'diff', '--diff-filter=D', '--name-only',
                  base_commit, commit, '--', 'vllm_ascend').strip()
    relative_deleted = [
        line[len('vllm_ascend/'):] for line in deleted.split('\n')
        if line.startswith('vllm_ascend/')
    ]
    (out / 'deleted-modules.txt').write_text(
        '\n'.join(sorted(relative_deleted)) + ('\n' if relative_deleted else ''))
    shutil.copyfile(ci / 'scripts/sync_python_overlay.py', out / 'sync_python_overlay.py')
    shutil.copyfile(ci / 'docker/Dockerfile.fast', out / 'Dockerfile')
    (out / '.dockerignore').write_text('Dockerfile\nsource-version.json\n')

    report['modules_deleted_upstream'] = sorted(relative_deleted)
    report['context_files'] = {
        str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()
    }
    (out / 'source-version.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'source_commit', 'version', 'base_version', 'version_derivation',
        'base_source_commit', 'csrc_changed_since_base')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
