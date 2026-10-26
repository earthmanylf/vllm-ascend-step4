#!/usr/bin/env python3
"""Materialize the six exact candidate build inputs from this repository.

Two modes:
  pin mode     --commit equals ci/pins/source-input.json commit:
                  byte-exact reproduction of the sealed r2 context. The Git bundle is
                  copied from the delivery (build/inputs/source.bundle); source.tar is
                  regenerated from this repo; all six digests must equal
                  ci/pins/recipe-origin.json context_files.
  commit mode  --commit is any other commit present in this repo:
                  source.bundle is created from the repo; source-input.json and
                  feature-source-record.json are regenerated for that commit. Actual
                  digests are recorded in the result JSON (no pin assertion).

The context is always created OUTSIDE this repository and outside the delivery root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

CONTEXT_NAMES = {'Dockerfile', 'candidate_probe.py', 'source-input.json',
                 'feature-source-record.json', 'source.bundle', 'source.tar'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def regular(path):
    path = Path(path)
    need(path == path.resolve() and path.is_file() and not path.is_symlink(),
         'Canonical regular file required: ' + str(path))
    return path


def git(repo, *args, **kwargs):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', '-C', str(repo), *args],
                                   env=env, **kwargs)


def rev_parse(repo, ref):
    return git(repo, 'rev-parse', ref, text=True).strip()


def prepare_bare(repo, commit, temporary):
    """One temporary bare repo: fetch the commit into a named ref and point HEAD at it.

    Needed because git bundle refuses raw SHAs as ref names ("Refusing to create
    empty bundle"); the delivery pin expects the bundle to carry bundle_ref=HEAD.
    Only the temporary bare repo is written; the delivered repo is never modified.
    """
    bare = Path(temporary) / 'source.git'
    subprocess.run(['git', 'init', '--bare', '-q', str(bare)], check=True)
    subprocess.run(['git', '-C', str(bare), 'fetch', '--no-tags', str(repo),
                    '+' + commit + ':refs/heads/ci-build'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', '-C', str(bare), 'symbolic-ref', 'HEAD', 'refs/heads/ci-build'], check=True)
    return bare


def make_source_tar(bare, out):
    """git archive with export attributes overridden, exactly like the original build."""
    (bare / 'info' / 'attributes').write_text('* -export-ignore -export-subst\n')
    with Path(out).open('xb') as stream:
        subprocess.run(['git', '-C', str(bare), 'archive', '--format=tar', 'HEAD'],
                       stdout=stream, check=True)


def make_bundle(bare, out):
    subprocess.run(['git', '-C', str(bare), 'bundle', 'create', str(out), 'HEAD'],
                   check=True, stdout=subprocess.DEVNULL)


def next_wheel_version(pin_version, commit):
    """0.30.0rc2.dev23+g<sha> -> 0.30.0rc2.dev24+g<new sha> (PEP 440 valid)."""
    m = re.fullmatch(r'(?P<base>.+?\.dev)(?P<n>\d+)(?P<local>\+g[0-9a-f]+)?', pin_version)
    base = (m.group('base') + str(int(m.group('n')) + 1)) if m else pin_version
    return base + '+g' + commit


def file_digests(repo, commit, paths):
    """Digest each path that exists at commit; paths absent at commit (deletions,
    e.g. when building a commit older than base) are skipped and returned apart."""
    rows, deleted = {}, []
    for rel in paths:
        # 走 git() 助手：继承 GIT_* 环境清洗与 core.fsmonitor=false
        probe = subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', commit + ':' + rel],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode != 0:
            deleted.append(rel)
            continue
        blob = git(repo, 'show', commit + ':' + rel)
        rows[rel] = hashlib.sha256(blob).hexdigest()
    return rows, deleted


def changed_paths(repo, base, commit):
    out = git(repo, 'diff', '--name-only', base + '..' + commit, text=True)
    return [line for line in out.splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='vllm-ascend git checkout')
    parser.add_argument('--ci-dir', type=Path, required=True, help='this ci/ directory')
    parser.add_argument('--delivery', type=Path, default=None,
                        help='delivery root (950_build); optional when ci/inputs/source.bundle exists')
    parser.add_argument('--commit', default=None,
                        help='commit to build (default: pinned commit in ci/pins/source-input.json)')
    parser.add_argument('--base', default=None,
                        help='base commit for changed_paths in commit mode (default: pinned base_commit)')
    parser.add_argument('--wheel-version', default=None, help='override wheel version in commit mode')
    parser.add_argument('--output', type=Path, required=True, help='new external context directory')
    args = parser.parse_args()

    repo = args.repo.absolute()
    ci_dir = args.ci_dir.absolute()
    delivery = args.delivery.absolute() if args.delivery is not None else None
    out = args.output.absolute()
    for name, path in (('repo', repo), ('ci-dir', ci_dir)):
        need(path == path.resolve() and path.is_dir(), 'Canonical directory required: ' + name)
    need(out == out.resolve(), 'Canonical output path required')
    need(out != repo and repo not in out.parents and out not in repo.parents,
         'Context must live outside the repository')
    if delivery is not None:
        need(delivery == delivery.resolve() and delivery.is_dir(), 'Canonical delivery directory required')
        need(out != delivery and delivery not in out.parents and out not in delivery.parents,
             'Context must live outside the delivery root')
    need(not out.exists(), 'Context directory must not exist yet: ' + str(out))

    pin = json.loads(regular(ci_dir / 'pins' / 'source-input.json').read_text())
    recipe = json.loads(regular(ci_dir / 'pins' / 'recipe-origin.json').read_text())
    pin_commit = pin['commit']
    commit = args.commit or pin_commit
    if commit in ('HEAD', '@'):
        commit = rev_parse(repo, 'HEAD')
    else:
        commit = rev_parse(repo, commit)
    tree = rev_parse(repo, commit + '^{tree}')
    need(git(repo, 'cat-file', '-t', commit, text=True).strip() == 'commit', 'Not a commit: ' + commit)

    pin_mode = commit == pin_commit
    mode = 'pin' if pin_mode else 'commit'

    out.mkdir()
    try:
        # 1) static context members from this ci directory
        shutil.copyfile(ci_dir / 'docker' / 'Dockerfile.legacy', out / 'Dockerfile')
        shutil.copyfile(ci_dir / 'scripts' / 'candidate_probe.py', out / 'candidate_probe.py')

        # 2) source.tar (both modes) and source.bundle (commit mode) from one bare repo
        with tempfile.TemporaryDirectory(prefix='step5-raw-archive-', dir=out.parent) as temporary:
            bare = prepare_bare(repo, commit, temporary)
            make_source_tar(bare, out / 'source.tar')
            if not pin_mode:
                make_bundle(bare, out / 'source.bundle')

        # 3) bundle + pin records
        if pin_mode:
            # 优先使用仓库内 vendor 的 bundle（ci/inputs/source.bundle），使本仓库
            # 不依赖交付目录即可完成 pin 复现；交付目录仅作 fallback。
            repo_bundle = ci_dir / 'inputs' / 'source.bundle'
            if repo_bundle.is_file():
                bundle_src, bundle_origin = regular(repo_bundle), 'repo'
            elif delivery is not None:
                bundle_src = regular(delivery / 'build' / 'inputs' / 'source.bundle')
                bundle_origin = 'delivery'
            else:
                need(False, 'pin 模式需要 ci/inputs/source.bundle 或 --delivery')
            need(sha(bundle_src) == pin['bundle_sha256'],
                 'Bundle digest differs from pin ({}); delivery inputs changed?'.format(bundle_origin))
            shutil.copyfile(bundle_src, out / 'source.bundle')
            shutil.copyfile(ci_dir / 'pins' / 'source-input.json', out / 'source-input.json')
            shutil.copyfile(ci_dir / 'pins' / 'feature-source-record.json',
                            out / 'feature-source-record.json')
        else:
            base = args.base or pin['base_commit']
            base = rev_parse(repo, base)
            bundle_sha = sha(out / 'source.bundle')
            archive_sha = sha(out / 'source.tar')
            wheel_version = args.wheel_version or next_wheel_version(pin['wheel_version'], commit)
            paths = changed_paths(repo, base, commit)
            digests, deleted = file_digests(repo, commit, paths)
            # 显式字段白名单重建 pin 记录：不继承 cpu_tests / format_ci_returncode /
            # local_contract_tests / remote 等 pin 专属字段（新 commit 的语义可能不同）。
            new_pin = {
                'schema_version': pin.get('schema_version', 1),
                'commit': commit,
                'tree': tree,
                'base_commit': base,
                'base': dict(pin['base']),
                'wheel_version': wheel_version,
                'wheel_filename': 'vllm_ascend-{}-cp312-cp312-linux_aarch64.whl'.format(wheel_version),
                'expected_wheel_sha256': None,
                'bundle_ref': pin.get('bundle_ref', 'HEAD'),
                'bundle_sha256': bundle_sha,
                'archive_sha256': archive_sha,
                'changed_paths': paths,
                'deleted_paths': deleted,
                'production_files': {k: v for k, v in digests.items() if not k.startswith('tests/')},
                'test_files': {k: v for k, v in digests.items() if k.startswith('tests/')},
                # cpu_tests 取原 pin 的 CPU 契约集中仍存在于新 commit 的部分（交集），
                # 不把新增测试自动纳入——新测试是否可 CPU 运行需人工确认。
                'cpu_tests': [t for t in pin.get('cpu_tests', []) if t in digests],
            }
            record = {
                'base_commit': base,
                'commit': commit,
                'tree': tree,
                'branch': git(repo, 'branch', '--show-current', text=True).strip() or '(detached)',
                'files': digests,
            }
            # 写序：先落盘 record，再算其 sha 注入 source-input.json——
            # candidate_probe.advance() 无条件断言 sha(feature-source-record.json)
            # == pin['feature_record_sha256']，顺序颠倒会使 commit 模式构建在容器内失败。
            (out / 'feature-source-record.json').write_text(json.dumps(record, indent=2) + '\n')
            new_pin['feature_record_sha256'] = sha(out / 'feature-source-record.json')
            (out / 'source-input.json').write_text(json.dumps(new_pin, indent=2) + '\n')

        # 4) verify member closure and digests
        actual = {name: sha(regular(out / name)) for name in sorted(CONTEXT_NAMES)}
        need({p.name for p in out.iterdir()} == CONTEXT_NAMES, 'Context member closure mismatch')
        pin_match = None
        if pin_mode:
            expected = recipe['context_files']
            for name, digest in actual.items():
                need(digest == expected[name],
                     'Pin reproduction mismatch for {}: expected {} got {}'.format(name, expected[name], digest))
            pin_match = True
        else:
            pin_match = False
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise

    print(json.dumps({'pass': True, 'mode': mode, 'source_commit': commit, 'source_tree': tree,
                      'context': str(out), 'context_files': actual, 'pin_match': pin_match}, indent=2))


if __name__ == '__main__':
    main()
