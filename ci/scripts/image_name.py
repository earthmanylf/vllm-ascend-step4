#!/usr/bin/env python3
"""Build the release image reference, following the stepfun/v0.17.0rc1 scheme.

That scheme tags as

    <endpoint>/stepcast:<release>-vllm-<sha>-ascend-<sha>-<date>-<time>

with an 8-character commit for each source (git's default abbreviation) and a
`YYYYMMDD` / `HHMMSS` stamp,
e.g. hub.i.basemind.com/stepcast/stepcast:
0.30.0rc2-step5-20261016-vllm-e177c621f2-ascend-a2ac749ee3.

The `vllm-` / `ascend-` prefixes name which commit is which; without them the tag
is two unlabelled SHAs. The stamp is the real build time, so two builds of one
commit stay distinguishable.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

# Registry shortcuts, mirrored from stepfun/v0.17.0rc1:ci/build.sh so a caller can
# say `--registry UNIHUB` instead of spelling the host out.
ENDPOINTS = {
    'SH-VOLCANO-MLP': 'registry-ivolces.stepfun-inc.com/mlsys',
    'SH-VOLCANO-SYS': 'basemind-sys-cn-shanghai.cr.volces.com/mlsys',
    'UNIHUB': 'hub.i.basemind.com/stepcast',
    'BJ-BAIDU': 'ccr-2guxcp55-vpc.cnc.bj.baidubce.com/basemind-core',
    'SH-ALI': 'sh-cr-alicloud.stepfun-inc.com/stepcast',
}
DEFAULT_ENDPOINT = 'UNIHUB'
REPOSITORY = 'stepcast'


def endpoint(value: str) -> str:
    """Resolve a shortcut or pass a full registry path through."""
    key = value.strip().upper()
    if key in ENDPOINTS:
        return ENDPOINTS[key]
    if '/' in value:
        return value.rstrip('/')
    raise SystemExit(f'Unknown registry: {value!r}. Known shortcuts: '
                     + ', '.join(sorted(ENDPOINTS)))


def release_of(version: str) -> str:
    """'0.30.0rc2+g500440ee3' -> '0.30.0rc2'; drops local/dev suffixes."""
    base = version.split('+', 1)[0]
    base = base.split('.dev', 1)[0]
    return base


def short(commit: str, length: int = 8) -> str:
    commit = commit.strip()
    if not re.fullmatch(r'[0-9a-f]{7,64}', commit):
        raise SystemExit(f'Not a commit SHA: {commit!r}')
    return commit[:length]


def build_tag(release: str, vllm_commit: str, ascend_commit: str,
              stamp: datetime.datetime | None = None) -> str:
    stamp = stamp or datetime.datetime.now()
    return '-'.join((
        release,
        'vllm-' + short(vllm_commit),
        'ascend-' + short(ascend_commit),
        stamp.strftime('%Y%m%d'),
        stamp.strftime('%H%M%S'),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', default=DEFAULT_ENDPOINT,
                        help='shortcut (UNIHUB, ...) or full registry path')
    parser.add_argument('--release', help='defaults to the derived version')
    parser.add_argument('--version-file', type=pathlib.Path,
                        help='context source-version.json')
    parser.add_argument('--vllm-commit', required=True)
    parser.add_argument('--ascend-commit', required=True)
    parser.add_argument('--stamp', help='fixed YYYYMMDD-HHMMSS, for tests')
    args = parser.parse_args()

    release = args.release
    if not release:
        if not args.version_file:
            raise SystemExit('--release or --version-file is required')
        release = release_of(
            json.loads(args.version_file.read_text())['version'])
    stamp = None
    if args.stamp:
        stamp = datetime.datetime.strptime(args.stamp, '%Y%m%d-%H%M%S')
    tag = build_tag(release, args.vllm_commit, args.ascend_commit, stamp)
    print(f'{endpoint(args.registry)}/{REPOSITORY}:{tag}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
