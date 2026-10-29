#!/usr/bin/env python3
"""Derive the vllm-ascend version for a build, anchored on the declared series.

The image has no `.git`, so `setup.py`'s setuptools_scm call cannot derive a
version on its own; the value is computed here and handed over as
`SETUPTOOLS_SCM_PRETEND_VERSION` (native build) or written into `_version.py`
(overlay build). One implementation so the two paths cannot drift.
"""
from __future__ import annotations

import re
import subprocess

DESCRIBE = re.compile(
    r'^v?(?P<release>\d+(?:\.\d+)*)'
    r'(?P<pre>a|b|rc)?(?P<pre_n>\d+)?'
    r'-(?P<distance>\d+)-g(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?$')
TAG_ONLY = re.compile(
    r'^v?(?P<release>\d+(?:\.\d+)*)'
    r'(?P<pre>a|b|rc)?(?P<pre_n>\d+)?(?P<dirty>-dirty)?$')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args],
                          check=True, capture_output=True, text=True).stdout


def release_line(version):
    """'0.30.0rc2.dev23+gabc' -> (0, 30)."""
    match = re.match(r'^(\d+)\.(\d+)', version)
    if not match:
        raise ValueError(f'Version has no release line: {version!r}')
    return int(match.group(1)), int(match.group(2))


def split_tag(tag):
    """'0.30.0rc1' -> ('0.30.0', 'rc', 1) so next_version can bump it."""
    match = re.match(r'^(?P<release>\d+(?:\.\d+)*)(?P<pre>a|b|rc)?(?P<pre_n>\d+)?$',
                     tag.lstrip('v'))
    if not match:
        raise ValueError(f'Unsupported base version: {tag!r}')
    return (match.group('release'), match.group('pre'),
            int(match.group('pre_n')) if match.group('pre_n') else None)


def next_version(release, pre, pre_n):
    """Bump a tag the way setuptools_scm's guess-next-dev does."""
    parts = [int(p) for p in release.split('.')]
    while len(parts) < 3:
        parts.append(0)
    if pre:
        return '{}.{}.{}{}{}'.format(*parts, pre, (pre_n or 0) + 1)
    parts[-1] += 1
    return '{}.{}.{}'.format(*parts)


def version_from_describe(describe):
    """`git describe --tags --long --dirty` -> PEP 440 version (no anchoring)."""
    describe = describe.strip()
    match = DESCRIBE.match(describe)
    if match:
        release, pre = match.group('release'), match.group('pre')
        pre_n = int(match.group('pre_n')) if match.group('pre_n') else None
        distance, sha = int(match.group('distance')), match.group('sha')
        dirty = bool(match.group('dirty'))
        if distance == 0:
            base = release + (f'{pre}{pre_n}' if pre else '')
            return base + ('.dirty' if dirty else '')
        local = f'+g{sha}' + ('.dirty' if dirty else '')
        return f'{next_version(release, pre, pre_n)}.dev{distance}{local}'
    match = TAG_ONLY.match(describe)
    if match:
        release, pre, pre_n = match.group('release'), match.group('pre'), match.group('pre_n')
        base = release + (f'{pre}{pre_n}' if pre else '')
        return base + ('.dirty' if match.group('dirty') else '')
    raise ValueError(f'Unrecognised git describe output: {describe!r}')


def version_for(describe, base_version, commit=None):
    """Version to install, anchored on the declared series.

    The fork's tags stop far below the series this recipe builds, so `git
    describe` can report a tag from an older line. Trusting it produced
    0.17.0rc2.dev3479 for a tree tracking v0.30.0rc1. The series therefore comes
    from `base_version`:

      * a tag inside the same series -> normal setuptools_scm derivation
      * otherwise -> bumped base + commit, omitting `dev` because the distance
        to a tag this clone does not have would be a fabricated number

    `commit` is used when `describe` is empty, i.e. the clone has no tags.
    """
    want = release_line(base_version)
    if describe:
        try:
            derived = version_from_describe(describe)
        except ValueError:
            derived = ''
        if derived and release_line(derived) == want:
            return derived, 'git-describe'
    # Prefer the commit we were given: a checkout with no tags at all still has
    # a SHA, and refusing it would make such a repository unbuildable.
    sha = commit or (describe.rsplit('-g', 1)[-1].split('-')[0]
                     if '-g' in (describe or '') else '')
    if not sha:
        raise ValueError('Cannot determine the commit to build: no describe output '
                         'and no commit given')
    bumped = next_version(*split_tag(base_version))
    return f'{bumped}+g{sha}', 'commit-only'
