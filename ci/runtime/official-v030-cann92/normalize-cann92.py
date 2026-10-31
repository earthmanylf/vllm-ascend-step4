"""Normalize a CANN-only A5 image without assuming a prebuilt vLLM filesystem."""

import json
import re
import sys
from pathlib import Path


def toolkit_version(home):
    for name in ('version.cfg', 'version.info', 'share/info/version.cfg'):
        path = home / name
        if path.is_file():
            text = path.read_text(errors='replace')
            versions = re.findall(r'(?<!\d)(\d+\.\d+\.\d+)(?!\d)', text)
            if versions:
                return versions[0]
    return '9.2.0' if '9.2.0' in str(home) else None


def discover_toolkit(root):
    candidates = (root / 'cann-9.2.0', root / 'ascend-toolkit/latest', root / 'latest')
    for candidate in candidates:
        home = candidate.resolve()
        if (home / 'set_env.sh').is_file() and (home / 'opp').is_dir():
            version = toolkit_version(home)
            if version == '9.2.0':
                return home
    raise RuntimeError('No verified CANN 9.2.0 toolkit found; inspect the downloaded CANN base')


def normalize(root, output, interpreter):
    home = discover_toolkit(root)
    canonical = root / 'cann-9.2.0'
    if not canonical.exists() and not canonical.is_symlink():
        canonical.symlink_to(home, target_is_directory=True)
    elif canonical.resolve() != home:
        raise RuntimeError('Existing CANN canonical path points to a different toolkit')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    commands = home / 'python/site-packages/bin'
    if commands.is_dir():
        for original in commands.iterdir():
            if not original.is_file():
                continue
            with original.open('rb') as stream:
                line = stream.readline()
            if line.startswith(b'#!') and b'python' in line:
                wrapper = output / original.name
                wrapper.write_text(f'#!/bin/sh\nexec "{interpreter}" "{original}" "$@"\n')
                wrapper.chmod(0o755)
                rows.append(str(original))
    result = {'toolkit': str(home), 'canonical': str(canonical),
              'python': str(interpreter), 'wrapped_commands': rows}
    (output.parent / 'wrappers.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    assert sys.version_info[:2] == (3, 12), 'A5 wheels require Python 3.12'
    print(json.dumps(normalize(Path('/usr/local/Ascend'),
                               Path('/opt/upgrade/cann-cli/bin'),
                               Path(sys.executable)), indent=2))
