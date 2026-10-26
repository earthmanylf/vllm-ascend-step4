from pathlib import Path
import json
home = Path('/usr/local/Ascend/cann-9.2.0')
assert home.is_dir()
output = Path('/opt/upgrade/cann-cli/bin')
output.mkdir(parents=True, exist_ok=True)
rows = []
for original in (home / 'python/site-packages/bin').iterdir():
    if not original.is_file():
        continue
    line = original.open('rb').readline()
    if line.startswith(b'#!/usr/local/python3.12.15/'):
        wrapper = output / original.name
        wrapper.write_text('#!/bin/sh\nexec /usr/local/python3.12.13/bin/python3.12 "' + str(original) + '" "$@"\n')
        wrapper.chmod(0o755)
        rows.append(str(original))
Path('/opt/upgrade/cann-cli/wrappers.json').write_text(json.dumps(rows, indent=2))
print('Formal CANN9.2 CLI interpreter wrappers:', len(rows))
