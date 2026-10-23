"""Install exactly one newly built Step5 run package and matching Python wrapper."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve(strict=True)
runs = sorted(root.glob('cann-ops-transformer-*.run'))
wheels = sorted(root.glob('cann_ops_transformer-*.whl'))
if len(runs) != 1 or len(wheels) != 1:
    raise SystemExit(f'Expected one run and wheel, found {len(runs)} and {len(wheels)}')
opp = Path(os.environ['ASCEND_OPP_PATH']).resolve(strict=True)
if not str(opp).startswith('/usr/local/Ascend/') or opp.name != 'opp':
    raise SystemExit('Unexpected OPP install root')
runs[0].chmod(0o755)
subprocess.run([str(runs[0]), '--quiet', '--install-for-all', '--force',
                f'--install-path={opp}'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
                str(wheels[0])], check=True)
evidence = [{'file': p.name, 'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [runs[0], wheels[0]]]
Path('/opt/upgrade/evidence/ops-artifacts.json').write_text(json.dumps(evidence, indent=2))
