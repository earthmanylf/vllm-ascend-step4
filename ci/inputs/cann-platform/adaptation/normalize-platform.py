#!/usr/bin/env python3
"""Preserve official CLI bytes and select the retained PR interpreter explicitly."""
from pathlib import Path
import hashlib,json,sys,os
assert sys.version_info[:3] == (3,12,13),sys.version
home=Path('/usr/local/Ascend/cann-9.2.0-beta.2')
output=Path('/opt/step5/cann-cli/bin');output.mkdir(parents=True,exist_ok=True)
rows=[]
for name in ['cannsim','msopgen','msopst','msserviceprofiler','npusim','optype_collector']:
    original=home/'python/site-packages/bin'/name
    data=original.read_bytes()
    assert data.splitlines()[0] == b'#!/usr/local/python3.12.14/bin/python3.12',original
    wrapper=output/name
    wrapper.write_text('#!/bin/sh\nexec /usr/local/python3.12.13/bin/python3.12 "'+str(original)+'" "$@"\n')
    wrapper.chmod(0o755)
    assert original.read_bytes()==data
    rows.append({'name':name,'official_path':str(original),'official_sha256':hashlib.sha256(data).hexdigest(),'official_bytes_unchanged':True,'wrapper':str(wrapper),'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),'interpreter':'/usr/local/python3.12.13/bin/python3.12'})
Path('/opt/step5-cann92-evidence/cli-interpreter-adaptation.json').write_text(json.dumps({'scope':'Environment wrappers only; original payload files and metadata unchanged; optional CLI dependencies are not added','python':sys.version,'tools':rows},indent=2)+'\n')
print('CANN92_CLI_WRAPPERS_CREATED',len(rows))
