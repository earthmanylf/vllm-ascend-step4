"""Adapt Flash AICPU include lookup to installed formal CANN HCC version."""
from pathlib import Path
import json
import hashlib
root=Path('/opt/upgrade/source-assets/flash-attention-npu')
p=root/'setup.py'
old='        hcc_cpp = os.path.join(hcc_isys, "c++/7.3.0")'
new='''        hcc_cpp_candidates = [p for p in Path(hcc_isys, "c++").iterdir() if (p / "cstdint").is_file()]
        if len(hcc_cpp_candidates) != 1:
            raise RuntimeError(f"Expected one installed HCC C++ header tree: {hcc_cpp_candidates}")
        hcc_cpp = str(hcc_cpp_candidates[0])'''
text=p.read_text()
assert text.count(old)==1
before=hashlib.sha256(p.read_bytes()).hexdigest()
p.write_text(text.replace(old,new))
Path('/opt/upgrade/evidence/flash-hcc-adaptation.json').write_text(json.dumps({'old':old,'new':new,'before_sha256':before,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
