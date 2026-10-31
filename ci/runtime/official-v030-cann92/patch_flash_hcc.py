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
text = p.read_text()
before = hashlib.sha256(p.read_bytes()).hexdigest()
if text.count(old) == 1:
    p.write_text(text.replace(old, new))
    record = {'action': 'rewrote hardcoded hcc c++ include', 'old': old, 'new': new,
              'before_sha256': before,
              'after_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
elif '_get_hcc_cpp_include' in text:
    # The retained FA3 950 patch (tools/step5/fa3_950) already replaced the
    # hardcoded "c++/7.3.0" with a compiler-probed lookup, which is the same
    # adaptation done generically. Nothing left to rewrite; record that fact
    # instead of failing, but only when the general form is really present.
    record = {'action': 'superseded by the FA3 950 patch', 'anchor': old,
              'resolved_by': '_get_hcc_cpp_include', 'sha256': before}
else:
    raise RuntimeError('Neither the hardcoded hcc c++ include nor the FA3 generic '
                       'lookup is present; the flash-attention source is not the '
                       'audited one')
Path('/opt/upgrade/evidence/flash-hcc-adaptation.json').write_text(json.dumps(record, indent=2))
