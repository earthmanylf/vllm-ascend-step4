"""CPU-only candidate identity proof; no driver/device mounts or external imports."""
from pathlib import Path
import base64
import csv
import hashlib
import importlib
from importlib import metadata, util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path('/opt/step5/feature-artifacts')
INPUT = Path('/opt/step5/feature-inputs')
SOURCE = Path('/vllm-workspace/vllm-ascend')

def wanted():
    return json.loads((INPUT / 'source-input.json').read_text())




def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    ROOT.mkdir(exist_ok=True)
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()


def distributions():
    rows = {}
    for d in metadata.distributions():
        name = re.sub(r'[-_.]+', '-', d.metadata['Name']).lower()
        if name == 'vllm-ascend':
            continue
        root = Path(d._path).resolve()
        rows.setdefault(name, []).append({'version': d.version, 'metadata_path': str(root),
            'metadata_files': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}})
    return {k: sorted(v, key=lambda r: (r['version'], r['metadata_path'])) for k, v in sorted(rows.items())}


def identity():
    result = {'packages': distributions(), 'python': sys.version, 'executable': sys.executable,
              'cann_home': os.environ['ASCEND_HOME_PATH'], 'cann_opp': os.environ['ASCEND_OPP_PATH'],
              'custom_opp': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
              'runtime_environment_sha256': sha('/opt/step5/bin/step5-env.sh'),
              'build_environment_sha256': sha('/opt/step5/bin/step5-build-env.sh'),
              'non_ascend_source_lock': [r for r in json.loads(Path('/opt/step5/inputs/source-lock.json').read_text())['sources'] if r['name'] != 'vllm-ascend'],
              'vendor_library_sha256': sha('/usr/local/Ascend/cann-9.2.0-beta.2/opp/vendors/step5_ops_transformer/op_api/lib/libcust_opapi.so')}
    assert result['cann_home'] == '/usr/local/Ascend/cann-9.2.0-beta.2', result['cann_home']
    triton = util.find_spec('triton')
    assert triton is not None and triton.submodule_search_locations, 'Triton package not found'
    result['triton_files'] = {str(p): sha(p) for root in triton.submodule_search_locations
                              for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name.endswith(('.py', '.so'))}
    assert result['triton_files'], 'Triton files not enumerated'
    assert result['vendor_library_sha256'] == 'd6d06852629e58b9f615176afe1a4558825bd1d9f47ca8821ca80d13ac22dcc5'
    return result


def source():
    pin = wanted()
    COMMIT, VERSION = pin['commit'], pin['wheel_version']
    assert sha(INPUT / 'source.bundle') == pin['bundle_sha256']
    assert os.environ['FEATURE_COMMIT'] == COMMIT and os.environ['FEATURE_TREE'] == pin['tree']
    assert git('rev-parse', 'HEAD') == COMMIT
    assert git('rev-parse', 'HEAD^{tree}') == pin['tree']
    assert not git('status', '--porcelain')
    assert not git('diff', '--name-only', pin['base_commit'] + '..HEAD', '--', 'csrc', '.gitmodules')
    for name, digest in {**pin['production_files'], **pin['test_files']}.items():
        assert sha(SOURCE / name) == digest, name
    assert sha(INPUT / 'source.tar') == pin['archive_sha256']
    # Verify archive is exactly the tracked regular-file tree, with safe unique names.
    rows = [r.split('\t', 1) for r in git('ls-tree', '-r', COMMIT).splitlines()]
    expected = {name: spec.split() for spec, name in rows if spec.split()[1] == 'blob'}
    seen = set()
    with tarfile.open(INPUT / 'source.tar', 'r:') as archive:
        for member in archive:
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert member.name not in seen
            seen.add(member.name)
            if member.isdir():
                continue
            assert member.isfile() or member.issym(), member.name
            assert expected[member.name][0] in ('100644', '100755', '120000'), member.name
            blob = subprocess.check_output(['git', '-C', str(SOURCE), 'show', COMMIT + ':' + member.name])
            data = member.linkname.encode() if member.issym() else archive.extractfile(member).read()
            assert data == blob, member.name
            assert member.issym() == (expected[member.name][0] == '120000'), member.name
        assert {n for n in seen if n in expected} == set(expected)
    lock = Path('/opt/step5/inputs/source-lock.json')
    original = json.loads(lock.read_text())
    save('base-source-lock.json', original)
    assert len([r for r in original['sources'] if r['name'] == 'vllm-ascend']) == 1
    for row in original['sources']:
        if row['name'] == 'vllm-ascend':
            row.update(commit=COMMIT, tree=pin['tree'], branch=git('branch', '--show-current'))
            row['refs'] = dict(line.split(' ', 1)[::-1] for line in git('for-each-ref', '--format=%(objectname) %(refname)').splitlines())
            row['commit_count'] = int(git('rev-list', '--all', 'HEAD', '--count'))
    lock.write_text(json.dumps(original, indent=2) + '\n')
    save('source-identity.json', {**pin, 'commit': COMMIT, 'branch': git('branch', '--show-current'),
                               'version': VERSION, 'tracked_clean': True, 'git_history_preserved': True})
    # Remove only untracked packaging outputs; the accepted native csrc cache is separate.
    cleaned = []
    for path in [SOURCE / 'build', *SOURCE.glob('*.egg-info')]:
        assert not git('ls-files', '--', str(path.relative_to(SOURCE))), path
        assert not path.is_symlink(), path
        if path.exists():
            shutil.rmtree(path)
            cleaned.append(str(path))
    save('packaging-cleanup.json', {'removed': cleaned, 'native_cache_preserved': True})


def wheel():
    pin = wanted()
    VERSION = pin['wheel_version']
    wheels = list(ROOT.glob('*.whl'))
    assert len(wheels) == 1
    path = wheels[0]
    assert path.name == pin['wheel_filename'], path.name
    assert pin['expected_wheel_sha256'] is None or sha(path) == pin['expected_wheel_sha256']
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names)
        assert not any('__editable__' in n or n.endswith('.pth') for n in names)
        record = [n for n in names if n.endswith('.dist-info/RECORD')]
        assert len(record) == 1
        verified = 0
        record_names = []
        for name, digest, size in csv.reader(io.StringIO(z.read(record[0]).decode())):
            record_names.append(name)
            data = z.read(name)
            if digest:
                alg, expected = digest.split('=', 1)
                assert alg == 'sha256'
                assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode() == expected
                assert len(data) == int(size)
                verified += 1
            else:
                assert name == record[0]
        assert len(record_names) == len(set(record_names))
        assert set(record_names) == {n for n in names if not n.endswith('/')}
        meta_name = next(n for n in names if n.endswith('.dist-info/METADATA'))
        from email.parser import BytesParser
        assert BytesParser().parsebytes(z.read(meta_name))['Version'] == VERSION
        members = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names if not n.endswith('/')}
        required = json.loads((INPUT / 'source-input.json').read_text())['production_files']
        assert all(members.get(n) == digest for n, digest in required.items())
        native = [n for n in names if n.startswith('vllm_ascend/') and n.endswith('.so')]
        assert any('/vllm_ascend_C.' in n or n.endswith('/vllm_ascend_C.so') for n in native), native
        custom = [n for n in names if n.startswith('vllm_ascend/_cann_ops_custom/')]
        assert custom and any(n.endswith('.so') for n in custom)
        wheel_meta = z.read(next(n for n in names if n.endswith('.dist-info/WHEEL'))).decode()
        assert 'Root-Is-Purelib: false' in wheel_meta and 'cp312-cp312-linux_aarch64' in wheel_meta
    save('wheel-identity.json', {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
        'version': VERSION, 'record_verified_members': verified, 'members': members,
        'native_members': native, 'custom_ops_members': len(custom), 'complete_native_wheel': True,
        'source_mapping_files': [], 'pass': True})


def after():
    pin = wanted()
    COMMIT, VERSION = pin['commit'], pin['wheel_version']
    before = json.loads((ROOT / 'before.json').read_text())
    now = identity()
    comparable = dict(now)
    relocation = json.loads((ROOT / 'custom-ops-relocation.json').read_text())
    assert now['custom_opp'] == before['custom_opp'], 'Accepted base wheel vendor path changed'
    if comparable != before:
        changed = sorted(k for k in set(before['packages']) | set(now['packages'])
                         if before['packages'].get(k) != comparable['packages'].get(k))
        print(json.dumps({'changed_distributions': changed,
              'distribution_differences': {k: {'before_paths': [r['metadata_path'] for r in before['packages'].get(k, [])],
                                             'after_paths': [r['metadata_path'] for r in now['packages'].get(k, [])]} for k in changed},
              'other_changed_keys': [k for k in now if k != 'packages' and now[k] != before.get(k)]}), flush=True)
    assert comparable == before, 'Non-Ascend distribution metadata/platform changed'
    assert str(SOURCE) not in {str(Path(p).resolve()) for p in sys.path if p}, 'Source path remains on sys.path'
    assert not any(Path('/dev').glob('davinci*')), 'Device unexpectedly exposed to CPU build'
    d = metadata.distribution('vllm-ascend')
    ascend_distributions = {str(Path(x._path).resolve()) for x in metadata.distributions()
                            if re.sub(r'[-_.]+', '-', x.metadata['Name']).lower() == 'vllm-ascend'}
    assert len(ascend_distributions) == 1, ascend_distributions
    assert d.version == VERSION, d.version
    info = Path(d._path).resolve()
    assert info.name.endswith('.dist-info') and '/site-packages/' in str(info)
    direct = json.loads((info / 'direct_url.json').read_text())
    assert not direct.get('dir_info', {}).get('editable', False)
    wheel_info = json.loads((ROOT / 'wheel-identity.json').read_text())
    package = importlib.import_module('vllm_ascend')
    package_root = Path(package.__file__).resolve().parent
    assert '/site-packages/vllm_ascend' in str(package_root)
    installed_files = {}
    for name, digest in json.loads((INPUT / 'source-input.json').read_text())['production_files'].items():
        p = (package_root / Path(name).relative_to('vllm_ascend')).resolve()
        assert '/site-packages/vllm_ascend/' in str(p) and SOURCE not in p.parents, p
        assert sha(p) == digest == wheel_info['members'][name]
        installed_files[name] = {'path': str(p), 'sha256': sha(p)}
    installed_members = {}
    for name, digest in wheel_info['members'].items():
        if not name.startswith('vllm_ascend/'):
            continue
        path = package_root / Path(name).relative_to('vllm_ascend')
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, name
        installed_members[name] = digest
    expected_code = {n for n in installed_members if n.endswith(('.py', '.so'))}
    actual_code = {'vllm_ascend/' + str(p.relative_to(package_root)) for p in package_root.rglob('*')
                   if p.is_file() and p.name.endswith(('.py', '.so'))}
    assert actual_code == expected_code, {'extra_installed_code': sorted(actual_code - expected_code),
                                        'missing_installed_code': sorted(expected_code - actual_code)}
    import torch_npu
    native = importlib.import_module('vllm_ascend.vllm_ascend_C')
    p = Path(native.__file__).resolve()
    assert '/site-packages/vllm_ascend/' in str(p)
    assert sha(p) == wheel_info['members']['vllm_ascend/' + p.name]
    for site in {str(p) for p in sys.path if '/site-packages' in str(p)}:
        for p in Path(site).glob('*'):
            if p.is_file() and ('editable' in p.name and 'ascend' in p.name):
                raise AssertionError('Ascend editable mapping remains: ' + str(p))
    assert not git('diff', '--name-only', 'HEAD')
    save('cpu-identity.json', {'pass': True, 'source_commit': COMMIT, 'version': VERSION,
        'distribution_path': str(info), 'direct_url': direct,
        'package_import': {'path': str(Path(package.__file__).resolve()), 'sha256': sha(package.__file__)},
        'installed_production_files': installed_files, 'installed_wheel_package_members': installed_members,
        'full_model_modules_imported_on_cpu': False,
        'full_model_import_scope': 'Full module import asks Triton for actual NPU architecture; deferred to eight-worker device test',
        'native_import': {
            'path': str(Path(native.__file__).resolve()), 'sha256': sha(native.__file__)},
        'dependencies_and_platform_unchanged': True, 'device_nodes_exposed': False,
        'custom_ops_relocation': relocation,
        'production_overlay_mounts': False, 'native_kernel_execution_tested': False,
        'scope': 'CPU import and wheel identity only; real device/model acceptance is separate',
        'platform_identity': now})


def relocate():
    # Accepted a2ac parent already uses a standard wheel. Preserve its environment byte for byte.
    package = importlib.import_module('vllm_ascend')
    installed = Path(package.__file__).resolve().parent / '_cann_ops_custom/vendors/custom_transformer'
    old = SOURCE / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
    assert installed.is_dir() and any(installed.rglob('*.so'))
    entries_before = [p for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if p]
    # Normal runtime registers the installed wheel vendor through this production
    # helper. Importing the lightweight package alone does not register its ops.
    from vllm_ascend.utils import bootstrap_custom_op_env
    bootstrap_custom_op_env()
    entries = [p for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if p]
    assert str(installed) in entries and str(old) not in entries, entries
    assert [p for p in entries if p != str(installed)] == [p for p in entries_before if p != str(installed)], 'Other vendor paths changed'
    vendor = '/usr/local/Ascend/cann-9.2.0-beta.2/opp/vendors/step5_ops_transformer'
    assert vendor in entries
    assert sha(Path(vendor) / 'op_api/lib/libcust_opapi.so') == 'd6d06852629e58b9f615176afe1a4558825bd1d9f47ca8821ca80d13ac22dcc5'
    env = Path('/opt/step5/bin/step5-env.sh')
    save('custom-ops-relocation.json', {'installed_vendor': str(installed), 'old_vendor': str(old),
         'environment_sha256': sha(env), 'environment_modified': False,
         'entries_before_bootstrap': entries_before, 'entries_after_bootstrap': entries,
         'registration': 'vllm_ascend.utils.bootstrap_custom_op_env',
         'step5_vendor_unchanged': True, 'scope': 'Actual production bootstrap registers installed wheel vendor; environment scripts preserved'})


def advance():
    pin = wanted()
    assert git('rev-parse', 'HEAD') == pin['base_commit']
    assert not git('status', '--porcelain')
    assert sha(INPUT / 'source.bundle') == pin['bundle_sha256']
    assert sha(INPUT / 'source.tar') == pin['archive_sha256']
    assert sha(INPUT / 'feature-source-record.json') == pin['feature_record_sha256']
    subprocess.run(['git', '-C', str(SOURCE), 'bundle', 'verify', str(INPUT / 'source.bundle')], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'fetch', str(INPUT / 'source.bundle'), pin['bundle_ref']], check=True)
    assert git('rev-parse', 'FETCH_HEAD') == pin['commit']
    subprocess.run(['git', '-C', str(SOURCE), 'merge', '--ff-only', 'FETCH_HEAD'], check=True)


def tests():
    pin = wanted()
    for name, digest in pin['test_files'].items():
        assert sha(SOURCE / name) == digest, name
    argv = [sys.executable, '-m', 'pytest', '--noconftest', '-p', 'no:cacheprovider', '-q',
            '--override-ini', 'addopts=', '--junitxml=' + str(ROOT / 'cpu-contracts.xml'),
            *[str(SOURCE / name) for name in pin['cpu_tests']]]
    with (ROOT / 'cpu-contracts.log').open('x') as log:
        result = subprocess.run(argv, cwd='/tmp', stdout=log, stderr=subprocess.STDOUT)
    suites = ET.parse(ROOT / 'cpu-contracts.xml').getroot().iter('testsuite')
    counts = {k: 0 for k in ('tests', 'failures', 'errors', 'skipped')}
    for suite in suites:
        for k in counts:
            counts[k] += int(suite.get(k, '0'))
    passed = result.returncode == 0 and counts['tests'] > 0 and not any(counts[k] for k in ('failures', 'errors', 'skipped'))
    save('cpu-contracts-status.json', {'pass': passed, 'returncode': result.returncode,
         'counts': counts, 'argv': argv, 'source_commit': pin['commit'],
         'scope': 'Pinned source CPU contracts only; installed wheel/device acceptance separate'})
    assert passed, 'CPU contracts failed, skipped or did not execute; inspect cpu-contracts.log'


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'before':
        save('before.json', identity())
    elif mode == 'advance':
        advance()
    elif mode == 'tests':
        tests()
    elif mode == 'source':
        source()
    elif mode == 'wheel':
        wheel()
    elif mode == 'after':
        after()
    elif mode == 'relocate':
        relocate()
    else:
        raise ValueError(mode)
