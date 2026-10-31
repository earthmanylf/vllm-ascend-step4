"""Host-only checks for the fused CANN-base A5 recipe; no image pull/build."""

import importlib.util
import tempfile
import subprocess
import re
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'normalize_cann_base', CI / 'runtime/official-v030-cann92/normalize-cann92.py')
normalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalize)
BUDGET_SPEC = importlib.util.spec_from_file_location(
    'fused_stage_budget', CI / 'scripts/check-image-layer-budget.py')
budget = importlib.util.module_from_spec(BUDGET_SPEC)
BUDGET_SPEC.loader.exec_module(budget)


class FusedA5RecipeTests(unittest.TestCase):
    def test_single_external_cann_base(self):
        recipe = (CI / 'docker/Dockerfile').read_text()
        self.assertIn('stepcast:cann-9.2.0_20261022115200-950-openeuler24.0', recipe)
        lines = [line for line in recipe.splitlines() if line.startswith('FROM ')]
        self.assertEqual(lines[0], 'FROM ${BASE_IMAGE} AS cann-base')
        self.assertTrue(all('vllm-ascend:' not in line for line in lines))
        self.assertEqual(lines[-1], 'FROM a5-system AS runtime')

    def test_runtime_does_not_inherit_compiler_stage(self):
        recipe = (CI / 'docker/Dockerfile').read_text()
        runtime = recipe.split('FROM a5-system AS runtime', 1)[1]
        self.assertIn('COPY --from=builder /opt/upgrade/.venv', runtime)
        self.assertNotIn('/opt/upgrade/source-assets', runtime)
        self.assertNotIn('csrc/build', runtime)
        self.assertNotIn('GITLAB_ACCESS_TOKEN', runtime)

    def test_original_a5_features_and_ci_production_ops_are_present(self):
        recipe = (CI / 'docker/Dockerfile').read_text()
        for marker in ('yum install', 'mooncake_transfer_engine', 'stepcast_plugin',
                       'tools/build_rust.sh', 'flash_linear_attention_npu_a5',
                       'FLASH_ATTENTION_FORCE_BUILD', 'step5_ops_transformer'):
            self.assertIn(marker, recipe)
        self.assertNotIn('COPY cann92-release/', recipe)
        self.assertNotIn('REUSE_BASE_ASSETS', recipe)

    def test_build_interface_uses_prepared_vllm_version(self):
        script = (CI / 'scripts/build-official-v030-cann92.sh').read_text()
        self.assertIn('VLLM_COMMIT="$(json_get "$VER_JSON" vllm_commit)"', script)
        self.assertNotIn("assert runtime['packages']['vllm']['version'] == '0.29", script)
        self.assertIn('--pull=false', script)

    def test_native_check_and_dry_run_flags_are_forwarded_without_docker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ci = root / 'ci'
            (ci / 'scripts').mkdir(parents=True)
            (ci / 'build.sh').write_text((CI / 'build.sh').read_text())
            output = root / 'native-args.txt'
            stub = ci / 'scripts/build-official-v030-cann92.sh'
            stub.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{output}"\n')
            for flags in (['--check-only'], ['--dry-run', '--keep']):
                result = subprocess.run(['bash', str(ci / 'build.sh'), *flags],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                recorded = output.read_text().splitlines()
                for flag in flags:
                    self.assertIn(flag, recorded)

    def test_source_archives_do_not_keep_foreign_build_user_ownership(self):
        recipe = (CI / 'docker/Dockerfile').read_text()
        self.assertIn('tar --no-same-owner -xf /opt/upgrade/input/vllm-ascend.tar.gz', recipe)
        self.assertIn('tar --no-same-owner -xzf /opt/upgrade/input/custom-assets.tar.gz', recipe)
        self.assertIn('tar --no-same-owner -xzf /opt/upgrade/input/ops-git-metadata.tar.gz', recipe)

    def test_venv_packages_outrank_the_cann_sdk_python_path(self):
        # The base image's own cann_ops_transformer lives on the SDK python path
        # that set_env.sh prepends; without this the venv copy (the one built by
        # this recipe, with indexer_region_score) is unreachable and Step5 DSA
        # fails to import at service start.
        env = (CI / 'runtime/official-v030-cann92/cann92-env.sh').read_text()
        self.assertIn('/opt/upgrade/.venv/lib/python3.12/site-packages', env)
        self.assertIn('export PYTHONPATH="$venv_site${PYTHONPATH:+:$PYTHONPATH}"', env)
        self.assertLess(env.index('source /usr/local/Ascend/cann-9.2.0/set_env.sh'),
                        env.index('export PYTHONPATH="$venv_site'))

    def test_rust_frontend_build_gets_an_explicit_version(self):
        # The extracted source has no .git, so setuptools-scm cannot derive the
        # rust frontend version and the build fails with LookupError.
        recipe = (CI / 'docker/Dockerfile').read_text()
        rust_step = recipe.split('bash tools/build_rust.sh', 1)[0]
        self.assertIn(
            'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_RUST_FRONTEND_BUILD="$VLLM_VERSION"',
            rust_step)
        self.assertIn('VLLM_RS_BUILD_VERSION="$VLLM_VERSION"', rust_step)
        self.assertIn('--mount=type=cache,target=/root/.cargo/registry', rust_step)
        self.assertIn('--mount=type=cache,target=/root/.rustup', rust_step)

    def test_triton_ascend_wins_over_the_upstream_triton_wheel(self):
        # torch requires the `triton` distribution, whose CUDA libtriton.so
        # overwrites triton-ascend's when it installs last (observed as
        # "triton.language has no attribute extra" at service start).
        recipe = (CI / 'docker/Dockerfile').read_text()
        block = recipe.split('python3 -m pip uninstall -y triton triton-ascend', 1)[1]
        block = block.split('# Mooncake installation', 1)[0]
        self.assertIn('--force-reinstall --no-deps "$wheel"', block)
        reinstall = block.index('--force-reinstall --no-deps "$wheel"')
        for earlier in ('requirements/common.txt', 'vllm-ascend/requirements.txt',
                        'build-system'):
            self.assertLess(block.index(earlier), reinstall, earlier)
        self.assertIn('assert hasattr(tl, "extra")', block)
        self.assertIn('triton.backends', block)

    def test_fa3_native_abi_patch_is_applied_and_verified(self):
        # A fresh CANN-only build compiles flash-attention from the archived
        # source, which lacks the retained 950 paged block-stride ABI. Without
        # the patch the service dies at kv-cache init with
        # "FA3 requires the compiled paged block-stride ABI".
        recipe = (CI / 'docker/Dockerfile').read_text()
        applies = 'tools/step5/fa3_950/apply.py'
        checker = 'tools/step5/fa3_950/check_library.py'
        self.assertIn(applies, recipe)
        self.assertIn(checker, recipe)
        self.assertLess(recipe.index(applies), recipe.index('bdist_wheel'))
        self.assertLess(recipe.index('bdist_wheel'), recipe.index(checker))
        # apply.py verifies pristine source hashes and rewrites setup.py, so the
        # hcc adaptation must not have touched setup.py before it runs. Reverse
        # this order and the build dies with "Native base source mismatch:
        # setup.py" (observed).
        self.assertLess(recipe.index(applies),
                        recipe.index('/opt/upgrade/patch_flash_hcc.py'))
        self.assertIn('fa3-abi.json', recipe)
        self.assertTrue((CI.parent / applies).is_file())
        self.assertTrue((CI.parent / checker).is_file())

    def test_union_ops_are_staged_and_installed_when_enabled(self):
        # WITH_UNION_OPS=1 must install both vendor trees under
        # $ASCEND_OPP_PATH/vendors and keep the production step5 ops on the
        # runtime payload; union is additive, never a replacement.
        recipe = (CI / 'docker/Dockerfile').read_text()
        block = recipe.split('COPY third-party/prebuilt/', 1)[1].split('# Record actual', 1)[0]
        for marker in ('if [ "$WITH_UNION_OPS" = 1 ]', 'union_test_transformer',
                       'custom_nn_nn', 'op_api/lib', 'third-party-prebuilt/*.whl'):
            self.assertIn(marker, block)
        self.assertIn('vendors=step5_ops_transformer', block)
        self.assertIn('vendors="$vendors union_test_transformer custom_nn_nn"', block)
        # The default stays opt-in: a plain build must not ship union vendors.
        self.assertIn('ARG WITH_UNION_OPS=0', recipe)

    def test_union_ops_mount_only_on_request(self):
        # union_test_transformer exports the same aclnnGenericBlockSparseAttention*
        # symbols as step5_ops_transformer, so mounting it unconditionally would
        # shadow the production operator. Keep it behind VLLM_ASCEND_UNION_OPS.
        env = (CI / 'runtime/official-v030-cann92/cann92-env.sh').read_text()
        block = env.split('if [[ "${VLLM_ASCEND_UNION_OPS:-0}" == "1" ]]', 1)
        self.assertEqual(len(block), 2, 'union vendors must stay behind the env switch')
        self.assertIn('union_test_transformer', block[1])
        self.assertIn('custom_nn_nn', block[1])
        # Production ops are mounted before the switch, unconditionally.
        self.assertIn('export ASCEND_CUSTOM_OPP_PATH="$vendor"', block[0])

    def test_triton_is_verified_from_staged_input(self):
        recipe = (CI / 'docker/Dockerfile').read_text()
        self.assertIn('/opt/upgrade/wheel-input/', recipe)
        self.assertNotIn("curl -fL 'https://ascend-triton-open", recipe)
        self.assertIn('sha256sum -c -', recipe)

    def test_runtime_numpy_pin_matches_triton_and_staged_opencv(self):
        profile = (CI / 'runtime/official-v030-cann92/apply_dependency_profile.py').read_text()
        # triton-ascend 3.6.0 pins numpy==1.26.4 exactly, so NumPy 2 is not
        # installable on the audited stack.
        self.assertIn('numpy==1.26.4', profile)
        self.assertNotIn('numpy==2.2.6', profile)
        self.assertNotIn('numpy<2.0', profile)
        # OpenCV 4.13 metadata demands NumPy 2; the audited profile removes the
        # requirement line and installs the ABI-compatible binding without deps.
        self.assertIn('# opencv-python-headless installed separately', profile)
        self.assertIn('mistral_common[image] >= 1.11.6', profile)
        self.assertIn('"mistral_common >= 1.11.6"', profile)
        recipe = (CI / 'docker/Dockerfile').read_text()
        self.assertIn(
            'python3 -m pip install --no-deps "opencv-python-headless==$OPENCV_VERSION"',
            recipe)
        self.assertIn('assert numpy.__version__ == "1.26.4"', recipe)
        self.assertIn('libxcb', recipe)

    def test_dependency_downloads_use_artifactory_and_survive_retries(self):
        recipe = (CI / 'docker/Dockerfile').read_text()
        self.assertIn(
            'ARG PIP_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/'
            'pypi-public/simple', recipe)
        self.assertIn(
            'ARG PYTORCH_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/'
            'pytorch-cpu/simple', recipe)
        self.assertIn(
            'ARG PIP_EXTRA_INDEX_URL=https://artifactory.stepfun-inc.com/artifactory/api/pypi/'
            'pypi-org/simple', recipe)
        self.assertNotIn('--no-cache-dir', recipe)
        self.assertNotIn('pip cache purge', recipe)
        pip_blocks = [block for block in re.split(r'(?m)^RUN ', recipe)
                      if 'pip install' in block or 'pip wheel' in block]
        self.assertTrue(pip_blocks)
        for block in pip_blocks:
            self.assertIn(
                '--mount=type=cache,target=/root/.cache/pip', block[:400])
        # Large HTTP payloads must survive a failing retry too.
        self.assertIn('--mount=type=cache,target=/var/cache/remote-wheels', recipe)
        self.assertNotIn('/tmp/$wheel', recipe)
        build = (CI / 'scripts/build-official-v030-cann92.sh').read_text()
        self.assertIn('DOCKER_BUILDKIT=1 "$DOCKER" build', build)

    def test_build_job_arguments_reach_docker(self):
        script = (CI / 'scripts/build-official-v030-cann92.sh').read_text()
        self.assertIn('--build-arg "MAX_JOBS=$MAX_JOBS"', script)
        self.assertIn('--build-arg "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS"', script)
        self.assertIn('Build job counts must be positive integers', script)

    def test_stage_depth_does_not_sum_compiler_and_runtime(self):
        recipe = 'FROM external AS base\nFROM base AS builder\n' + 'RUN true\n' * 30
        recipe += 'FROM base AS runtime\nCOPY --from=builder /venv /venv\n'
        result = budget.layer_budget(20, recipe)
        self.assertEqual(result['native_filesystem_instruction_budget'], 30)
        self.assertEqual(result['final_stage_filesystem_budget'], 1)
        self.assertEqual(result['estimated_depth'], 53)


class CannLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'Ascend'
        self.home = self.root / 'ascend-toolkit/9.2.0'
        (self.home / 'opp').mkdir(parents=True)
        (self.home / 'set_env.sh').write_text('# CANN env\n')
        (self.home / 'version.cfg').write_text('Version=9.2.0\n')
        (self.root / 'ascend-toolkit/latest').symlink_to(self.home, target_is_directory=True)

    def test_toolkit_alias_and_cli_python_are_discovered(self):
        commands = self.home / 'python/site-packages/bin'
        commands.mkdir(parents=True)
        (commands / 'compiler').write_text('#!/usr/local/python3.12.15/bin/python3.12\nprint(1)\n')
        output = Path(self.temp.name) / 'cli/bin'
        result = normalize.normalize(self.root, output, Path('/opt/upgrade/.venv/bin/python'))
        self.assertEqual((self.root / 'cann-9.2.0').resolve(), self.home)
        self.assertIn('/opt/upgrade/.venv/bin/python', (output / 'compiler').read_text())
        self.assertEqual(len(result['wrapped_commands']), 1)

    def test_wrong_toolkit_version_is_rejected(self):
        (self.home / 'version.cfg').write_text('Version=9.1.0\n')
        with self.assertRaises(RuntimeError):
            normalize.discover_toolkit(self.root)

    def test_base_without_python_cli_folder_is_supported(self):
        output = Path(self.temp.name) / 'cli/bin'
        result = normalize.normalize(self.root, output, Path('/opt/upgrade/.venv/bin/python'))
        self.assertEqual(result['wrapped_commands'], [])


if __name__ == '__main__':
    unittest.main()
