"""Host-only regression tests; no Docker daemon, network, model, or NPU use."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CI = Path(__file__).resolve().parents[1]
PREPARE = CI / "scripts/prepare-official-v030-cann92.py"
spec = importlib.util.spec_from_file_location("session_prepare", PREPARE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
fetch_spec = importlib.util.spec_from_file_location("session_fetch", CI / "scripts/fetch-cann92-payload.py")
fetch = importlib.util.module_from_spec(fetch_spec)
fetch_spec.loader.exec_module(fetch)
seed_spec = importlib.util.spec_from_file_location("session_seed", CI / "scripts/seed-shared-cache.py")
seed = importlib.util.module_from_spec(seed_spec)
seed_spec.loader.exec_module(seed)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecipeTests(unittest.TestCase):
    def test_exact_recipe_aliases_and_helpers(self):
        lock = json.loads((CI / "pins/official-v030-cann92.json").read_text())
        # One native recipe and one overlay recipe; only the native one is
        # checksum-locked against the pin.
        self.assertEqual(sha(CI / "docker/Dockerfile"), lock["recipe_sha256"])
        for name, expected in lock["runtime_sha256"].items():
            self.assertEqual(sha(CI / "runtime/official-v030-cann92" / name), expected)
        self.assertIn("!patch_flash_hcc.py", (CI / "docker/Dockerfile.dockerignore").read_text())
        recipe = (CI / "docker/Dockerfile").read_text()
        self.assertNotIn("c139edab", recipe)
        self.assertIn("source /opt/upgrade/cann92-build-env.sh", recipe)
        self.assertIn("patch_flash_hcc.py", recipe)
        self.assertNotIn("COPY cann92-release/", recipe)
        self.assertNotIn("ascend-payload.tar.gz", recipe)

    def test_prepared_vllm_version_is_a_consumed_build_argument(self):
        recipe = (CI / "docker/Dockerfile").read_text()
        build = (CI / "scripts/build-official-v030-cann92.sh").read_text()
        self.assertIn("ARG VLLM_VERSION=", recipe)
        self.assertIn('--build-arg "VLLM_VERSION=$VLLM_VERSION"', build)

    def test_shared_cache_tracking_rules(self):
        self.assertNotIn("\n/cache/\n", (CI / ".gitignore").read_text())
        self.assertIn("/cache/**/*.tar.gz filter=lfs", (CI / ".gitattributes").read_text())
        self.assertIn(
            "ci/cache/tailring-cann9.2.0-143f3649d8",
            (CI / "scripts/build-official-v030-cann92.sh").read_text().replace("$CI_DIR/cache/", "ci/cache/"),
        )
        lock = json.loads((CI / "pins/official-v030-cann92.json").read_text())
        self.assertNotEqual(lock["session_recipe_sha256"], lock["recipe_sha256"])
        manifest = json.loads((CI / "cache/manifest.json").read_text())
        self.assertEqual(manifest["storage"], "git-lfs")
        self.assertEqual(len(manifest["archives"]), len(lock["inputs"]))

    def test_shell_syntax(self):
        paths = [
            CI / "build.sh",
            CI / "scripts/build-official-v030-cann92.sh",
        ]
        paths += list((CI / "runtime/official-v030-cann92").glob("*.sh"))
        for path in paths:
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_cli_help_and_fail_fast(self):
        result = subprocess.run(["bash", str(CI / "build.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("python-overlay", result.stdout)
        for flags in (["--image"], ["--image", "--dry-run"], ["--unknown"], ["--backend", "buildctl"]):
            result = subprocess.run(["bash", str(CI / "build.sh"), *flags], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, flags)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="official-session-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ci = self.root / "delivery/repo/ci"
        self.inputs = self.root / "delivery/inputs"
        for name in ("docker", "pins", "runtime/official-v030-cann92"):
            (self.ci / name).mkdir(parents=True)
        self.inputs.mkdir()
        for name in ("Dockerfile", "Dockerfile.fast", "Dockerfile.official-v030-cann92"):
            (self.ci / "docker" / name).write_text("FROM test\n")
        (self.ci / "docker/Dockerfile.dockerignore").write_text("*\n!Dockerfile\n!helper.py\n!a.tar.gz\n")
        helper = self.ci / "runtime/official-v030-cann92/helper.py"
        helper.write_text('print("test")\n')
        archive = self.inputs / "a.tar.gz"
        archive.write_bytes(b"fake archive for checksum/copy tests, not a build")
        self.lock = {
            "session_id": "test",
            "recipe_sha256": sha(self.ci / "docker/Dockerfile"),
            "cache_profile": "test-profile",
            "vllm_ascend_base_version": "0.30.0rc1",
            "vllm_base_version": "0.30.0rc1",
            "vllm_source": {"commit": "a" * 40},
            "runtime_sha256": {"helper.py": sha(helper)},
            "inputs": [{"file": "a.tar.gz", "bytes": archive.stat().st_size, "sha256": sha(archive)}],
        }
        (self.ci / "pins/official-v030-cann92.json").write_text(json.dumps(self.lock))
        # vllm_ascend is exported from git now, so the fixture needs a real
        # checkout with one commit rather than a bare directory.
        (self.ci.parent / "vllm_ascend").mkdir()
        (self.ci.parent / "vllm_ascend" / "__init__.py").write_text("")
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", str(self.ci.parent)], check=True, env={**os.environ, **env})
        subprocess.run(["git", "-C", str(self.ci.parent), "add", "-A"], check=True,
                       env={**os.environ, **env})
        subprocess.run(["git", "-C", str(self.ci.parent), "commit", "-qm", "fixture"],
                       check=True, env={**os.environ, **env})
        self.out = self.root / "context"

    def run_main(self, *flags):
        argv = [str(PREPARE), "--ci-dir", str(self.ci), "--inputs-dir", str(self.inputs), *flags]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            prepare.main()

    def test_prepare_snapshots_and_refuses_overwrite(self):
        self.run_main("--output", str(self.out))
        report = json.loads((self.out / "context-manifest.json").read_text())
        self.assertTrue(report["all_inputs_available"])
        version_info = json.loads((self.out / "source-version.json").read_text())
        self.assertEqual(version_info["vllm_base_version"], self.lock["vllm_base_version"])
        self.assertEqual(version_info["vllm_commit"], self.lock["vllm_source"]["commit"])
        self.assertTrue(version_info["vllm_version"].startswith("0.30.0"))
        self.assertIn(self.lock["vllm_source"]["commit"], version_info["vllm_version"])
        self.assertEqual(sha(self.out / "a.tar.gz"), self.lock["inputs"][0]["sha256"])
        (self.inputs / "a.tar.gz").write_bytes(b"changed after snapshot")
        self.assertEqual(sha(self.out / "a.tar.gz"), self.lock["inputs"][0]["sha256"])

    def test_non_union_context_has_empty_optional_prebuilt_input(self):
        self.run_main("--output", str(self.out))
        prebuilt = self.out / "third-party/prebuilt"
        self.assertTrue(prebuilt.is_dir())
        self.assertTrue((prebuilt / ".keep").is_file())
        self.assertFalse(any(p.is_dir() for p in prebuilt.iterdir()))

    def test_union_context_stages_required_vendor_libraries(self):
        exclude = self.ci.parent / ".git/info/exclude"
        with exclude.open("a") as stream:
            stream.write("\nci/third-party/prebuilt/\n")
        vendors = ("union_test_transformer", "custom_nn_nn")
        for vendor in vendors:
            library = self.ci / "third-party/prebuilt" / vendor / "op_api/lib/libcust_opapi.so"
            library.parent.mkdir(parents=True)
            library.write_bytes(vendor.encode())
        self.run_main("--output", str(self.out), "--with-union-ops")
        for vendor in vendors:
            library = self.out / "third-party/prebuilt" / vendor / "op_api/lib/libcust_opapi.so"
            self.assertEqual(library.read_bytes(), vendor.encode())
        report = json.loads((self.out / "context-manifest.json").read_text())
        self.assertEqual(report["union_ops_staged"], sorted(vendors))

    def test_existing_context_refused(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_main("--output", str(self.out))

    def test_missing_input_reported_and_build_refused(self):
        (self.inputs / "a.tar.gz").unlink()
        self.run_main("--check-only")
        with self.assertRaisesRegex(ValueError, "Missing inputs"):
            self.run_main("--output", str(self.out))
        self.assertFalse(self.out.exists())

    def test_tamper_refused(self):
        (self.inputs / "a.tar.gz").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.run_main("--check-only")

    def test_same_size_tamper_refused(self):
        p = self.inputs / "a.tar.gz"
        p.write_bytes(b"x" * p.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_main("--check-only")

    def test_helper_and_recipe_tamper_refused(self):
        p = self.ci / "runtime/official-v030-cann92/helper.py"
        p.write_text("changed")
        with self.assertRaisesRegex(ValueError, "helper hash mismatch"):
            self.run_main("--check-only")
        p.write_text('print("test")\n')
        (self.ci / "docker/Dockerfile").write_text("FROM old-native\n")
        with self.assertRaisesRegex(ValueError, "differs from locked recipe"):
            self.run_main("--check-only")

    def test_context_inside_delivery_refused(self):
        with self.assertRaisesRegex(ValueError, "outside repository"):
            self.run_main("--output", str(self.inputs.parent / "context"))

    def test_symlink_escape_refused(self):
        p = self.inputs / "a.tar.gz"
        p.rename(self.root / "external")
        p.symlink_to(self.root / "external")
        with self.assertRaisesRegex(ValueError, "escapes archive root"):
            self.run_main("--check-only")

    def test_lfs_pointer_is_missing_not_an_archive(self):
        row = self.lock["inputs"][0]
        (self.inputs / row["file"]).write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{row['sha256']}\nsize {row['bytes']}\n"
        )
        self.run_main("--check-only")
        with self.assertRaisesRegex(ValueError, "git lfs pull"):
            self.run_main("--output", str(self.out))
        self.assertFalse(self.out.exists())

    def test_wrong_lfs_pointer_refused(self):
        (self.inputs / "a.tar.gz").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:wrong\nsize 123\n"
        )
        with self.assertRaisesRegex(ValueError, "pointer does not match"):
            self.run_main("--check-only")

    def test_seed_shared_cache_copy_and_hit(self):
        args = [str(CI / "scripts/seed-shared-cache.py"), "--ci-dir", str(self.ci), "--source", str(self.inputs)]
        with mock.patch.object(sys, "argv", args), mock.patch("builtins.print"):
            seed.main()
            target = self.ci / "cache/test-profile/a.tar.gz"
            self.assertEqual(sha(target), self.lock["inputs"][0]["sha256"])
            first_inode = target.stat().st_ino
            seed.main()
            self.assertEqual(target.stat().st_ino, first_inode)

    def test_seed_rehydrates_matching_pointer(self):
        target = self.ci / "cache/test-profile/a.tar.gz"
        target.parent.mkdir(parents=True)
        row = self.lock["inputs"][0]
        target.write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{row['sha256']}\nsize {row['bytes']}\n"
        )
        args = [str(CI / "scripts/seed-shared-cache.py"), "--ci-dir", str(self.ci), "--source", str(self.inputs)]
        with mock.patch.object(sys, "argv", args), mock.patch("builtins.print"):
            seed.main()
        self.assertEqual(sha(target), row["sha256"])


class DownloadTests(unittest.TestCase):
    def test_named_tag_resolves_locked_payload(self):
        index = {"manifests": [{"platform": {"architecture": "arm64", "os": "linux"}, "digest": fetch.ARM64_MANIFEST}]}
        manifest = {"layers": [{"digest": "sha256:" + fetch.SHA, "size": fetch.SIZE}]}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"token": "test-token"}'
        with (
            mock.patch.object(fetch.urllib.request, "urlopen", return_value=response),
            mock.patch.object(
                fetch, "get_manifest", side_effect=[(index, "sha256:index"), (manifest, fetch.ARM64_MANIFEST)]
            ),
        ):
            _, identity = fetch.resolve_image()
        self.assertEqual(identity["image"], fetch.IMAGE)
        self.assertEqual(identity["filename"], fetch.FILENAME)

    def test_moved_named_tag_refused(self):
        index = {"manifests": [{"platform": {"architecture": "arm64", "os": "linux"}, "digest": "sha256:changed"}]}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"token": "test-token"}'
        with (
            mock.patch.object(fetch.urllib.request, "urlopen", return_value=response),
            mock.patch.object(fetch, "get_manifest", return_value=(index, "sha256:index")),
            self.assertRaisesRegex(ValueError, "no longer matches"),
        ):
            fetch.resolve_image()

    def test_download_cache_hit_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / fetch.FILENAME
            path.write_bytes(b"test-valid-cache")
            with (
                mock.patch.object(fetch, "SIZE", path.stat().st_size),
                mock.patch.object(fetch, "SHA", sha(path)),
                mock.patch.object(sys, "argv", ["fetch", "--output", str(path)]),
                mock.patch.object(fetch.urllib.request, "urlopen") as network,
                mock.patch("builtins.print"),
            ):
                fetch.main()
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
