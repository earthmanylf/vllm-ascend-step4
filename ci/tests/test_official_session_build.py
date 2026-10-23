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
TAILRING_PREPARE = CI / "scripts/prepare-tailring-source-rebase.py"
tailring_spec = importlib.util.spec_from_file_location("tailring_prepare", TAILRING_PREPARE)
tailring = importlib.util.module_from_spec(tailring_spec)
tailring_spec.loader.exec_module(tailring)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecipeTests(unittest.TestCase):
    def test_exact_recipe_aliases_and_helpers(self):
        lock = json.loads((CI / "pins/official-v030-cann92.json").read_text())
        for name in ("Dockerfile", "Dockerfile.fast", "Dockerfile.official-v030-cann92"):
            self.assertEqual(sha(CI / "docker" / name), lock["recipe_sha256"])
        for name, expected in lock["runtime_sha256"].items():
            self.assertEqual(sha(CI / "runtime/official-v030-cann92" / name), expected)
        self.assertIn("!patch_flash_hcc.py", (CI / "docker/Dockerfile.dockerignore").read_text())
        recipe = (CI / "docker/Dockerfile").read_text()
        self.assertNotIn("c139edab", recipe)
        self.assertIn("source /opt/upgrade/cann92-build-env.sh", recipe)
        self.assertIn("patch_flash_hcc.py", recipe)
        self.assertIn("cann-9.2.0-950-openeuler24.03-py3.12-devel-ascend.tar.gz", recipe)
        self.assertNotIn("ascend-payload.tar.gz", recipe)

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
            CI / "build_legacy.sh",
            CI / "scripts/build-official-v030-cann92.sh",
            CI / "scripts/build-tailring-source-rebase.sh",
        ]
        paths += list((CI / "runtime/official-v030-cann92").glob("*.sh"))
        for path in paths:
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_cli_help_and_fail_fast(self):
        result = subprocess.run(["bash", str(CI / "build.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("formal CANN 9.2.0", result.stdout)
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
            "runtime_sha256": {"helper.py": sha(helper)},
            "inputs": [{"file": "a.tar.gz", "bytes": archive.stat().st_size, "sha256": sha(archive)}],
        }
        (self.ci / "pins/official-v030-cann92.json").write_text(json.dumps(self.lock))
        self.out = self.root / "context"

    def run_main(self, *flags):
        argv = [str(PREPARE), "--ci-dir", str(self.ci), "--inputs-dir", str(self.inputs), *flags]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            prepare.main()

    def test_prepare_snapshots_and_refuses_overwrite(self):
        self.run_main("--output", str(self.out))
        report = json.loads((self.out / "context-manifest.json").read_text())
        self.assertTrue(report["all_inputs_available"])
        self.assertEqual(sha(self.out / "a.tar.gz"), self.lock["inputs"][0]["sha256"])
        (self.inputs / "a.tar.gz").write_bytes(b"changed after snapshot")
        self.assertEqual(sha(self.out / "a.tar.gz"), self.lock["inputs"][0]["sha256"])

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
        (self.ci / "docker/Dockerfile.fast").write_text("FROM old-native\n")
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
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


class TailRingContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tailring-source-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ci = self.root / "delivery/repo/ci"
        self.inputs = self.root / "inputs"
        for name in ("docker", "pins", "runtime/official-v030-cann92"):
            (self.ci / name).mkdir(parents=True)
        self.inputs.mkdir()
        recipe = self.ci / "docker/Dockerfile.tailring-source-rebase"
        recipe.write_text("FROM locked-test-base\n")
        self.lock = {
            "cache_profile": "tailring-cann9.2.0-143f3649d8",
            "source_rebase_recipe_sha256": sha(recipe),
            "source_rebase_base_image": "test-base",
            "source_rebase_base_config_digest": "sha256:test-base-config",
            "source_branch": "step5-preview-tailring-npu",
            "source_commits": {"vllm": "test-vllm", "vllm-ascend": "test-ascend"},
            "source_snapshots": {"catlass": {"commit": "test-catlass"}},
            "package_versions": {"vllm": "test-version"},
            "runtime_sha256": {},
            "inputs": [],
        }
        for name in ("apply_dependency_profile.py", "probe_environment.py"):
            helper = self.ci / "runtime/official-v030-cann92" / name
            helper.write_text("# locked test helper: " + name + "\n")
            self.lock["runtime_sha256"][name] = sha(helper)
        for name in (
            "vllm.tar.gz",
            "vllm-ascend.tar.gz",
            "ascend-catlass.tar.gz",
            "uv-test.whl",
            "unused/nested.tar.gz",
        ):
            path = self.inputs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("locked test payload: " + name).encode())
            self.lock["inputs"].append(
                {
                    "file": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
            )
        self.lock["uv"] = dict(self.lock["inputs"][3], version="test-uv")
        self.pins = self.ci / "pins/official-v030-cann92.json"
        self.write_lock()
        self.out = self.root / "context"

    def write_lock(self):
        self.pins.write_text(json.dumps(self.lock))

    def run_main(self, *flags):
        argv = [str(TAILRING_PREPARE), "--ci-dir", str(self.ci), "--inputs-dir", str(self.inputs), *flags]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print") as printed:
            tailring.main()
        return json.loads(printed.call_args.args[0])

    def test_context_exports_complete_locked_identity_and_immutable_copies(self):
        report = self.run_main("--output", str(self.out))
        manifest = json.loads((self.out / "context-manifest.json").read_text())
        self.assertEqual(report["context_manifest_sha256"], sha(self.out / "context-manifest.json"))
        self.assertEqual(report["pins_sha256"], sha(self.pins))
        self.assertEqual((self.out / report["pins_file"]).read_bytes(), self.pins.read_bytes())
        for key in ("source_branch", "source_commits", "source_snapshots", "package_versions", "uv"):
            self.assertEqual(manifest[key], self.lock[key])
        self.assertEqual(manifest["base_image"], self.lock["source_rebase_base_image"])
        self.assertEqual(manifest["base_config_digest"], self.lock["source_rebase_base_config_digest"])
        self.assertEqual(manifest["runtime_sha256"], self.lock["runtime_sha256"])
        self.assertEqual(manifest["device_or_build_validation"], "not run")
        self.assertTrue(manifest["all_inputs_available"])
        self.assertEqual(len(manifest["verified_inputs"]), 4)
        self.assertNotIn("unused/nested.tar.gz", manifest["context_files"])
        self.assertEqual(
            manifest["context_files"],
            {path.name: sha(path) for path in self.out.iterdir() if path.name != "context-manifest.json"},
        )
        (self.inputs / "vllm.tar.gz").write_bytes(b"changed after snapshot")
        self.assertEqual(sha(self.out / "vllm.tar.gz"), self.lock["inputs"][0]["sha256"])

    def test_default_build_entry_uses_tailring_prepare_without_building(self):
        scripts = self.ci / "scripts"
        scripts.mkdir()
        for source in (CI / "build.sh", CI / "scripts/build-tailring-source-rebase.sh", TAILRING_PREPARE):
            destination = self.ci / "build.sh" if source.name == "build.sh" else scripts / source.name
            destination.write_bytes(source.read_bytes())
        # The default cache contains pointers, not hydrated inputs.
        cache = self.ci / "cache" / self.lock["cache_profile"]
        cache.mkdir(parents=True)
        for row in self.lock["inputs"][:4]:
            (cache / row["file"]).write_text(
                f"version https://git-lfs.github.com/spec/v1\noid sha256:{row['sha256']}\nsize {row['bytes']}\n"
            )
        environment = os.environ.copy()
        environment.pop("CI_UPGRADE_INPUTS_DIR", None)
        result = subprocess.run(
            ["bash", str(self.ci / "build.sh"), "--check-only"], capture_output=True, text=True, env=environment
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["source_commits"], self.lock["source_commits"])
        self.assertEqual(len(report["unhydrated_lfs_inputs"]), 4)
        self.assertFalse(report["all_inputs_available"])
        self.assertFalse(self.out.exists())
        result = subprocess.run(
            ["bash", str(self.ci / "build.sh"), "--dry-run", "--context-dir", str(self.out)],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git lfs pull", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.out.exists())

    def test_duplicate_or_missing_required_lock_entries_refused(self):
        original = list(self.lock["inputs"])
        for rows, message in (
            (original + [original[0]], "Duplicate input lock"),
            (original + [original[-1]], "Duplicate input lock"),
            (original[1:], "input lock is incomplete"),
            (original[:3] + original[4:], "input lock is incomplete"),
        ):
            with self.subTest(message=message, files=[row["file"] for row in rows]):
                self.lock["inputs"] = rows
                self.write_lock()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_main("--check-only")

    def test_unsafe_lock_paths_and_invalid_sizes_or_hashes_refused(self):
        original = dict(self.lock["inputs"][-1])
        invalid = [
            {"file": name}
            for name in (
                "../outside.tar.gz",
                "/outside.tar.gz",
                "nested/../../outside",
                "nested/./archive",
                "nested//archive",
                r"nested\archive",
                "!Dockerfile",
                "",
            )
        ]
        invalid += [{"bytes": size} for size in (-1, True, "123")]
        invalid += [{"sha256": "not-a-sha256"}]
        for change in invalid:
            with self.subTest(change=change):
                self.lock["inputs"][-1] = dict(original, **change)
                self.write_lock()
                with self.assertRaisesRegex(ValueError, "Unsafe input lock path|Invalid input lock"):
                    self.run_main("--check-only")

    def test_uv_identity_must_match_selected_lock_row(self):
        self.lock["uv"]["sha256"] = "0" * 64
        self.write_lock()
        with self.assertRaisesRegex(ValueError, "uv identity does not match"):
            self.run_main("--check-only")

    def test_missing_input_is_reported_but_cannot_prepare_context(self):
        (self.inputs / "vllm.tar.gz").unlink()
        report = self.run_main("--check-only")
        self.assertEqual(report["missing_inputs"], ["vllm.tar.gz"])
        self.assertEqual(report["unhydrated_lfs_inputs"], [])
        self.assertFalse(report["all_inputs_available"])
        with self.assertRaisesRegex(ValueError, "Missing source rebase inputs"):
            self.run_main("--output", str(self.out))
        self.assertFalse(self.out.exists())

    def test_matching_lfs_pointer_is_reported_as_unhydrated_and_cannot_build(self):
        row = self.lock["inputs"][0]
        (self.inputs / row["file"]).write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{row['sha256']}\nsize {row['bytes']}\n"
        )
        report = self.run_main("--check-only")
        self.assertEqual(report["unhydrated_lfs_inputs"], [row["file"]])
        self.assertEqual(report["missing_inputs"], [row["file"]])
        self.assertFalse(report["all_inputs_available"])
        with self.assertRaisesRegex(ValueError, "git lfs pull"):
            self.run_main("--output", str(self.out))
        self.assertFalse(self.out.exists())

    def test_mismatched_or_ambiguous_lfs_pointer_refused(self):
        row = self.lock["inputs"][0]
        for oid, size in (("0" * 64, row["bytes"]), (row["sha256"], row["bytes"] + 1)):
            with self.subTest(oid=oid, size=size):
                (self.inputs / row["file"]).write_text(
                    f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
                )
                with self.assertRaisesRegex(ValueError, "LFS pointer does not match"):
                    self.run_main("--check-only")
        (self.inputs / row["file"]).write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{row['sha256']}\nsize {row['bytes']}\nsize 1\n"
        )
        with self.assertRaisesRegex(ValueError, "LFS pointer does not match"):
            self.run_main("--check-only")

    def test_input_size_and_same_size_tamper_refused(self):
        row = self.lock["inputs"][0]
        for payload in (b"x", b"x" * row["bytes"]):
            with self.subTest(size=len(payload)):
                (self.inputs / row["file"]).write_bytes(payload)
                with self.assertRaisesRegex(ValueError, "Input checksum/size mismatch"):
                    self.run_main("--check-only")

    def test_input_directory_and_symlink_escape_refused(self):
        path = self.inputs / "vllm.tar.gz"
        payload = path.read_bytes()
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.run_main("--check-only")
        path.rmdir()
        external = self.root / "external"
        external.write_bytes(payload)
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink escapes root"):
            self.run_main("--check-only")
        path.unlink()
        path.symlink_to(self.inputs / "missing")
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.run_main("--check-only")

    def test_ci_recipe_helper_and_pins_symlink_escape_refused(self):
        for relative in (
            "docker/Dockerfile.tailring-source-rebase",
            "runtime/official-v030-cann92/probe_environment.py",
            "pins/official-v030-cann92.json",
        ):
            with self.subTest(path=relative):
                path = self.ci / relative
                contents = path.read_bytes()
                external = self.root / "external"
                external.write_bytes(contents)
                path.unlink()
                path.symlink_to(external)
                with self.assertRaisesRegex(ValueError, "symlink escapes root"):
                    self.run_main("--check-only")
                path.unlink()
                path.write_bytes(contents)

    def test_recipe_and_helper_checksum_tamper_refused(self):
        for relative, message in (
            ("docker/Dockerfile.tailring-source-rebase", "Dockerfile checksum mismatch"),
            ("runtime/official-v030-cann92/probe_environment.py", "Helper checksum mismatch"),
        ):
            with self.subTest(path=relative):
                path = self.ci / relative
                contents = path.read_bytes()
                path.write_bytes(b"x" * len(contents))
                with self.assertRaisesRegex(ValueError, message):
                    self.run_main("--check-only")
                path.write_bytes(contents)

    def test_copy_time_input_helper_recipe_and_pins_tamper_refused(self):
        original_copy = tailring.shutil.copyfile
        for name, label in (
            ("vllm.tar.gz", "Input"),
            ("probe_environment.py", "Helper"),
            ("Dockerfile", "Recipe"),
            ("official-v030-cann92.json", "Pins"),
        ):
            with self.subTest(file=name):
                output = self.root / ("context-" + label)

                def changed_copy(source, destination, name=name):
                    result = original_copy(source, destination)
                    if destination.name == name:
                        destination.write_bytes(b"x" * destination.stat().st_size)
                    return result

                with (
                    mock.patch.object(tailring.shutil, "copyfile", side_effect=changed_copy),
                    self.assertRaisesRegex(ValueError, label + " changed during copy"),
                ):
                    self.run_main("--output", str(output))
                self.assertFalse((output / "context-manifest.json").exists())

    def test_final_context_rechecks_previously_copied_files(self):
        original_copy = tailring.shutil.copyfile

        def changed_earlier_copy(source, destination):
            result = original_copy(source, destination)
            if destination.name == "official-v030-cann92.json":
                archive = destination.parent / "vllm.tar.gz"
                archive.write_bytes(b"x" * archive.stat().st_size)
            return result

        with (
            mock.patch.object(tailring.shutil, "copyfile", side_effect=changed_earlier_copy),
            self.assertRaisesRegex(ValueError, "Context changed during preparation"),
        ):
            self.run_main("--output", str(self.out))
        self.assertFalse((self.out / "context-manifest.json").exists())

    def test_context_existing_overlap_symlink_and_missing_parent_refused(self):
        self.out.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(self.root / "delivery", target_is_directory=True)
        non_directory = self.root / "not-a-directory"
        non_directory.write_text("not a context parent")
        cases = (
            (self.out, "new canonical directory"),
            (self.ci.parent / "context", "must not overlap"),
            (self.inputs / "context", "must not overlap"),
            (self.root / "missing/context", "Context parent does not exist"),
            (non_directory / "context", "Context parent does not exist"),
            (alias / "context", "new canonical directory"),
        )
        for output, message in cases:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, message):
                self.run_main("--output", str(output))


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
