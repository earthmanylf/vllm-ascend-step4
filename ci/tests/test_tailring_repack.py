"""Host-only synthetic wheel tests; no model imports, Docker, network or devices."""

import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

REPACK = Path(__file__).resolve().parents[1] / "runtime/repack_ascend_revision.py"
spec = importlib.util.spec_from_file_location("tailring_repack", REPACK)
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)


class RepackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tailring-repack-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.module = self.source / repack.MODULE
        self.module.parent.mkdir(parents=True)
        self.module.write_bytes(b"# synthetic locked MTP replacement\n")
        self.identity = self.source / "SOURCE-IDENTITY.json"
        self.identity.write_text(json.dumps({"commit": repack.SOURCE_COMMIT, "branch": repack.SOURCE_BRANCH}))
        module_lock = mock.patch.object(repack, "MODULE_SHA256", repack.sha(self.module.read_bytes()))
        module_lock.start()
        self.addCleanup(module_lock.stop)
        self.wheel = self.root / "wheels" / f"vllm_ascend-{repack.OLD_VERSION}-{repack.TAG}.whl"
        self.wheel.parent.mkdir()
        self.output = self.root / "output"
        self.files = {
            repack.MODULE: b"# old MTP module\n",
            "vllm_ascend/_version.py": f"__version__ = version = {repack.OLD_VERSION!r}\n".encode(),
            "vllm_ascend/models/untouched.py": b"# unchanged model\n",
            repack.NATIVE_EXTENSION: b"synthetic shared library",
            repack.OLD_INFO + "/METADATA": (
                f"Metadata-Version: 2.1\nName: vllm_ascend\nVersion: {repack.OLD_VERSION}\n"
                "Requires-Dist: triton-ascend==3.6.0+dev20261013182112\n\n"
            ).encode(),
            repack.OLD_INFO + "/WHEEL": (f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {repack.TAG}\n").encode(),
            repack.OLD_INFO + "/entry_points.txt": b"[vllm.platform_plugins]\nascend=vllm_ascend:register\n",
        }
        for index in range(repack.NATIVE_FILE_COUNT - 1):
            self.files[f"vllm_ascend/_cann_ops_custom/native-{index}"] = f"native {index}".encode()
        self.write_wheel()

    def write_wheel(self, extra=None, fix_record=True):
        if fix_record:
            record = io.StringIO()
            writer = csv.writer(record, lineterminator="\n")
            for name, data in sorted(self.files.items()):
                if name == repack.OLD_INFO + "/RECORD":
                    continue
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                writer.writerow((name, "sha256=" + digest, len(data)))
            writer.writerow((repack.OLD_INFO + "/RECORD", "", ""))
            self.files[repack.OLD_INFO + "/RECORD"] = record.getvalue().encode()
        with zipfile.ZipFile(self.wheel, "w") as wheel:
            for name, data in self.files.items():
                wheel.writestr(name, data)
            if extra:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    wheel.writestr(*extra)
        self.base_sha = repack.sha(self.wheel.read_bytes())

    def run_repack(self):
        return repack.repack(self.wheel, self.source, self.output, self.base_sha)

    def assert_refused(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_repack()
        self.assertFalse(self.output.exists(), "Validation must precede output creation")

    def test_preserves_payload_dependencies_tags_and_rebuilds_record(self):
        destination, ledger = self.run_repack()
        self.assertEqual(destination.name, f"vllm_ascend-{repack.VERSION}-{repack.TAG}.whl")
        with zipfile.ZipFile(destination) as wheel:
            result = {name: wheel.read(name) for name in wheel.namelist()}
        repack.verify_record(result, repack.NEW_INFO)
        self.assertEqual(repack.native_hashes(result), repack.native_hashes(self.files))
        self.assertEqual(ledger["native_file_count"], 1114)
        self.assertEqual(ledger["base_wheel_sha256"], self.base_sha)
        self.assertEqual(ledger["native_source_commit"], repack.NATIVE_COMMIT)
        self.assertEqual(ledger["source_commit"], repack.SOURCE_COMMIT)
        self.assertEqual(ledger["module_sha256"], repack.sha(self.module.read_bytes()))
        self.assertEqual(ledger["wheel_sha256"], repack.sha(destination.read_bytes()))
        self.assertEqual(json.loads((self.output / "ascend-wheel-revision-audit.json").read_text()), ledger)
        self.assertEqual(json.loads(result[repack.AUDIT])["native_files"], ledger["native_files"])
        self.assertEqual(result[repack.MODULE], self.module.read_bytes())
        generated = {}
        for node in ast.parse(result["vllm_ascend/_version.py"]).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    generated[target.id] = ast.literal_eval(node.value)
        self.assertEqual(generated["version"], repack.VERSION)
        self.assertEqual(generated["__version__"], repack.VERSION)
        self.assertEqual(generated["commit_id"], repack.SOURCE_COMMIT)
        self.assertEqual(generated["version_tuple"], (0, 19, 1, "rc2", "dev2977", "g143f3649d8"))
        for name, data in self.files.items():
            if name in (repack.MODULE, "vllm_ascend/_version.py", repack.OLD_INFO + "/RECORD"):
                continue
            new_name = name.replace(repack.OLD_INFO + "/", repack.NEW_INFO + "/", 1)
            expected = data
            if name == repack.OLD_INFO + "/METADATA":
                expected = data.replace(
                    f"Version: {repack.OLD_VERSION}\n".encode(), f"Version: {repack.VERSION}\n".encode(), 1
                )
            self.assertEqual(result[new_name], expected, name)
        expected_names = {name.replace(repack.OLD_INFO + "/", repack.NEW_INFO + "/", 1) for name in self.files} | {
            repack.AUDIT
        }
        self.assertEqual(set(result), expected_names)

    def test_base_digest_and_filename_must_match_explicit_lock(self):
        self.base_sha = "0" * 64
        self.assert_refused("Base wheel SHA256")
        self.base_sha = "not-a-sha256"
        self.assert_refused("lowercase base wheel SHA256")
        self.base_sha = repack.sha(self.wheel.read_bytes())
        self.wheel = self.wheel.rename(self.wheel.with_name("wrong-version.whl"))
        self.assert_refused("filename/version/tag")

    def test_source_revision_and_module_bytes_are_locked(self):
        self.identity.write_text(json.dumps({"commit": repack.NATIVE_COMMIT, "branch": repack.SOURCE_BRANCH}))
        self.assert_refused("source revision/branch")
        self.identity.write_text(json.dumps({"commit": repack.SOURCE_COMMIT, "branch": "wrong"}))
        self.assert_refused("source revision/branch")
        self.identity.write_text(json.dumps({"commit": repack.SOURCE_COMMIT, "branch": repack.SOURCE_BRANCH}))
        self.module.write_bytes(b"# tampered module\n")
        self.assert_refused("MTP source module SHA256")

    def test_source_symlink_escape_and_missing_module_refused(self):
        self.module.unlink()
        self.assert_refused("Source file missing")
        external = self.root / "external.py"
        external.write_bytes(b"# synthetic locked MTP replacement\n")
        self.module.symlink_to(external)
        self.assert_refused("Source file escapes root")

    def test_native_count_requires_exactly_1114_files(self):
        missing = self.files.pop("vllm_ascend/_cann_ops_custom/native-0")
        self.write_wheel()
        self.assert_refused("Expected 1114 native files, got 1113")
        self.files["vllm_ascend/_cann_ops_custom/native-0"] = missing
        self.files["vllm_ascend/_cann_ops_custom/extra"] = b"unexpected"
        self.write_wheel()
        self.assert_refused("Expected 1114 native files, got 1115")

    def test_metadata_version_name_and_abi_cannot_be_relabelled(self):
        metadata = self.files[repack.OLD_INFO + "/METADATA"]
        wheel_metadata = self.files[repack.OLD_INFO + "/WHEEL"]
        cases = (
            (repack.OLD_INFO + "/METADATA", metadata.replace(repack.OLD_VERSION.encode(), b"9.9"), "metadata version"),
            (
                repack.OLD_INFO + "/METADATA",
                metadata.replace(b"Name: vllm_ascend", b"Name: other"),
                "distribution name",
            ),
            (repack.OLD_INFO + "/METADATA", metadata.replace(b"\n\n", b"\nVersion: extra\n\n"), "metadata version"),
            (repack.OLD_INFO + "/WHEEL", wheel_metadata.replace(b"cp312", b"cp311"), "ABI/platform"),
            (repack.OLD_INFO + "/WHEEL", wheel_metadata.replace(b"false", b"true"), "must not be pure"),
            ("vllm_ascend/_version.py", b"version = '9.9'\n", "_version.py identity"),
        )
        for name, data, message in cases:
            with self.subTest(message=message):
                old = self.files[name]
                self.files[name] = data
                self.write_wheel()
                self.assert_refused(message)
                self.files[name] = old

    def test_missing_required_files_and_extra_distribution_refused(self):
        for name in (
            repack.MODULE,
            repack.NATIVE_EXTENSION,
            repack.OLD_INFO + "/METADATA",
            repack.OLD_INFO + "/WHEEL",
            repack.OLD_INFO + "/RECORD",
        ):
            with self.subTest(name=name):
                data = self.files.pop(name)
                self.write_wheel(fix_record=name != repack.OLD_INFO + "/RECORD")
                self.assert_refused("missing")
                self.files[name] = data
        self.files["other-1.0.dist-info/METADATA"] = b"unexpected distribution"
        self.write_wheel()
        self.assert_refused("Unexpected wheel root")

    def test_duplicate_unsafe_and_symlink_zip_members_refused(self):
        for name, message in (
            (repack.MODULE, "Duplicate wheel member"),
            ("../escape", "Unsafe wheel member"),
            ("/escape", "Unsafe wheel member"),
            (r"vllm_ascend\escape", "Unsafe wheel member"),
        ):
            with self.subTest(name=name):
                self.write_wheel(extra=(name, b"unexpected"))
                self.assert_refused(message)
        link = zipfile.ZipInfo("vllm_ascend/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.write_wheel(extra=(link, b"../../external"))
        self.assert_refused("Non-regular wheel member")

    def test_invalid_incomplete_or_duplicate_record_refused(self):
        record = self.files[repack.OLD_INFO + "/RECORD"]
        for data, message in (
            (record.replace(b"sha256=", b"sha512=", 1), "checksum/size mismatch"),
            (b"\n".join(record.splitlines()[1:]) + b"\n", "cover every file"),
            (record + record.splitlines()[0] + b"\n", "duplicate wheel RECORD"),
            (record + b"vllm_ascend/missing.py,sha256=missing,0\n", "names missing file"),
        ):
            with self.subTest(message=message):
                self.files[repack.OLD_INFO + "/RECORD"] = data
                self.write_wheel(fix_record=False)
                self.assert_refused(message)

    def test_signed_or_previously_repacked_base_refused(self):
        for name, message in (
            (repack.OLD_INFO + "/RECORD.jws", "signature policy"),
            (repack.AUDIT, "already contains a revision audit"),
        ):
            with self.subTest(name=name):
                self.files[name] = b"existing audit or signature"
                self.write_wheel()
                self.assert_refused(message)
                self.files.pop(name)

    def test_outputs_are_not_overwritten_or_written_into_sources(self):
        destination, _ = self.run_repack()
        before = destination.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite wheel"):
            self.run_repack()
        self.assertEqual(destination.read_bytes(), before)
        self.output = self.source / "output"
        self.assert_refused("must not overlap")
        self.output = self.wheel.parent / "output"
        self.assert_refused("must not overlap")
        self.output = self.root / "another-output"
        self.output.mkdir()
        audit = self.output / "ascend-wheel-revision-audit.json"
        audit.write_text("existing")
        with self.assertRaisesRegex(ValueError, "overwrite revision audit"):
            self.run_repack()
        self.assertEqual(audit.read_text(), "existing")

    def test_base_wheel_symlink_and_output_symlink_parent_refused(self):
        original = self.wheel.rename(self.wheel.parent / "original.whl")
        self.wheel.symlink_to(original)
        self.assert_refused("non-symlink file")
        self.wheel.unlink()
        original.rename(self.wheel)
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.output = alias / "output"
        self.assert_refused("must be canonical")

    def test_output_corruption_is_detected_and_no_artifact_is_published(self):
        original_writestr = zipfile.ZipFile.writestr

        def corrupt_output(wheel, name, data, *args, **kwargs):
            if name == "vllm_ascend/models/untouched.py":
                data = b"# corrupt while writing\n"
            return original_writestr(wheel, name, data, *args, **kwargs)

        with (
            mock.patch.object(zipfile.ZipFile, "writestr", corrupt_output),
            self.assertRaisesRegex(ValueError, "Output wheel changed while writing"),
        ):
            self.run_repack()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_requires_base_digest_and_import_has_no_side_effects(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPACK),
                "--wheel",
                str(self.wheel),
                "--source",
                str(self.source),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--base-wheel-sha256", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
