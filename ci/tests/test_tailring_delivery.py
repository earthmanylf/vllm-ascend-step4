"""Check published identity, launch defaults and experimental separation."""

import hashlib
import json
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def test_final_identity_recipe_and_helper_are_locked(self):
        lock = json.loads((CI / "pins/official-v030-cann92.json").read_text())["validated_final_image"]
        for relative, key in (
            ("runtime/repack_ascend_revision.py", "repack_helper_sha256"),
            ("docker/Dockerfile.tailring-final-identity", "repack_recipe_sha256"),
        ):
            self.assertEqual(hashlib.sha256((CI / relative).read_bytes()).hexdigest(), lock[key])
        recipe = (CI / "docker/Dockerfile.tailring-final-identity").read_text()
        self.assertIn("--base-wheel-sha256 " + lock["base_wheel_sha256"], recipe)
        self.assertEqual(lock["installed_distribution_version"], "0.19.1rc2.dev2977+g143f3649d8")
        self.assertEqual(lock["preserved_native_file_count"], 1114)

    def test_launch_sync_export_use_verified_final_not_rebuild(self):
        final = json.loads((CI / "pins/official-v030-cann92.json").read_text())["validated_final_image"]["image"]
        for relative in (
            "scripts/start-tailring-dp8tp4.sh",
            "scripts/export-tailring-image.sh",
            "scripts/sync-tailring-image.sh",
        ):
            self.assertIn(final, (CI / relative).read_text())
        build = (CI / "scripts/build-tailring-source-rebase.sh").read_text()
        self.assertIn("tailring-143f3649d8-rebuild", build)
        serve = (CI / "runtime/serve-tailring-dp8tp4.sh").read_text()
        self.assertNotIn("step5_compact_summary", serve)
        self.assertNotIn("controlled_mtp_extension", serve)

    def test_benchmark_report_and_unvalidated_candidate_are_separate(self):
        summary = json.loads((CI / "reports/benchmark-final-run1-summary.json").read_text())
        measured = [key for key in summary if key.startswith("bs")]
        self.assertEqual(len(measured), 7)
        for key in measured:
            self.assertEqual(summary[key]["status"], "completed")
            self.assertEqual(summary[key]["incomplete_requests"], 0)
        self.assertNotIn("bs128_input131072_output2048", summary)
        candidate = CI / "experiments/step5-compact-summary"
        self.assertTrue((candidate / "candidate.patch").is_file())
        self.assertIn("UNVALIDATED", (candidate / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
