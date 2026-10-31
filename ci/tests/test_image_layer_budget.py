import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts/check-image-layer-budget.py"
SPEC = importlib.util.spec_from_file_location("image_layer_budget", PATH)
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


class ImageLayerBudgetTests(unittest.TestCase):
    def test_measured_deep_base_is_rejected(self):
        recipe = "RUN true\n" * 22
        self.assertFalse(budget.layer_budget(103, recipe)["within_budget"])

    def test_compacted_base_has_room(self):
        recipe = "RUN true\n" * 22
        self.assertTrue(budget.layer_budget(80, recipe)["within_budget"])

    def test_metadata_and_continuations_are_not_filesystem_layers(self):
        recipe = "ARG A\nENV A=b\nRUN echo x \\\n && echo y\nCOPY a /a\nADD b /b\nLABEL x=y\n"
        self.assertEqual(budget.filesystem_instruction_count(recipe), 3)

    def test_boundary_is_inclusive(self):
        self.assertTrue(budget.layer_budget(116, "RUN true\n")["within_budget"])
        self.assertFalse(budget.layer_budget(117, "RUN true\n")["within_budget"])

    def test_invalid_counts_fail(self):
        with self.assertRaises(ValueError):
            budget.layer_budget(-1, "")
        with self.assertRaises(ValueError):
            budget.layer_budget(0, "", limit=0)


if __name__ == "__main__":
    unittest.main()
