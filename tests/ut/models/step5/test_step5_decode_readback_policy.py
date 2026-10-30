"""CPU-only regression coverage for the MRv2 MTP D2H selection."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "vllm_ascend/worker/v2/spec_decode/multi_module_mtp/speculator.py"


def load_policy(source):
    tree = ast.parse(source.read_text())
    helper = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef)
         and node.name == "_needs_exact_draft_seq_lens"),
        None,
    )
    namespace = {"os": os, "VllmConfig": object}
    if helper is not None:
        module = ast.Module(body=[helper], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        return namespace["_needs_exact_draft_seq_lens"]
    # Evaluate the old initializer's real assignment to demonstrate that the
    # regression test fails before the patch; do not reimplement the policy.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "AscendMultiModuleMTPSpeculator")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "__init__")
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute)
                              and t.attr == "_exact_draft_seq_lens"
                              for t in n.targets))
    expression = compile(ast.Expression(assignment.value), str(source), "eval")
    return lambda config: eval(expression, namespace, {"vllm_config": config})


class ReadbackPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = staticmethod(load_policy(SOURCE))

    def config(self, model="step5", transfer=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=model)),
            kv_transfer_config=transfer,
        )

    def decide(self, model="step5", exact="0", fa3="1", transfer=None, alias=None):
        environment = {
            "VLLM_ASCEND_MTP_EXACT_SEQ_LENS": exact,
            "VLLM_ASCEND_ENABLE_FA3": fa3,
        }
        if alias is not None:
            environment["VLLM_STEP4_ENABLE_FA3"] = alias
        with patch.dict(os.environ, environment, clear=True):
            return self.policy(self.config(model, transfer))

    def test_step5_fa3_monolithic_can_disable_readback(self):
        self.assertFalse(self.decide())

    def test_explicit_readback_opt_in_is_preserved(self):
        self.assertTrue(self.decide(exact="1"))

    def test_pd_roles_keep_exact_lengths_even_when_disabled(self):
        for role in ("kv_producer", "kv_consumer"):
            with self.subTest(role=role):
                self.assertTrue(self.decide(transfer=SimpleNamespace(kv_role=role)))

    def test_non_fa3_step5_keeps_exact_lengths(self):
        self.assertTrue(self.decide(fa3="0"))

    def test_fa3_alias_has_the_same_precedence_as_runtime(self):
        self.assertFalse(self.decide(fa3="0", alias="1"))
        self.assertTrue(self.decide(fa3="1", alias="0"))

    def test_other_models_keep_existing_opt_in_behavior(self):
        self.assertFalse(self.decide(model="other"))
        self.assertTrue(self.decide(model="other", exact="1"))

    def test_bad_switch_fails_instead_of_silently_changing_policy(self):
        with self.assertRaises(ValueError):
            self.decide(exact="invalid")

    def test_no_switch_keeps_non_pd_fa3_fast_default(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FA3": "1"}, clear=True):
            self.assertFalse(self.policy(self.config()))

    def test_initializer_uses_the_tested_policy(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "AscendMultiModuleMTPSpeculator")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__init__")
        self.assertTrue(any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_needs_exact_draft_seq_lens"
            for n in ast.walk(init)))


if __name__ == "__main__":
    unittest.main()
