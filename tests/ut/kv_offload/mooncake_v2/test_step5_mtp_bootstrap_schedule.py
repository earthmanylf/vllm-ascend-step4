# SPDX-License-Identifier: Apache-2.0
"""Characterize actual upstream first-decode padding with a constant MTP3 lookup."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS


def test_constant_three_token_schedule_keeps_first_resume_unpadded():
    root = Path(os.environ["VLLM_SOURCE_ROOT"])
    dynamic = root / "vllm/v1/spec_decode/dynamic/utils.py"
    ns = {}
    exec(compile(dynamic.read_text(), str(dynamic), "exec"), ns)
    lookup = ns["build_dynamic_sd_schedule_lookup"]([(1, 8, 3)], 8, 3)
    assert lookup[1:] == [3] * 8  # No reduction of the actual speculative depth.
    source = root / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(source.read_text())
    condition = next(
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "self.dynamic_sd_lookup is None" in ast.unparse(node.test)
        and "num_new_tokens == 1" in ast.unparse(node.test)
    )
    expr = compile(ast.Expression(condition), str(source), "eval")
    common = dict(num_new_tokens=1, num_computed_tokens=128, prefill_scheduled=False, scheduled_running_reqs=[])
    assert eval(expr, dict(common, self=NS(num_spec_tokens=3, dynamic_sd_lookup=None, num_sampled_tokens_per_step=1)))
    assert not eval(
        expr, dict(common, self=NS(num_spec_tokens=3, dynamic_sd_lookup=lookup, num_sampled_tokens_per_step=1))
    )


def test_dp_preserves_only_full_range_constant_static_k_schedule():
    source = Path(os.environ["VLLM_SOURCE_ROOT"]) / "vllm/config/vllm.py"
    tree = ast.parse(source.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_maybe_disable_dynamic_sd_for_data_parallel"
    )
    calls = []
    ns = {"logger": NS(warning_once=lambda *args: calls.append(args))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    for schedule, retained in (([(1, 256, 3)], True), ([(1, 128, 1), (129, 256, 3)], False), ([(1, 128, 3)], False)):
        spec = NS(
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=schedule,
            uses_dynamic_speculative_decoding=lambda: True,
        )
        cfg = NS(
            speculative_config=spec, parallel_config=NS(data_parallel_size=2), scheduler_config=NS(max_num_seqs=256)
        )
        ns[method.name](cfg)
        assert (spec.num_speculative_tokens_per_batch_size is not None) is retained
    assert len(calls) == 2
