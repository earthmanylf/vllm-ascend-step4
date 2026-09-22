# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest

from test_torch_step4_block_sparse_attention import (
    build_cases,
    run_case,
    run_pd_shared,
    step4_block_sparse_attention,
)


def _case_id(cs) -> str:
    return cs.tag or "case"


@pytest.mark.npu
@pytest.mark.parametrize("cs", build_cases(), ids=_case_id)
def test_accuracy(npu_device, cs):
    print(f"\n=== {cs.tag} mode={cs.sparse_mode} qT={cs.q_t} kvT={cs.kv_t} ===", flush=True)
    ok = run_case(npu_device, cs, step4_block_sparse_attention, verbose=True)
    assert ok, cs.tag


@pytest.mark.npu
def test_pd_shared(npu_device):
    ok = run_pd_shared(npu_device, step4_block_sparse_attention, verbose=True)
    assert ok, "scene/pd_shared"
