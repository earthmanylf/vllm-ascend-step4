# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def loader_probe():
    # Import the real wrapper without importing unrelated optional NPU kernels.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/step5/vl.py"
    spec = importlib.util.spec_from_file_location("step5_vl_weight_test", path)
    module = importlib.util.module_from_spec(spec)
    upstream = SimpleNamespace(Step5ForConditionalGeneration=object)
    with patch.dict(
        "sys.modules",
        {
            "vllm.model_executor.models.step5_vl": upstream,
            "vllm.model_executor.models.utils": SimpleNamespace(AutoWeightsLoader=MagicMock()),
        },
    ):
        spec.loader.exec_module(module)
    batches = []

    class ConsumingLoader:
        def __init__(self, model):
            pass

        def load_weights(self, weights, mapper):
            batch = [(mapper._map_name(n), t) for n, t in weights]
            batches.append(batch)
            return {n for n, _ in batch}

    module.AutoWeightsLoader = ConsumingLoader
    model = module.AscendStep5ForConditionalGeneration()
    # Names here are already mapped; this isolates stream partitioning, not
    # upstream HF name translation. Every iterator is consumed exactly once.
    model.hf_to_vllm_mapper = SimpleNamespace(_map_name=lambda n: n)
    return model, batches


def test_step5_language_weights_in_one_stream(loader_probe):
    model, batches = loader_probe
    weights = [
        (name, torch.tensor(i))
        for i, name in enumerate(
            [
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "vision_model.weight",
                "language_model.model.layers.0.self_attn.k_proj.weight",
                "vit_large_projector.weight",
                "language_model.lm_head.weight",
            ]
        )
    ]
    for _ in range(2):
        batches.clear()
        loaded = model.load_weights(iter(weights))
        assert len(batches) == 2
        assert [n for n, _ in batches[0]] == [weights[i][0] for i in (0, 2, 4)]
        assert all(t is weights[i][1] for (_, t), i in zip(batches[0], (0, 2, 4), strict=True))
        assert loaded == {n for n, _ in weights}


@pytest.mark.parametrize("include_vision", [False, True])
def test_step5_vision_weights_loaded_after_language(loader_probe, include_vision):
    model, batches = loader_probe
    vision = [("vision_model.weight", torch.tensor(1)), ("vit_large_projector.weight", torch.tensor(2))]
    language = [("language_model.lm_head.weight", torch.tensor(3))]
    weights = (vision if include_vision else []) + language
    loaded = model.load_weights(iter(weights))
    assert [n for n, _ in batches[0]] == [language[0][0]]
    assert len(batches) == (2 if include_vision else 1)
    if include_vision:
        assert [n for n, _ in batches[1]] == [n for n, _ in vision]
        assert all(t is original for (_, t), (_, original) in zip(batches[1], vision, strict=True))
    assert loaded == {n for n, _ in weights}
