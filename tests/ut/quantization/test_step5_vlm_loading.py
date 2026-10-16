# SPDX-License-Identifier: Apache-2.0
"""Interleaved VLM streams share one fused-weight completeness transaction."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
)
from vllm.models.step5.nvidia.model import Step5ForCausalLM, Step5FusedQKVIndexerLinear

from vllm_ascend.models.step5.vl import AscendStep5ForConditionalGeneration


class _TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the real tracking/reset/assert methods without creating an NPU
        # projection. The stream consumer stands in for tensor copies only.
        self.projection = object.__new__(Step5FusedQKVIndexerLinear)
        torch.nn.Module.__init__(self.projection)
        self.projection._loaded_shards = set()
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, _ in weights:
            self.projection._loaded_shards.add(name.removesuffix(".weight"))
        return {"projection.weight"}


def _vlm():
    model = object.__new__(AscendStep5ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    language = object.__new__(Step5ForCausalLM)
    torch.nn.Module.__init__(language)
    language.config = SimpleNamespace(tie_word_embeddings=False)
    language.model = _TextModel()
    language.lm_head = torch.nn.Linear(1, 1, bias=False)
    model.language_model = language
    model.vision_model = torch.nn.Linear(1, 1, bias=False)
    return model


def _stream(*, split_projection=False, omitted=None):
    shards = ("q", "k", "v", "index_q", "index_k", "index_z", "index_g")
    for index, shard in enumerate(shards):
        if shard != omitted:
            yield f"model.{shard}.weight", torch.ones(1, 1)
        if split_projection and index == 2:
            yield "vision_model.weight", torch.full((1, 1), 2.0)
    yield "vision_model.weight", torch.full((1, 1), 2.0)
    # This later language group caused a second reset and an all-shards-missing
    # error even when the first group contained the complete projection.
    yield "lm_head.weight", torch.full((1, 1), 3.0)


@pytest.mark.parametrize("split_projection", [False, True])
def test_interleaved_vlm_stream_validates_once_after_all_language_groups(split_projection):
    model = _vlm()
    loaded = model.load_weights(_stream(split_projection=split_projection))
    assert "language_model.model.projection.weight" in loaded
    assert "language_model.lm_head.weight" in loaded
    assert "vision_model.weight" in loaded
    assert model.language_model.model.calls == (2 if split_projection else 1)
    torch.testing.assert_close(model.language_model.lm_head.weight, torch.full((1, 1), 3.0))
    assert not model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()


@pytest.mark.parametrize("omitted", ["q", "index_q", "index_g"])
def test_incomplete_reload_still_fails_and_cannot_reuse_previous_shards(omitted):
    model = _vlm()
    model.load_weights(_stream())
    with pytest.raises(ValueError, match=f"missing shards .*{omitted}"):
        model.load_weights(_stream(split_projection=True, omitted=omitted))
    assert not model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()
    model.load_weights(_stream())


def test_stream_exception_clears_tracking_and_restores_validation_scope():
    model = _vlm()

    def broken_stream():
        yield "model.q.weight", torch.ones(1, 1)
        raise RuntimeError("checkpoint read failed")

    with pytest.raises(RuntimeError, match="checkpoint read failed"):
        model.load_weights(broken_stream())
    assert not model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()
    with pytest.raises(ValueError, match="missing shards"):
        model.load_weights(_stream(omitted="q"))


def test_explicitly_disabled_caller_keeps_its_scope_and_original_behavior():
    model = _vlm()
    with disable_mtp_completeness_check():
        model.load_weights(_stream(omitted="q"))
        assert not is_mtp_completeness_check_enabled()
        assert model.language_model.model.projection._loaded_shards
    assert is_mtp_completeness_check_enabled()
    # A following ordinary transaction resets that record and still rejects
    # a missing shard; externally deferred work cannot donate completeness.
    with pytest.raises(ValueError, match="missing shards .*q"):
        model.load_weights(_stream(omitted="q"))
    assert not model.language_model.model.projection._loaded_shards


def test_real_projection_copy_still_records_shards_while_validation_is_deferred():
    projection = _vlm().language_model.model.projection
    parameter = torch.nn.Parameter(torch.zeros(1, 2), requires_grad=False)
    source = torch.tensor([[2.0, 3.0]])
    with (
        disable_mtp_completeness_check(),
        patch.object(Step5FusedQKVIndexerLinear, "_indexer_source_offset", return_value=(0, 1)),
        patch.object(Step5FusedQKVIndexerLinear, "_indexer_dest_offset", return_value=0),
    ):
        projection.weight_loader_v2(parameter, source, "index_q")
    torch.testing.assert_close(parameter, source)
    assert projection._loaded_shards == {"index_q"}
    assert is_mtp_completeness_check_enabled()
