# SPDX-License-Identifier: Apache-2.0
"""Limited FA3 PD gates: real Step5 validators/specs, no device/network claim."""

import copy
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from tests.ut.attention.test_device_fa3_mrv2 import fa3  # noqa: F401
from tests.ut.worker.v2.test_step5_kv_contract import pd_upstream_fixture


@pytest.fixture
def pd_gate(fa3):  # noqa: F811 - imported pytest fixture
    upstream = pd_upstream_fixture()
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config_or_none = lambda: config_module.current
    config_module.current = None
    interface = ModuleType("vllm.v1.kv_cache_interface")
    interface.FullAttentionSpec = upstream.env["FullAttentionSpec"]
    layout = ModuleType("vllm.v1.kv_cache_layout")
    layout.KVCacheLayout = upstream.helper.KVCacheLayout
    fa3.AscendFlashAttentionImpl.__init__.__globals__.update(
        FullAttentionSpec=interface.FullAttentionSpec,
        uses_step5_blnhc=upstream.helper.uses_step5_blnhc,
        _validate_step5_blnhc_spec=upstream.helper._validate_step5_blnhc_spec,
    )
    with patch.dict(
        "sys.modules",
        {
            "vllm.config": config_module,
            "vllm.v1.kv_cache_interface": interface,
            "vllm.v1.kv_cache_layout": layout,
            "vllm_ascend.core.step5_kv_cache": upstream.helper,
        },
    ):
        yield NS(fa3=fa3, current=config_module, upstream=upstream)


def pd_config(gate, role="kv_producer", profile="bare", connector="MooncakeConnectorV2"):
    cfg = gate.upstream.cfg
    cfg.use_v2_model_runner = True
    cfg.model_config.hf_text_config = NS(model_type="step5", hidden_size=8192)
    cfg.model_config.dtype = torch.bfloat16
    cfg.model_config.get_head_size = lambda: 192
    cfg.model_config.get_num_kv_heads = lambda parallel: 1
    cfg.model_config.enforce_eager = profile == "eager_mtp3"
    cfg.cache_config.cache_dtype = "auto"
    cfg.cache_config.block_size = 128
    cfg.cache_config.get_resolved_kv_cache_layout = lambda: gate.upstream.helper.KVCacheLayout.BLNHC
    cfg.cache_config.enable_prefix_caching = True
    cfg.additional_config = {}
    cfg.kv_transfer_config = NS(
        is_kv_transfer_instance=True, kv_connector=connector, kv_connector_module_path=None, kv_role=role
    )
    cfg.parallel_config = NS(
        pipeline_parallel_size=1,
        tensor_parallel_size=4,
        data_parallel_size=4 if role == "kv_consumer" else 1,
        data_parallel_size_local=2 if role == "kv_consumer" else 1,
        enable_expert_parallel=True,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    cfg.scheduler_config = NS(
        disable_hybrid_kv_cache_manager=False, async_scheduling=True, max_num_seqs=256 if role == "kv_consumer" else 32
    )
    cfg.compilation_config = NS(mode=3, cudagraph_mode=NS(name="FULL_DECODE_ONLY"), cudagraph_capture_sizes=[4, 8, 512])
    cfg.speculative_config = None
    if profile != "bare":
        cfg.speculative_config = NS(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=profile == "eager_mtp3",
            num_speculative_tokens_per_batch_size=None,
            draft_model_config=NS(
                hf_text_config=NS(model_type="step5_mtp", num_nextn_predict_layers=3, hidden_size=8192)
            ),
        )
    if role == "kv_producer":
        cfg.parallel_config.pipeline_parallel_size = 4
    return cfg


def construct(gate, cfg, head_size=192, num_kv_heads=1):
    return gate.fa3.AscendFlashAttentionImpl(8, head_size, 0.125, num_kv_heads, vllm_config=cfg)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("profile", ["bare", "eager_mtp3", "graph_mtp3"])
def test_limited_step5_pd_capability_and_constructor(pd_gate, connector, role, profile):
    cfg = pd_config(pd_gate, role, profile, connector)
    pd_gate.current.current = cfg
    # This is the real shared config contract, not a mocked capability result.
    assert pd_gate.upstream.helper.uses_step5_blnhc(cfg)
    assert pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    construct(pd_gate, cfg)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("profile", ["bare", "eager_mtp3", "graph_mtp3"])
def test_limited_step5_pd_constructor(pd_gate, role, profile):
    construct(pd_gate, pd_config(pd_gate, role, profile))


def test_constructor_accepts_supported_argument_without_current_config(pd_gate):
    cfg = pd_config(pd_gate)
    assert pd_gate.current.current is None
    assert not pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    construct(pd_gate, cfg)


@pytest.mark.parametrize(
    "path,value",
    [
        ("model_config.hf_text_config.model_type", "llama"),
        ("use_v2_model_runner", False),
        ("kv_transfer_config.kv_connector", "MooncakeConnector"),
        ("kv_transfer_config.kv_connector_module_path", "custom.connector"),
        ("kv_transfer_config.kv_role", "kv_both"),
        ("kv_transfer_config.is_kv_transfer_instance", False),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.prefill_context_parallel_size", 2),
        ("parallel_config.decode_context_parallel_size", 2),
        ("scheduler_config.disable_hybrid_kv_cache_manager", True),
        ("additional_config", {"enable_kvpp": True}),
        ("additional_config", {"sparse_kv_offload_config": {"enabled": True}}),
        ("attention_config.hisparse_config", {}),
        ("cache_config.block_size", 64),
        ("cache_config.cache_dtype", "float16"),
        ("cache_config.cache_dtype", "fp8"),
        ("model_config.dtype", torch.float16),
        ("model_config.get_head_size", lambda: 128),
        ("model_config.get_num_kv_heads", lambda parallel: 2),
        ("cache_config.kv_cache_layout", "LBNHC"),
        ("speculative_config.method", "eagle"),
        ("speculative_config.num_speculative_tokens", 2),
        ("speculative_config.num_speculative_tokens_per_batch_size", {1: 3}),
        ("speculative_config.draft_model_config.hf_text_config.num_nextn_predict_layers", 2),
        ("scheduler_config.async_scheduling", False),
        ("scheduler_config.max_num_seqs", 16),
        ("compilation_config.cudagraph_capture_sizes", [4, 510]),
        ("compilation_config.cudagraph_mode.name", "NONE"),
        ("parallel_config.enable_expert_parallel", False),
    ],
)
def test_unsupported_config_rejected_by_both_gates(pd_gate, path, value):
    cfg = pd_config(pd_gate, profile="graph_mtp3")
    obj = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    pd_gate.current.current = cfg
    assert not pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    with pytest.raises(ValueError):
        construct(pd_gate, cfg)


@pytest.mark.parametrize("head_size,num_kv_heads", [(128, 1), (192, 2)])
def test_constructor_checks_actual_layer_geometry(pd_gate, head_size, num_kv_heads):
    cfg = pd_config(pd_gate)
    with pytest.raises(ValueError):
        construct(pd_gate, cfg, head_size, num_kv_heads)


def test_constructor_uses_argument_not_unrelated_current_config(pd_gate):
    cfg = pd_config(pd_gate)
    wrong = copy.deepcopy(cfg)
    wrong.kv_transfer_config.kv_connector = "MooncakeConnector"
    pd_gate.current.current = cfg
    assert pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    with pytest.raises(ValueError):
        construct(pd_gate, wrong)


def test_no_transfer_keeps_local_fa3_behavior(pd_gate):
    cfg = pd_config(pd_gate)
    cfg.kv_transfer_config = None
    cfg.model_config.hf_text_config.model_type = "llama"
    pd_gate.current.current = cfg
    assert not pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    construct(pd_gate, cfg, head_size=128, num_kv_heads=2)


@pytest.mark.parametrize("resolved_layout", [None, "BLNHC"])
@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
@pytest.mark.parametrize("prefix", [False, True])
def test_legal_graph_flags_and_layout_resolution_lifecycle(pd_gate, resolved_layout, cache_dtype, prefix):
    cfg = pd_config(pd_gate, "kv_consumer", "graph_mtp3")
    cfg.cache_config.kv_cache_layout = resolved_layout
    cfg.cache_config.cache_dtype = cache_dtype
    cfg.cache_config.enable_prefix_caching = prefix
    pd_gate.current.current = cfg
    assert pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    construct(pd_gate, cfg)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_bare_single_rank_profile_for_next_micro_chain(pd_gate, role):
    cfg = pd_config(pd_gate, role)
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.data_parallel_size = 1
    cfg.parallel_config.data_parallel_size_local = 1
    cfg.parallel_config.enable_expert_parallel = False
    pd_gate.current.current = cfg
    assert pd_gate.fa3.AscendFlashAttentionBackend.supports_kv_connector()
    construct(pd_gate, cfg)


@pytest.mark.parametrize("spec_name", ["FullAttentionSpec", "SlidingWindowSpec"])
def test_pd_builder_accepts_real_main_specs(pd_gate, spec_name):
    cfg = pd_config(pd_gate, "kv_consumer", "graph_mtp3")
    spec_cls = pd_gate.upstream.env[spec_name]
    extra = {"sliding_window": 512} if spec_name == "SlidingWindowSpec" else {}
    spec = spec_cls(block_size=128, num_kv_heads=1, head_size=192, head_size_v=192, dtype=torch.bfloat16, **extra)
    pd_gate.fa3.AscendFlashAttentionMetadataBuilder(spec, ["layer"], cfg, torch.device("cpu"))


@pytest.mark.parametrize("invalid", ["head_size_v", "dtype", "block_size", "heads", "padded", "type"])
def test_pd_builder_rejects_unsupported_actual_spec(pd_gate, invalid):
    cfg = pd_config(pd_gate)
    spec_cls = pd_gate.upstream.env["FullAttentionSpec"]
    spec = spec_cls(block_size=128, num_kv_heads=1, head_size=192, head_size_v=192, dtype=torch.bfloat16)
    changes = {
        "head_size_v": {"head_size_v": 128},
        "dtype": {"dtype": torch.float16},
        "block_size": {"block_size": 64},
        "heads": {"num_kv_heads": 2},
        "padded": {"page_size_padded": 131072},
    }
    if invalid == "type":
        spec = NS(**vars(spec))
    else:
        spec = replace(spec, **changes[invalid])
    with pytest.raises(ValueError):
        pd_gate.fa3.AscendFlashAttentionMetadataBuilder(spec, ["layer"], cfg, torch.device("cpu"))
