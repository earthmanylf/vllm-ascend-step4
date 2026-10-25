# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.attention.attention_v1 as attn_module
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AttentionType,
    FIAParamProvider,
)


def _make_impl(impl_cls=AscendAttentionBackendImpl):
    impl = impl_cls.__new__(impl_cls)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl.scale = 0.5
    impl.attn_type = AttentionType.DECODER
    impl.use_bnsd_kv_cache = False
    impl.key_cache = None
    impl.value_cache = None
    impl.sliding_window = None
    impl.sinks = None
    impl.enable_c8_quant = False
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "layer"
    return impl


def _make_metadata(state=AscendAttentionState.PrefillNoCache):
    return SimpleNamespace(
        attn_state=state,
        actual_seq_lengths_q=[3],
        seq_lens=torch.tensor([3], dtype=torch.int32),
        seq_lens_list=[3],
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        num_actual_tokens=3,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        causal=True,
        attn_mask=None,
    )


def _strided_kv(num_tokens=3, offset=0):
    backing = torch.arange(num_tokens * 8, dtype=torch.float32) + offset
    return backing.view(num_tokens, 1, 8)[..., ::2]


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("cross_attention", [False, True])
def test_non_pa_params_preserve_current_kv_inputs(contiguous, cross_attention):
    impl = _make_impl()
    metadata = _make_metadata()
    if cross_attention:
        impl.attn_type = AttentionType.ENCODER_DECODER
        metadata.seq_lens = torch.tensor([5], dtype=torch.int32)
    key = _strided_kv(5 if cross_attention else 3)
    value = _strided_kv(key.shape[0], offset=100)
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    if contiguous:
        key = key.contiguous()
        value = value.contiguous()

    actual_key, actual_value, _, block_table, seq_lens = impl._get_fia_params(key, value, metadata)

    assert block_table is None
    assert seq_lens == ([5] if cross_attention else [3])
    for actual, original in ((actual_key, key), (actual_value, value)):
        assert actual is original
        assert actual.is_contiguous() == contiguous


@pytest.mark.parametrize("use_bnsd", [False, True])
@pytest.mark.parametrize(
    "state",
    [
        AscendAttentionState.PrefillCacheHit,
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.SpecDecoding,
    ],
)
def test_pa_params_preserve_cache_storage_and_stride(use_bnsd, state):
    impl = _make_impl()
    impl.use_bnsd_kv_cache = use_bnsd
    shape = (2, 2, 1, 3, 4) if use_bnsd else (2, 2, 3, 1, 4)
    backing = torch.arange(48, dtype=torch.float32).view(shape)
    impl.key_cache, impl.value_cache = backing[:, 0], backing[:, 1]
    metadata = _make_metadata(state)

    key, value, block_size, block_table, _ = impl._get_fia_params(_strided_kv(), _strided_kv(), metadata)

    assert block_size == 3
    assert block_table.data_ptr() == metadata.block_tables.data_ptr()
    for actual, original in ((key, impl.key_cache), (value, impl.value_cache)):
        assert not actual.is_contiguous()
        assert actual.data_ptr() == original.data_ptr()
        assert actual.storage_offset() == original.storage_offset()
        assert actual.stride(0) == original.stride(0)
        torch.testing.assert_close(actual.reshape(original.shape), original)


@pytest.mark.parametrize("branch", ["causal", "non_causal", "sliding_window"])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("cross_attention", [False, True])
def test_forward_non_pa_branches_receive_contiguous_kv(branch, contiguous, cross_attention):
    impl = _make_impl()
    metadata = _make_metadata()
    metadata.causal = branch != "non_causal"
    impl.sliding_window = 16 if branch == "sliding_window" else None
    if cross_attention:
        impl.attn_type = AttentionType.ENCODER_DECODER
        metadata.seq_lens = torch.tensor([5], dtype=torch.int32)
    num_kv_tokens = 5 if cross_attention else 4
    key, value = _strided_kv(num_kv_tokens), _strided_kv(num_kv_tokens, offset=100)
    if contiguous:
        key, value = key.contiguous(), value.contiguous()
    query = torch.zeros(3, 2, 4)
    output = torch.empty_like(query)

    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch.object(attn_module.envs_vllm, "VLLM_BATCH_INVARIANT", False),
        patch.object(attn_module, "get_current_hardware_profile", return_value=Mock(supports=lambda _: False)),
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", create=True) as direct_fia,
        patch.object(attn_module.DeviceOperator, "npu_fused_infer_attention_score") as device_fia,
    ):
        selected_fia = device_fia if branch == "causal" else direct_fia
        selected_fia.return_value = (torch.ones_like(query), None)
        result = impl.forward_fused_infer_attention(query, key, value, metadata, output)

    selected_fia.assert_called_once()
    kwargs = selected_fia.call_args.kwargs
    assert kwargs["block_table"] is None
    assert kwargs["actual_seq_lengths_kv"] == ([5] if cross_attention else [3])
    for name, original in (("key", key), ("value", value)):
        assert kwargs[name].is_contiguous()
        torch.testing.assert_close(kwargs[name], original if cross_attention else original[:3])
        assert (kwargs[name].data_ptr() == original.data_ptr()) == contiguous
    assert result is output
    torch.testing.assert_close(output, torch.ones_like(query))


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("use_pa", [False, True])
def test_graph_workspace_and_task_preserve_fia_layout_contract(v2, use_pa):
    impl = _make_impl()
    metadata = _make_metadata(AscendAttentionState.DecodeOnly if use_pa else AscendAttentionState.PrefillNoCache)
    query = torch.zeros(3, 2, 4)
    key, value = _strided_kv(), _strided_kv()
    if use_pa:
        backing = torch.arange(48, dtype=torch.float32).view(2, 2, 3, 1, 4)
        impl.key_cache, impl.value_cache = backing[:, 0], backing[:, 1]
    output = torch.empty_like(query)
    workspace_name = (
        "_npu_fused_infer_attention_score_v2_get_max_workspace"
        if v2
        else "_npu_fused_infer_attention_score_get_max_workspace"
    )
    op_name = "npu_fused_infer_attention_score_v2" if v2 else "npu_fused_infer_attention_score"
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
        patch.object(attn_module.torch_npu, op_name, create=True),
        patch.object(attn_module.torch_npu, workspace_name, create=True) as get_workspace,
        patch.object(attn_module, "get_capture_resource", side_effect=lambda _, factory, *args: factory()),
        patch.object(attn_module, "register_task") as register,
    ):
        graph_fia = impl.full_graph_fia_v2 if v2 else impl.full_graph_fia
        result, num_tokens = graph_fia(query, key, value, metadata, output)

    get_workspace.assert_called_once()
    register.assert_called_once()
    workspace_kwargs = get_workspace.call_args.kwargs
    task_kwargs = register.call_args.args[1]
    expected_key = impl.key_cache.view(2, 3, 4) if use_pa else key
    expected_value = impl.value_cache.view(2, 3, 4) if use_pa else value
    for kwargs in (workspace_kwargs, task_kwargs):
        assert kwargs["block_table"] is (metadata.block_tables if use_pa else None)
        for name, original in (("key", expected_key), ("value", expected_value)):
            assert kwargs[name].is_contiguous() == (not use_pa)
            torch.testing.assert_close(kwargs[name], original)
            if use_pa:
                assert kwargs[name].data_ptr() == original.data_ptr()
                assert kwargs[name].stride() == original.stride()
    assert task_kwargs["key"] is workspace_kwargs["key"]
    assert task_kwargs["value"] is workspace_kwargs["value"]
    if not v2:
        runtime_metadata = _make_metadata(
            AscendAttentionState.PrefillNoCache if use_pa else AscendAttentionState.DecodeOnly
        )
        provider = register.call_args.args[2]
        runtime_kwargs = {**task_kwargs, **provider.resolve({"layer": runtime_metadata})}
        assert runtime_kwargs["block_table"] is (runtime_metadata.block_tables if use_pa else None)
        assert runtime_kwargs["key"] is task_kwargs["key"]
        assert runtime_kwargs["value"] is task_kwargs["value"]
    assert result is output
    assert num_tokens == 3


@pytest.mark.parametrize("continuing_prefill", [False, True])
def test_c8_chunked_non_pa_prefill_receives_contiguous_kv(continuing_prefill):
    impl = _make_impl(AscendC8AttentionBackendImpl)
    impl.key_cache = torch.empty(2, 32, 1, 4, dtype=torch.int8)
    impl.value_cache = torch.empty_like(impl.key_cache)
    impl._nz_5d_view = Mock(side_effect=lambda cache, _: cache)
    dense_key, dense_value = _strided_kv(5).contiguous(), _strided_kv(5).contiguous()
    impl._dequant_paged_kv_to_dense = Mock(return_value=(dense_key, dense_value))
    metadata = _make_metadata(AscendAttentionState.ChunkedPrefill)
    if continuing_prefill:
        metadata.seq_lens_list = [5]
    key, value = _strided_kv(), _strided_kv()
    query = torch.zeros(3, 2, 4)
    output = torch.empty_like(query)
    with patch.object(
        attn_module.torch_npu,
        "npu_fused_infer_attention_score",
        create=True,
        return_value=(torch.ones_like(query), None),
    ) as fia:
        result = impl._forward_c8_chunked_prefill(query, key, value, metadata, output, SimpleNamespace())

    fia.assert_called_once()
    kwargs = fia.call_args.kwargs
    assert kwargs["block_table"] is None
    expected_key, expected_value = (dense_key, dense_value) if continuing_prefill else (key, value)
    for name, original in (("key", expected_key), ("value", expected_value)):
        assert kwargs[name].is_contiguous()
        torch.testing.assert_close(kwargs[name], original)
    assert impl._dequant_paged_kv_to_dense.call_count == int(continuing_prefill)
    assert result is output
    torch.testing.assert_close(output, torch.ones_like(query))


@pytest.mark.parametrize("contiguous", [False, True])
def test_c8_fused_non_pa_prefill_receives_contiguous_kv(contiguous):
    impl = _make_impl(AscendC8AttentionBackendImpl)
    metadata = _make_metadata()
    query = torch.zeros(3, 2, 4)
    key, value = _strided_kv(4), _strided_kv(4, offset=100)
    if contiguous:
        key, value = key.contiguous(), value.contiguous()
    output = torch.empty_like(query)

    with patch.object(
        attn_module.torch_npu,
        "npu_fused_infer_attention_score",
        create=True,
        return_value=(torch.ones_like(query), None),
    ) as fia:
        result = impl._forward_c8_fused_infer_attention(query, key, value, metadata, output, SimpleNamespace())

    fia.assert_called_once()
    kwargs = fia.call_args.kwargs
    assert kwargs["block_table"] is None
    for name, original in (("key", key), ("value", value)):
        assert kwargs[name].is_contiguous()
        torch.testing.assert_close(kwargs[name], original[:3])
        assert (kwargs[name].data_ptr() == original.data_ptr()) == contiguous
    assert result is output
    torch.testing.assert_close(output, torch.ones_like(query))


@pytest.mark.parametrize("sliding_window,is_draft_model", [(None, False), (16, True), (16, False)])
@pytest.mark.parametrize("state", [AscendAttentionState.PrefillNoCache, AscendAttentionState.DecodeOnly])
@pytest.mark.parametrize("is_non_pa", [False, True])
def test_graph_provider_preserves_captured_pa_mode(sliding_window, is_draft_model, state, is_non_pa):
    metadata = _make_metadata(state)
    provider = FIAParamProvider("layer", sliding_window, is_draft_model, is_non_pa=is_non_pa)
    captured_block_table = None if is_non_pa else torch.tensor([[1]], dtype=torch.int32)

    params = provider.resolve({"layer": metadata})

    assert params["actual_seq_lengths"] is metadata.actual_seq_lengths_q
    assert params["actual_seq_lengths_kv"] is metadata.seq_lens_list
    runtime_kwargs = {"block_table": captured_block_table, **params}
    if sliding_window and not is_draft_model:
        assert "block_table" not in params
        assert runtime_kwargs["block_table"] is captured_block_table
    else:
        assert runtime_kwargs["block_table"] is (None if is_non_pa else metadata.block_tables)


@pytest.mark.parametrize("window", [1, 128, 512, None])
@pytest.mark.parametrize("qlen", [1, 4])
@pytest.mark.parametrize("use_pa", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_fia_v1_window_counts_current_token(window, qlen, use_pa, graph):
    impl = _make_impl()
    impl.sliding_window = window
    metadata = _make_metadata(AscendAttentionState.DecodeOnly if use_pa else AscendAttentionState.PrefillNoCache)
    metadata.actual_seq_lengths_q = [qlen]
    metadata.num_actual_tokens = qlen
    metadata.seq_lens = torch.tensor([qlen], dtype=torch.int32)
    metadata.seq_lens_list = [qlen]
    query = torch.zeros(qlen, 2, 4)
    key, value = _strided_kv(qlen), _strided_kv(qlen, offset=100)
    if use_pa:
        backing = torch.arange(2 * 2 * qlen * 4, dtype=torch.float32).view(2, 2, qlen, 1, 4)
        impl.key_cache, impl.value_cache = backing[:, 0], backing[:, 1]
    output = torch.empty_like(query)
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(capturing=False, is_draft_model=False)),
        patch.object(attn_module.envs_vllm, "VLLM_BATCH_INVARIANT", False),
        patch.object(attn_module, "get_current_hardware_profile", return_value=Mock(supports=lambda _: False)),
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", create=True) as direct_fia,
        patch.object(attn_module.DeviceOperator, "npu_fused_infer_attention_score") as device_fia,
        patch.object(
            attn_module.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", create=True
        ) as get_workspace,
        patch.object(attn_module, "get_capture_resource", side_effect=lambda _, factory, *args: factory()),
        patch.object(attn_module, "register_task") as register,
    ):
        selected_fia = direct_fia if window is not None else device_fia
        selected_fia.return_value = (torch.ones_like(query), None)
        if graph:
            impl.full_graph_fia(query, key, value, metadata, output)
            kwargs_list = [get_workspace.call_args.kwargs, register.call_args.args[1]]
        else:
            impl.forward_fused_infer_attention(query, key, value, metadata, output)
            kwargs_list = [selected_fia.call_args.kwargs]

    for kwargs in kwargs_list:
        assert kwargs["sparse_mode"] == (4 if window is not None else 3)
        if window is None and not graph:
            # The non-SWA DeviceOperator call retains its previous defaults.
            assert "pre_tokens" not in kwargs
        else:
            assert kwargs["pre_tokens"] == (window - 1 if window is not None else attn_module.SWA_INT_MAX)
        if window is not None:
            assert kwargs["next_tokens"] == 0
        elif graph:
            assert kwargs["next_tokens"] == attn_module.SWA_INT_MAX
        assert kwargs["block_table"] is (metadata.block_tables if use_pa else None)
    if graph:
        assert kwargs_list[0]["pre_tokens"] == kwargs_list[1]["pre_tokens"]
        assert kwargs_list[0]["key"] is kwargs_list[1]["key"]
        assert kwargs_list[0]["value"] is kwargs_list[1]["value"]
    assert impl.sliding_window is window


@pytest.mark.parametrize("window", [0, -1])
@pytest.mark.parametrize("graph", [False, True])
def test_fia_v1_rejects_nonpositive_window(window, graph):
    impl = _make_impl()
    impl.sliding_window = window
    query = torch.zeros(3, 2, 4)
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", create=True),
        pytest.raises(ValueError, match="sliding_window must be positive or None"),
    ):
        method = impl.full_graph_fia if graph else impl.forward_fused_infer_attention
        method(query, _strided_kv(), _strided_kv(), _make_metadata(), torch.empty_like(query))


@pytest.mark.parametrize("window", [1, 512, None])
@pytest.mark.parametrize("graph", [False, True])
def test_fia_v2_sink_window_is_not_reinterpreted(window, graph):
    impl = _make_impl()
    impl.sliding_window = window
    impl.sinks = torch.zeros(2)
    query = torch.zeros(3, 2, 4)
    metadata = _make_metadata()
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score_v2", create=True) as fia_v2,
        patch.object(
            attn_module.torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", create=True
        ) as get_workspace,
        patch.object(attn_module, "get_capture_resource", side_effect=lambda _, factory, *args: factory()),
        patch.object(attn_module, "register_task") as register,
    ):
        fia_v2.return_value = (torch.ones_like(query), None)
        if graph:
            impl.full_graph_fia_v2(query, _strided_kv(), _strided_kv(), metadata, torch.empty_like(query))
            kwargs_list = [get_workspace.call_args.kwargs, register.call_args.args[1]]
        else:
            impl.forward_fused_infer_attention(query, _strided_kv(), _strided_kv(), metadata, torch.empty_like(query))
            kwargs_list = [fia_v2.call_args.kwargs]
    for kwargs in kwargs_list:
        assert kwargs["pre_tokens"] == (window if window is not None else attn_module.SWA_INT_MAX)
        assert kwargs["sparse_mode"] == (4 if window is not None else 3)
        assert kwargs["learnable_sink"] is impl.sinks


def test_fia_v1_boundary_sentinel_excludes_extra_historical_token():
    impl = _make_impl()
    impl.sliding_window = 512
    query = torch.zeros(1, 2, 4)
    metadata = _make_metadata()
    metadata.actual_seq_lengths_q = [1]
    metadata.num_actual_tokens = 1
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch.object(
            attn_module.torch_npu,
            "npu_fused_infer_attention_score",
            create=True,
            return_value=(torch.ones_like(query), None),
        ) as fia,
    ):
        impl.forward_fused_infer_attention(query, _strided_kv(1), _strided_kv(1), metadata, torch.empty_like(query))
    left = fia.call_args.kwargs["pre_tokens"]
    target, draft, excluded = 131073, 131071, 130559
    first_kept_page = (target - (512 - 1 + 2)) // 128
    actual_positions = range(max(draft - left, 0), draft + 1)
    assert len(actual_positions) == 512
    assert excluded not in actual_positions
    assert all(position // 128 >= first_kept_page for position in actual_positions)
    # Included-left and current sentinels remain visible with the upstream budget.
    assert excluded + 1 in actual_positions
    assert draft in actual_positions
    # Independent original-adapter counterexample: uniform QK => sentinel weight 1/513.
    old_positions = range(draft - 512, draft + 1)
    assert excluded in old_positions
    assert excluded // 128 < first_kept_page
    assert 1 / len(old_positions) > 0


@pytest.mark.parametrize("window", [512, None])
def test_fia_v1_c8_graph_keeps_specialized_mode_zero_contract(window):
    impl = _make_impl(AscendC8AttentionBackendImpl)
    impl.sliding_window = window
    impl.enable_c8_quant = True
    impl.key_cache = torch.empty(2, 3, 1, 4, dtype=torch.int8)
    impl.value_cache = torch.empty_like(impl.key_cache)
    impl._nz_5d_view = Mock(side_effect=lambda cache, _: cache)
    layer = SimpleNamespace(
        _c8_k_aq_scale_nz_bnsd=torch.ones(1),
        _c8_v_aq_scale_nz_bnsd=torch.ones(1),
    )
    metadata = _make_metadata(AscendAttentionState.DecodeOnly)
    query = torch.zeros(3, 2, 4)
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", create=True),
        patch.object(
            attn_module.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", create=True
        ) as get_workspace,
        patch.object(attn_module, "get_capture_resource", side_effect=lambda _, factory, *args: factory()),
        patch.object(attn_module, "register_task") as register,
    ):
        impl.full_graph_fia(query, _strided_kv(), _strided_kv(), metadata, torch.empty_like(query), layer)
    for kwargs in (get_workspace.call_args.kwargs, register.call_args.args[1]):
        assert kwargs["input_layout"] == "BNSD"
        assert kwargs["sparse_mode"] == 0
        assert kwargs["atten_mask"] is None
        assert kwargs["pre_tokens"] == (window if window is not None else attn_module.SWA_INT_MAX)
        assert kwargs["inner_precise"] == 1
