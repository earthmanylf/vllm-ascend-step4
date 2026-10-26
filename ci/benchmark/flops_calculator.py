# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
from pathlib import Path


def get_gpu_type():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"])
        gpu_types = output.decode("utf-8").strip().split("\n")
        return gpu_types
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_gpu_flops():
    gpu_types = get_gpu_type()
    if len(gpu_types) == 0:
        raise ValueError("No GPU found.")
    gpu_type = gpu_types[0]

    if "A800" in gpu_type or "A100" in gpu_type:
        return 312
    elif "L40S" in gpu_type:
        return 362
    elif "4090" in gpu_type:
        return 165
    elif "L20" in gpu_type:
        return 119
    elif "H800" in gpu_type or "H200" in gpu_type or "H100" in gpu_type:
        return 989
    elif "H20" in gpu_type:
        return 148
    else:
        raise ValueError(f"Unsupported GPU type: {gpu_type}")


def _resolve_num_kv_heads(
    transformer_config,
    num_heads,
    is_step2_mini,
    use_attention_groups_as_kv_heads=False,
):
    if "num_key_value_heads" in transformer_config and transformer_config["num_key_value_heads"] is not None:
        num_kv_heads = transformer_config["num_key_value_heads"]
    elif "num_attention_groups" in transformer_config and transformer_config["num_attention_groups"] > 0:
        if use_attention_groups_as_kv_heads:
            num_kv_heads = transformer_config["num_attention_groups"]
        else:
            num_kv_heads = num_heads // transformer_config["num_attention_groups"]
    else:
        num_kv_heads = num_heads

    if is_step2_mini:
        return 1
    return num_kv_heads


def _resolve_sliding_window(transformer_config):
    sliding_window = transformer_config.get("sliding_window")
    if sliding_window is None:
        sliding_window = transformer_config.get("sliding_window_size")
    if isinstance(sliding_window, list):
        for value in sliding_window:
            if value:
                return value
        return None
    return sliding_window


def _resolve_layer_types(transformer_config, num_layers, sliding_window):
    layer_types = transformer_config.get("layer_types")
    if layer_types:
        resolved = list(layer_types)[:num_layers]
        if len(resolved) < num_layers:
            resolved.extend(["full_attention"] * (num_layers - len(resolved)))
        return resolved

    return ["full_attention"] * num_layers


def _resolve_swa_num_heads(transformer_config, num_heads):
    sliding_num_heads = transformer_config.get("num_sliding_attention_heads")
    if sliding_num_heads is None:
        sliding_num_heads = transformer_config.get("swa_num_attention_heads")
    if sliding_num_heads is None or sliding_num_heads <= 0:
        return num_heads
    return sliding_num_heads


def _resolve_effective_num_layers(transformer_config):
    num_layers = transformer_config["num_hidden_layers"]
    if transformer_config.get("model_type") != "step3p5":
        return num_layers

    layer_types = transformer_config.get("layer_types") or []
    extra_output_layers = max(int(transformer_config.get("num_nextn_predict_layers", 0) or 0), 0)
    return max(num_layers, num_layers + extra_output_layers, len(layer_types))


def _resolve_ffn_num_layers(transformer_config, effective_num_layers):
    num_layers = transformer_config["num_hidden_layers"]
    if transformer_config.get("model_type") != "step3p5":
        return num_layers

    layer_types = transformer_config.get("layer_types") or []
    return min(max(num_layers, len(layer_types)), effective_num_layers)


def _resolve_use_headwise_attn_gate(transformer_config):
    return bool(
        transformer_config.get("use_headwise_attn_gate")
        or transformer_config.get("use_head_wise_attn_gate")
        or transformer_config.get("attention_gate")
    )


def _resolve_count_router_flops(transformer_config):
    if transformer_config.get("model_type") == "step3p5":
        return bool(transformer_config.get("count_router_flops", False))
    return bool(transformer_config.get("count_router_flops", True))


def _resolve_num_output_heads(transformer_config):
    nextn_predict_layers = max(int(transformer_config.get("num_nextn_predict_layers", 0) or 0), 0)
    num_mtp_layers = max(int(transformer_config.get("num_mtp_layers", 0) or 0), 0)
    return 1 + max(nextn_predict_layers, num_mtp_layers)


def _resolve_moe_config(transformer_config):
    use_moe = bool(transformer_config.get("use_moe", False))
    moe_intermediate_size = transformer_config.get("moe_intermediate_size")
    num_experts = transformer_config.get("moe_num_experts")
    if num_experts is None:
        num_experts = transformer_config.get("num_experts")
    if num_experts is None:
        num_experts = transformer_config.get("n_routed_experts")

    top_k = transformer_config.get("moe_top_k")
    if top_k is None:
        top_k = transformer_config.get("num_experts_per_tok")

    if moe_intermediate_size and num_experts and top_k:
        use_moe = True

    return use_moe, moe_intermediate_size, num_experts, top_k


def _resolve_num_moe_layers(transformer_config, total_layers):
    moe_layer_indices = set()
    num_moe_layers = 0

    if "moe_layers_enum" in transformer_config and transformer_config["moe_layers_enum"]:
        try:
            moe_layer_indices = set(int(i) for i in transformer_config["moe_layers_enum"].split(","))
            num_moe_layers = len(moe_layer_indices)
        except ValueError:
            num_moe_layers = 0

    if num_moe_layers == 0:
        moe_every_n = transformer_config.get("moe_every_n_layer")
        if moe_every_n is None:
            moe_every_n = transformer_config.get("decoder_sparse_step", -1)

        if moe_every_n > 0:
            moe_layer_offset = transformer_config.get("moe_layer_offset", 0)
            num_moe_layers = (total_layers - moe_layer_offset + moe_every_n - 1) // moe_every_n
        elif moe_every_n == -1 and not moe_layer_indices:
            num_moe_layers = total_layers

    num_moe_layers = max(0, min(num_moe_layers, total_layers))
    return num_moe_layers, total_layers - num_moe_layers


def _resolve_attention_other_setting(transformer_config, layer_type):
    attention_other_setting = transformer_config.get("attention_other_setting") or {}
    if not attention_other_setting:
        return None
    if layer_type != attention_other_setting.get("attention_type"):
        return None
    return attention_other_setting


def _resolve_layer_attention_spec(
    transformer_config,
    layer_type,
    num_heads,
    head_dim,
    num_kv_heads,
    sliding_window,
):
    attention_other_setting = _resolve_attention_other_setting(transformer_config, layer_type)
    q_heads = num_heads
    layer_head_dim = head_dim
    layer_num_kv_heads = num_kv_heads
    if attention_other_setting is not None:
        q_heads = attention_other_setting.get("num_attention_heads", q_heads)
        layer_head_dim = attention_other_setting.get("head_dim", layer_head_dim)
        layer_num_kv_heads = attention_other_setting.get("num_attention_groups", layer_num_kv_heads)
    elif layer_type == "sliding_attention":
        q_heads = _resolve_swa_num_heads(transformer_config, num_heads)

    q_attn_dim = q_heads * layer_head_dim
    kv_attn_dim = layer_num_kv_heads * layer_head_dim
    extra_qkv_dim = q_heads if _resolve_use_headwise_attn_gate(transformer_config) else 0
    layer_sliding_window = None
    if layer_type == "sliding_attention" and sliding_window is not None:
        layer_sliding_window = sliding_window

    return {
        "q_attn_dim": q_attn_dim,
        "kv_attn_dim": kv_attn_dim,
        "extra_qkv_dim": extra_qkv_dim,
        "sliding_window": layer_sliding_window,
    }


def _build_layer_attention_specs(
    transformer_config,
    num_layers,
    num_heads,
    head_dim,
    num_kv_heads,
):
    sliding_window = _resolve_sliding_window(transformer_config)
    layer_types = _resolve_layer_types(transformer_config, num_layers, sliding_window)
    return [
        _resolve_layer_attention_spec(
            transformer_config=transformer_config,
            layer_type=layer_type,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            sliding_window=sliding_window,
        )
        for layer_type in layer_types
    ]


def _causal_sliding_qk_sv_flops(seq_len, sliding_window, q_attn_dim):
    if sliding_window is None or sliding_window <= 0:
        return 2 * seq_len * seq_len * q_attn_dim

    local_window = min(seq_len, sliding_window)
    visible_pairs = 0.5 * local_window * (local_window + 1)
    if seq_len > local_window:
        visible_pairs += (seq_len - local_window) * local_window
    return 4 * visible_pairs * q_attn_dim


def _standard_prefill_attn_flops(
    seq_len,
    hidden_size,
    q_attn_dim,
    kv_attn_dim,
    sliding_window=None,
    extra_qkv_dim=0,
):
    qkv_flops = 2 * seq_len * hidden_size * (q_attn_dim + 2 * kv_attn_dim + extra_qkv_dim)
    if sliding_window is None:
        qk_sv_flops = 2 * seq_len * seq_len * q_attn_dim
    else:
        qk_sv_flops = _causal_sliding_qk_sv_flops(seq_len, sliding_window, q_attn_dim)
    o_proj_flops = 2 * seq_len * q_attn_dim * hidden_size
    return qkv_flops + qk_sv_flops + o_proj_flops


def _standard_decode_attn_flops(
    seq_len,
    context_length,
    hidden_size,
    q_attn_dim,
    kv_attn_dim,
    sliding_window=None,
    extra_qkv_dim=0,
):
    qkv_flops = 2 * seq_len * hidden_size * (q_attn_dim + 2 * kv_attn_dim + extra_qkv_dim)
    effective_context = context_length
    if sliding_window is not None and sliding_window > 0:
        effective_context = min(context_length, sliding_window)
    qk_sv_flops = 4 * seq_len * effective_context * q_attn_dim
    o_proj_flops = 2 * seq_len * q_attn_dim * hidden_size
    return qkv_flops + qk_sv_flops + o_proj_flops


def _step3p5_dense_mlp_flops(seq_len, hidden_size, intermediate_size):
    return 6 * seq_len * hidden_size * intermediate_size + seq_len * intermediate_size


def _step3p5_moe_mlp_flops(
    seq_len,
    hidden_size,
    num_experts,
    top_k,
    moe_intermediate_size,
    share_expert_dim=0,
    count_router_flops=False,
):
    router_flops = 0
    if count_router_flops:
        router_flops = 2 * seq_len * hidden_size * num_experts
    expert_flops = top_k * (6 * seq_len * hidden_size * moe_intermediate_size + seq_len * moe_intermediate_size)
    shared_expert_flops = 0
    if share_expert_dim > 0:
        shared_expert_flops = 6 * seq_len * hidden_size * share_expert_dim + seq_len * share_expert_dim
    return router_flops + expert_flops + shared_expert_flops


def _step3p5_embedding_flops(seq_len, hidden_size):
    return 2 * seq_len * hidden_size


def _step3p5_lm_head_flops(seq_len, hidden_size, vocab_size, num_output_heads):
    return 2 * seq_len * hidden_size * vocab_size * num_output_heads


def sequence_flops(
    transformer_config,
    input_length,
    output_length,
    test_name,
):
    """
    Calculates the theoretical TFLOPs for a single sequence pass (prefill or decode)
    through a transformer model, considering potential MoE layers and specific
    attention mechanisms like Step2MiniAttention.

    Args:
        transformer_config: Dictionary containing model configuration parameters.
        input_length: The length of the input sequence.
        output_length: The number of tokens to generate (used for decode).
        test_name: Either "prefill" or "decode".

    Returns:
        Total TFLOPs for the specified phase.
    """
    hidden_size = transformer_config["hidden_size"]
    num_layers = _resolve_effective_num_layers(transformer_config)
    num_heads = transformer_config["num_attention_heads"]
    head_dim = transformer_config.get("head_dim", hidden_size // num_heads)  # Use head_dim if present

    # Step2Mini uses num_kv_heads=1 in the implementation.
    is_step2_mini = transformer_config.get("model_type") == "step2_mini"
    is_step3p5 = transformer_config.get("model_type") == "step3p5"
    share_q_dim = transformer_config.get("share_q_dim") if is_step2_mini else None

    num_kv_heads = _resolve_num_kv_heads(
        transformer_config,
        num_heads,
        is_step2_mini,
        use_attention_groups_as_kv_heads=is_step3p5,
    )

    q_attn_dim = num_heads * head_dim
    kv_attn_dim = num_kv_heads * head_dim
    layer_attention_specs = None
    if is_step3p5:
        layer_attention_specs = _build_layer_attention_specs(
            transformer_config=transformer_config,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

    use_moe, moe_intermediate_size, num_experts, top_k = _resolve_moe_config(transformer_config)

    num_moe_layers = 0
    ffn_num_layers = _resolve_ffn_num_layers(transformer_config, num_layers)
    num_dense_layers = ffn_num_layers
    dense_intermediate_size = transformer_config.get("intermediate_size")
    share_expert_dim = transformer_config.get("share_expert_dim", 0) if (is_step2_mini or is_step3p5) else 0
    count_router_flops = _resolve_count_router_flops(transformer_config)
    num_output_heads = _resolve_num_output_heads(transformer_config)

    if use_moe:
        num_moe_layers, num_dense_layers = _resolve_num_moe_layers(transformer_config, ffn_num_layers)

        if num_dense_layers > 0 and dense_intermediate_size is None:
            raise ValueError("Model has dense MLP layers but 'intermediate_size' is missing.")

    elif dense_intermediate_size is None:
        raise ValueError("Model is not MoE but 'intermediate_size' is missing.")

    total_flops = 0.0

    # --- Calculate FLOPs based on phase ---
    if test_name == "prefill":
        seq_len = input_length
        context_length = input_length  # Causal mask means max context is seq_len

        # Attention FLOPs (per layer)
        if is_step2_mini and share_q_dim:
            # Step2MiniAttention specific calculation
            qkv_proj_flops = (
                2 * seq_len * hidden_size * (share_q_dim + 2 * head_dim)
            )  # num_kv_heads=1 -> kv_size=head_dim
            wq_proj_flops = 2 * seq_len * share_q_dim * q_attn_dim  # q_attn_dim = num_heads * head_dim
            qk_sv_flops = (
                2 * seq_len * context_length * q_attn_dim
            )  # Step2Mini uses single KV head, so kv_attn_dim == head_dim.
            o_proj_flops = 2 * seq_len * q_attn_dim * hidden_size
            attn_flops_per_layer = qkv_proj_flops + wq_proj_flops + qk_sv_flops + o_proj_flops
            total_attn_flops = num_layers * attn_flops_per_layer
        elif is_step3p5:
            total_attn_flops = sum(
                _standard_prefill_attn_flops(
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    q_attn_dim=attn_spec["q_attn_dim"],
                    kv_attn_dim=attn_spec["kv_attn_dim"],
                    sliding_window=attn_spec["sliding_window"],
                    extra_qkv_dim=attn_spec["extra_qkv_dim"],
                )
                for attn_spec in layer_attention_specs
            )
        else:
            qkv_flops = 2 * seq_len * hidden_size * (q_attn_dim + 2 * kv_attn_dim)
            sv_flops = 2 * seq_len * context_length * q_attn_dim
            o_proj_flops = 2 * seq_len * q_attn_dim * hidden_size
            attn_flops_per_layer = qkv_flops + sv_flops + o_proj_flops
            total_attn_flops = num_layers * attn_flops_per_layer

        # Dense MLP FLOPs (per layer) - Assuming SwiGLU (3 matrices)
        dense_mlp_flops_per_layer = 0
        if num_dense_layers > 0:
            if is_step3p5:
                dense_mlp_flops_per_layer = _step3p5_dense_mlp_flops(seq_len, hidden_size, dense_intermediate_size)
            else:
                dense_mlp_flops_per_layer = 6 * seq_len * hidden_size * dense_intermediate_size

        # MoE MLP FLOPs (per layer)
        moe_mlp_flops_per_layer = 0
        if num_moe_layers > 0:
            if is_step3p5:
                moe_mlp_flops_per_layer = _step3p5_moe_mlp_flops(
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    moe_intermediate_size=moe_intermediate_size,
                    share_expert_dim=share_expert_dim,
                    count_router_flops=count_router_flops,
                )
            else:
                router_flops = 2 * seq_len * hidden_size * num_experts  # Input * Output for router linear layer
                # Only top_k experts are active; SwiGLU has three matrices.
                expert_flops = top_k * (6 * seq_len * hidden_size * moe_intermediate_size)
                # Shared Expert FLOPs (Step2Mini specific)
                shared_expert_flops = 0
                if share_expert_dim > 0:
                    shared_expert_flops = 6 * seq_len * hidden_size * share_expert_dim
                moe_mlp_flops_per_layer = router_flops + expert_flops + shared_expert_flops

        # Total FLOPs for Prefill
        total_flops = (
            total_attn_flops + num_dense_layers * dense_mlp_flops_per_layer + num_moe_layers * moe_mlp_flops_per_layer
        )
        if is_step3p5:
            total_flops += _step3p5_embedding_flops(seq_len, hidden_size)
            total_flops += _step3p5_lm_head_flops(
                seq_len, hidden_size, transformer_config["vocab_size"], num_output_heads
            )

    elif test_name == "decode":
        # Decode calculates flops token by token for `output_length` tokens
        total_flops_decode = 0.0
        for i in range(output_length):
            seq_len = 1  # Processing one token
            context_length = input_length + i  # Current total sequence length

            # Attention FLOPs (per layer, per token)
            if is_step2_mini and share_q_dim:
                qkv_proj_flops = 2 * seq_len * hidden_size * (share_q_dim + 2 * head_dim)
                wq_proj_flops = 2 * seq_len * share_q_dim * q_attn_dim
                qk_sv_flops = 4 * seq_len * context_length * head_dim  # Context * Value dim
                o_proj_flops = 2 * seq_len * q_attn_dim * hidden_size
                attn_flops_per_token = qkv_proj_flops + wq_proj_flops + qk_sv_flops + o_proj_flops
                total_attn_flops = num_layers * attn_flops_per_token
            elif is_step3p5:
                total_attn_flops = sum(
                    _standard_decode_attn_flops(
                        seq_len=seq_len,
                        context_length=context_length,
                        hidden_size=hidden_size,
                        q_attn_dim=attn_spec["q_attn_dim"],
                        kv_attn_dim=attn_spec["kv_attn_dim"],
                        sliding_window=attn_spec["sliding_window"],
                        extra_qkv_dim=attn_spec["extra_qkv_dim"],
                    )
                    for attn_spec in layer_attention_specs
                )
            else:
                qkv_flops = 2 * seq_len * hidden_size * (q_attn_dim + 2 * kv_attn_dim)
                qk_sv_flops = 4 * seq_len * context_length * q_attn_dim
                o_proj_flops = 2 * seq_len * q_attn_dim * hidden_size
                attn_flops_per_token = qkv_flops + qk_sv_flops + o_proj_flops
                total_attn_flops = num_layers * attn_flops_per_token

            # Dense MLP FLOPs (per layer, per token)
            dense_mlp_flops_per_token = 0
            if num_dense_layers > 0:
                if is_step3p5:
                    dense_mlp_flops_per_token = _step3p5_dense_mlp_flops(seq_len, hidden_size, dense_intermediate_size)
                else:
                    dense_mlp_flops_per_token = 6 * seq_len * hidden_size * dense_intermediate_size

            # MoE MLP FLOPs (per layer, per token)
            moe_mlp_flops_per_token = 0
            if num_moe_layers > 0:
                if is_step3p5:
                    moe_mlp_flops_per_token = _step3p5_moe_mlp_flops(
                        seq_len=seq_len,
                        hidden_size=hidden_size,
                        num_experts=num_experts,
                        top_k=top_k,
                        moe_intermediate_size=moe_intermediate_size,
                        share_expert_dim=share_expert_dim,
                        count_router_flops=count_router_flops,
                    )
                else:
                    router_flops = 2 * seq_len * hidden_size * num_experts
                    expert_flops = top_k * (6 * seq_len * hidden_size * moe_intermediate_size)
                    # Shared Expert FLOPs (Step2Mini specific, per token)
                    shared_expert_flops_token = 0
                    if share_expert_dim > 0:
                        shared_expert_flops_token = 6 * seq_len * hidden_size * share_expert_dim  # seq_len is 1 here
                    moe_mlp_flops_per_token = router_flops + expert_flops + shared_expert_flops_token

            # Accumulate FLOPs for this token across all layers
            flops_this_token = (
                total_attn_flops
                + num_dense_layers * dense_mlp_flops_per_token
                + num_moe_layers * moe_mlp_flops_per_token
            )
            if is_step3p5:
                flops_this_token += _step3p5_embedding_flops(seq_len, hidden_size)
                flops_this_token += _step3p5_lm_head_flops(
                    seq_len,
                    hidden_size,
                    transformer_config["vocab_size"],
                    num_output_heads,
                )
            total_flops_decode += flops_this_token

        total_flops = total_flops_decode
    else:
        raise ValueError(f"Unknown test_name: {test_name}")

    # Convert FLOPs to TFLOPs (FLOPs / 1e12)
    # Note: Original calculation multiplied by 2.0 / 1e12.
    # Standard definition of FLOPs often counts multiply-adds as 2 FLOPs.
    # If the formulas above already count MACs as 2 FLOPs (common in LLM papers),
    # we just need to divide by 1e12. Let's stick to the original script's
    # convention and assume the base formulas count operations, so multiply by 2.
    total_tflops = total_flops / 1e12

    return total_tflops


def calculate_params(transformer_config):
    """
    Calculates the estimated total and active parameters of a transformer model.

    Args:
        transformer_config: Dictionary containing model configuration parameters.

    Returns:
        Tuple[float, float]:
            Estimated total parameters (in billions).
            Estimated active parameters in billions.
    """
    hidden_size = transformer_config["hidden_size"]
    vocab_size = transformer_config["vocab_size"]
    num_layers = transformer_config["num_hidden_layers"]
    num_heads = transformer_config["num_attention_heads"]
    head_dim = transformer_config.get("head_dim", hidden_size // num_heads)

    # Assume untied embeddings when tie_word_embeddings is absent.
    embed_params = vocab_size * hidden_size
    lm_head_params = hidden_size * vocab_size
    total_params = embed_params + lm_head_params
    active_params = embed_params + lm_head_params  # Embeddings always active

    # --- Norms --- (Approximate, ignoring exact count per layer type)
    # Final norm
    final_norm_params = 2 * hidden_size
    total_params += final_norm_params
    active_params += final_norm_params

    # Layer-specific components
    # Step2Mini specific attention parameters
    is_step2_mini = transformer_config.get("model_type") == "step2_mini"
    share_q_dim = transformer_config.get("share_q_dim") if is_step2_mini else None

    # Corrected num_kv_heads calculation
    if "num_key_value_heads" in transformer_config:
        num_kv_heads = transformer_config["num_key_value_heads"]
    elif "num_attention_groups" in transformer_config and transformer_config["num_attention_groups"] > 0:
        num_kv_heads = num_heads // transformer_config["num_attention_groups"]
    else:
        num_kv_heads = num_heads  # Default to MHA

    # Override for step2_mini as per its implementation
    if is_step2_mini:
        num_kv_heads = 1

    q_attn_dim = num_heads * head_dim

    # MLP configurations
    use_moe, moe_intermediate_size, num_experts, top_k = _resolve_moe_config(transformer_config)

    num_moe_layers = 0
    num_dense_layers = num_layers
    dense_intermediate_size = transformer_config.get("intermediate_size")
    share_expert_dim = transformer_config.get("share_expert_dim") if is_step2_mini else 0

    if use_moe:
        num_moe_layers, num_dense_layers = _resolve_num_moe_layers(transformer_config, num_layers)

    # --- Per-Layer Parameters ---
    attn_params_per_layer = 0
    dense_mlp_params_per_layer = 0
    moe_mlp_total_params_per_layer = 0
    moe_mlp_active_params_per_layer = 0
    shared_expert_params_per_layer = 0  # Only for MoE layers in Step2Mini
    norm_params_per_layer = 0  # Layer norms

    # Attention Params
    if is_step2_mini and share_q_dim:
        qkv_proj_params = hidden_size * (share_q_dim + 2 * head_dim)  # Wqkv
        wq_proj_params = share_q_dim * q_attn_dim  # Wq
        o_proj_params = q_attn_dim * hidden_size  # Wo
        attn_params_per_layer = qkv_proj_params + wq_proj_params + o_proj_params
        # Norms within Step2Mini Attention + Layer norms
        norm_params_per_layer = (
            2 * hidden_size + 2 * hidden_size + 2 * share_q_dim
        )  # input_ln, post_attn_ln, inter_norm
    else:
        # Standard Attention (approximated)
        kv_attn_dim = num_kv_heads * head_dim
        qkv_proj_params = hidden_size * (q_attn_dim + 2 * kv_attn_dim)  # Wq, Wk, Wv
        o_proj_params = q_attn_dim * hidden_size  # Wo
        attn_params_per_layer = qkv_proj_params + o_proj_params
        # Standard layer norms
        norm_params_per_layer = 2 * hidden_size + 2 * hidden_size  # input_ln, post_attn_ln

    # Dense MLP Params (SwiGLU approx)
    if num_dense_layers > 0 and dense_intermediate_size:
        dense_mlp_params_per_layer = 3 * hidden_size * dense_intermediate_size

    # MoE MLP Params
    if num_moe_layers > 0 and moe_intermediate_size:
        router_params = hidden_size * num_experts
        expert_params = 3 * hidden_size * moe_intermediate_size  # Params per expert
        moe_mlp_total_params_per_layer = router_params + num_experts * expert_params
        moe_mlp_active_params_per_layer = router_params + top_k * expert_params
        # Shared Expert (Step2Mini specific, added to MoE layers)
        if is_step2_mini and share_expert_dim > 0:
            shared_expert_params_per_layer = 3 * hidden_size * share_expert_dim
            moe_mlp_total_params_per_layer += shared_expert_params_per_layer
            moe_mlp_active_params_per_layer += shared_expert_params_per_layer

    # --- Aggregate Layer Params ---
    total_layer_params = (
        num_layers * (attn_params_per_layer + norm_params_per_layer)
        + num_dense_layers * dense_mlp_params_per_layer
        + num_moe_layers * moe_mlp_total_params_per_layer
    )

    active_layer_params = (
        num_layers * (attn_params_per_layer + norm_params_per_layer)
        + num_dense_layers * dense_mlp_params_per_layer
        + num_moe_layers * moe_mlp_active_params_per_layer
    )

    total_params += total_layer_params
    active_params += active_layer_params

    # --- Vision Tower Parameters ---
    if "vision_tower_config" in transformer_config:
        vision_config = transformer_config["vision_tower_config"]
        vision_params = calculate_vision_tower_params(vision_config)
        total_params += vision_params
        active_params += vision_params

    # --- SAM Model Parameters ---
    if "sam_model_config" in transformer_config:
        sam_config = transformer_config["sam_model_config"]
        sam_params = calculate_sam_model_params(sam_config)
        total_params += sam_params
        active_params += sam_params

    return total_params / 1e9, active_params / 1e9  # Return in billions


def calculate_vision_tower_params(vision_config):
    """
    Calculate parameters for vision tower component.

    Args:
        vision_config: Dictionary containing vision tower configuration.

    Returns:
        Total parameters count for vision tower.
    """
    hidden_size = vision_config.get("hidden_size", 0)
    output_hidden_size = vision_config.get("output_hidden_size", hidden_size)
    intermediate_size = vision_config.get("intermediate_size", 0)
    num_layers = vision_config.get("num_hidden_layers", 0)
    image_size = vision_config.get("image_size", 224)
    patch_size = vision_config.get("patch_size", 16)

    if hidden_size == 0 or num_layers == 0:
        return 0

    # Patch embedding (patch_size^2 * 3 channels -> hidden_size)
    patch_embed_params = (patch_size * patch_size * 3) * hidden_size

    # Position embedding (approximate based on image size and patch size)
    num_patches = (image_size // patch_size) ** 2 + 1  # +1 for cls token
    pos_embed_params = num_patches * hidden_size

    # Multi-head attention per layer
    attn_params_per_layer = 4 * hidden_size * hidden_size  # qkv + output projection

    # MLP per layer (assuming 4x expansion ratio)
    mlp_params_per_layer = (
        2 * hidden_size * intermediate_size if intermediate_size > 0 else 8 * hidden_size * hidden_size
    )

    # Layer norms per layer
    norm_params_per_layer = 2 * hidden_size  # pre-attn and pre-mlp norms

    # Total layer params
    layer_params = num_layers * (attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer)

    # Final layer norm
    final_norm_params = hidden_size

    # Output projection if different output size
    output_proj_params = 0
    if output_hidden_size != hidden_size:
        output_proj_params = hidden_size * output_hidden_size

    total = patch_embed_params + pos_embed_params + layer_params + final_norm_params + output_proj_params
    return total


def calculate_sam_model_params(sam_config):
    """
    Calculate parameters for SAM (Segment Anything Model) component.

    Args:
        sam_config: Dictionary containing SAM model configuration.

    Returns:
        Total parameters count for SAM model.
    """
    embed_dim = sam_config.get("embed_dim", 0)
    depth = sam_config.get("depth", 0)  # num_layers
    mlp_ratio = sam_config.get("mlp_ratio", 4)
    image_size = sam_config.get("image_size", 1024)
    patch_size = sam_config.get("patch_size", 16)
    out_channels = sam_config.get("out_channels", 256)

    if embed_dim == 0 or depth == 0:
        return 0

    # Patch embedding (patch_size^2 * 3 channels -> embed_dim)
    patch_embed_params = (patch_size * patch_size * 3) * embed_dim

    # Position embedding
    num_patches = (image_size // patch_size) ** 2
    pos_embed_params = num_patches * embed_dim

    # Transformer layers
    # Multi-head attention per layer
    attn_params_per_layer = 4 * embed_dim * embed_dim  # qkv + output projection

    # MLP per layer
    mlp_hidden_dim = int(embed_dim * mlp_ratio)
    mlp_params_per_layer = 2 * embed_dim * mlp_hidden_dim

    # Layer norms per layer
    norm_params_per_layer = 2 * embed_dim

    # Total transformer params
    transformer_params = depth * (attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer)

    # Neck/output projection
    neck_params = embed_dim * out_channels + out_channels * out_channels  # Simple approximation

    total = patch_embed_params + pos_embed_params + transformer_params + neck_params
    return total


if __name__ == "__main__":
    model_zoo_dir = Path("ci-model-zoo")
    if not model_zoo_dir.is_dir():
        print(f"Error: Model zoo directory not found at '{model_zoo_dir}'")
        print("Please ensure the script is run from the workspace root.")
        exit(1)

    print(f"--- Running Parameter and FLOPs Calculation for Models in: {model_zoo_dir} ---")

    # Example parameters for FLOPs calculation (used for all models)
    input_len_prefill = 4096
    input_len_decode = 4096
    output_len_decode = 1

    # Iterate through subdirectories in the model zoo
    for model_dir in sorted(model_zoo_dir.iterdir()):  # Sort for consistent order
        if not model_dir.is_dir():
            continue  # Skip files, .git, etc.

        # Ignore hidden directories like .git
        if model_dir.name.startswith("."):
            continue

        config_path = model_dir / "config.json"

        if config_path.is_file():
            print(f"\n--- Model: {model_dir.name} ---")
            try:
                with open(config_path) as f:
                    model_config = json.load(f)

                print(f"  Config Path: {config_path}")
                print(f"  Model Type: {model_config.get('model_type', 'N/A')}")
                # Add more config details if needed for debugging
                # print(f"  Hidden Size: {model_config.get('hidden_size')}")
                # print(f"  Num Layers: {model_config.get('num_hidden_layers')}")
                is_moe, _, _, _ = _resolve_moe_config(model_config)

                # Calculate and print parameters
                total_b, active_b = 0.0, 0.0
                try:
                    total_b, active_b = calculate_params(model_config)
                    print(f"  Params (Est.): {total_b:.2f} B total", end="")
                    if is_moe and active_b != total_b:
                        print(f" / {active_b:.2f} B active")
                    else:
                        print()  # Newline if not MoE or active == total
                except Exception as e:
                    print(f"\n  Error calculating parameters: {e}")

                # Calculate Prefill TFLOPs
                try:
                    prefill_tflops = sequence_flops(model_config, input_len_prefill, None, "prefill")
                    print(f"  Prefill TFLOPs (in={input_len_prefill}): {prefill_tflops:.4f}")
                except Exception as e:
                    print(f"  Error calculating prefill TFLOPs: {e}")

                # Calculate Decode TFLOPs
                try:
                    decode_total_tflops = sequence_flops(model_config, input_len_decode, output_len_decode, "decode")
                    decode_tflops_per_token = decode_total_tflops / output_len_decode if output_len_decode > 0 else 0
                    print(
                        "  Decode TFLOPs "
                        f"(in={input_len_decode}, out={output_len_decode}): "
                        f"{decode_tflops_per_token:.4f}"
                    )
                except Exception as e:
                    print(f"  Error calculating decode TFLOPs: {e}")

            except json.JSONDecodeError:
                print(f"  Error: Could not decode JSON from '{config_path}'")
            except KeyError as e:
                print(f"  Error: Missing key '{e}' in config '{config_path}'")
            except ValueError as e:
                print(f"  Error: Value error during calculation for '{config_path}': {e}")
            except Exception as e:
                print(f"  An unexpected error occurred for '{config_path}': {e}")
        # else:
        # Optional: print a message if config.json is not found
        # print(f"\n--- Skipping: {model_dir.name} (no config.json) ---")

    print("\n--- Calculation Complete ---")
