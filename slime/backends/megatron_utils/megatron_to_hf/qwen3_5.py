import re

import torch


def convert_qwen3_5_to_hf(args, name, param):
    """Convert a Qwen3.5 MCore parameter to its canonical HF tensor(s)."""

    if name.startswith("module.module.language_model."):
        name = "module.module." + name.removeprefix(
            "module.module.language_model."
        )

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        target = (
            "lm_head.weight"
            if getattr(args, "untie_embeddings_and_output_weights", False)
            else "model.language_model.embed_tokens.weight"
        )
        return [(target, param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = (
            args.kv_channels
            if args.kv_channels is not None
            else args.hidden_size // args.num_attention_heads
        )
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    queries_per_group = (
        args.num_attention_heads // args.num_query_groups
    )

    match = re.match(
        r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name
    )
    if match:
        layer_idx, rest = match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", param)]
        if rest in {
            "self_attention.linear_qkv.weight",
            "self_attention.linear_qgkv.weight",
        }:
            param = param.view(
                args.num_query_groups,
                -1,
                head_dim,
                args.hidden_size,
            )
            q_param, k_param, v_param = torch.split(
                param,
                split_size_or_sections=[
                    2 * queries_per_group,
                    1,
                    1,
                ],
                dim=1,
            )
            q_param = (
                q_param.reshape(
                    args.num_query_groups,
                    2,
                    queries_per_group,
                    head_dim,
                    args.hidden_size,
                )
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_param),
                (f"{prefix}.self_attn.k_proj.weight", k_param),
                (f"{prefix}.self_attn.v_proj.weight", v_param),
            ]
        if rest in {
            "self_attention.linear_qkv.bias",
            "self_attention.linear_qgkv.bias",
        }:
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    queries_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            return [
                (f"{prefix}.self_attn.q_proj.bias", q_bias.flatten()),
                (f"{prefix}.self_attn.k_proj.bias", k_bias.flatten()),
                (f"{prefix}.self_attn.v_proj.bias", v_bias.flatten()),
            ]
        if rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.up_proj.weight", up_weight),
            ]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.down_proj.weight", param)]
        if rest in {
            "self_attention.linear_qkv.layer_norm_weight",
            "self_attention.linear_qgkv.layer_norm_weight",
        }:
            return [(f"{prefix}.input_layernorm.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"{prefix}.post_attention_layernorm.weight", param)
            ]
        if rest == "pre_mlp_layernorm.weight":
            return [
                (f"{prefix}.post_attention_layernorm.weight", param)
            ]
        if rest == "self_attention.q_layernorm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", param)]
        if rest == "self_attention.k_layernorm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", param)]

        direct_attention_names = {
            "input_layernorm.weight",
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        }
        if rest.startswith("self_attention."):
            direct_name = rest.removeprefix("self_attention.")
            if direct_name in direct_attention_names:
                return [(f"{prefix}.{direct_name}", param)]

    raise ValueError(f"Unknown Qwen3.5 parameter name: {name}")
