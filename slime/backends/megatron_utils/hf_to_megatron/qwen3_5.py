from __future__ import annotations

import re

import torch

from .common import SafetensorReader, strip_mcore_wrappers


def _merge_qkv(
    reader: SafetensorReader,
    prefix: str,
    text_config,
    suffix: str,
) -> torch.Tensor:
    q = reader.get_tensor(f"{prefix}.q_proj.{suffix}")
    k = reader.get_tensor(f"{prefix}.k_proj.{suffix}")
    v = reader.get_tensor(f"{prefix}.v_proj.{suffix}")
    num_groups = text_config.num_key_value_heads
    queries_per_group = (
        text_config.num_attention_heads // num_groups
    )
    head_dim = text_config.head_dim

    trailing_shape = q.shape[1:]
    # Qwen3.5 interleaves the attention output gate with Q.  MCore expects
    # [group, gated-query, key, value], hence this differs from Qwen3.
    q = q.reshape(
        num_groups,
        queries_per_group,
        2,
        head_dim,
        *trailing_shape,
    ).transpose(1, 2)
    q = q.flatten(1, 3)
    k = k.reshape(num_groups, head_dim, *trailing_shape)
    v = v.reshape(num_groups, head_dim, *trailing_shape)
    return torch.cat((q, k, v), dim=1).reshape(
        -1, *trailing_shape
    ).contiguous()


def qwen3_5_hf_tensor(
    name: str, reader: SafetensorReader, hf_config
) -> torch.Tensor:
    """Return an unsharded MCore tensor for a Qwen3.5 parameter."""

    name = strip_mcore_wrappers(name)
    if name.startswith("model.visual."):
        return reader.get_tensor(name)
    name = name.removeprefix("language_model.")

    text_config = getattr(hf_config, "text_config", hf_config)
    direct_mapping = {
        "embedding.word_embeddings.weight": (
            "model.language_model.embed_tokens.weight"
        ),
        "decoder.final_layernorm.weight": (
            "model.language_model.norm.weight"
        ),
        "output_layer.weight": (
            "model.language_model.embed_tokens.weight"
            if (
                getattr(hf_config, "tie_word_embeddings", False)
                or getattr(text_config, "tie_word_embeddings", False)
            )
            else "lm_head.weight"
        ),
    }
    if name in direct_mapping:
        return reader.get_tensor(direct_mapping[name])

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise KeyError(
            f"Unsupported Qwen3.5 Megatron parameter {name!r}"
        )
    layer_idx, rest = layer_match.groups()
    prefix = f"model.language_model.layers.{layer_idx}"

    if rest.startswith("self_attention.linear_attn."):
        suffix = rest.removeprefix("self_attention.")
        return reader.get_tensor(f"{prefix}.{suffix}")
    if rest == "self_attention.input_layernorm.weight":
        return reader.get_tensor(f"{prefix}.input_layernorm.weight")
    if rest == "self_attention.linear_proj.weight":
        return reader.get_tensor(f"{prefix}.self_attn.o_proj.weight")
    if rest in {
        "self_attention.linear_qkv.weight",
        "self_attention.linear_qgkv.weight",
    }:
        return _merge_qkv(
            reader, f"{prefix}.self_attn", text_config, "weight"
        )
    if rest in {
        "self_attention.linear_qkv.bias",
        "self_attention.linear_qgkv.bias",
    }:
        return _merge_qkv(
            reader, f"{prefix}.self_attn", text_config, "bias"
        )
    if rest in {
        "self_attention.linear_qkv.layer_norm_weight",
        "self_attention.linear_qgkv.layer_norm_weight",
    }:
        return reader.get_tensor(f"{prefix}.input_layernorm.weight")
    if rest == "self_attention.q_layernorm.weight":
        return reader.get_tensor(
            f"{prefix}.self_attn.q_norm.weight"
        )
    if rest == "self_attention.k_layernorm.weight":
        return reader.get_tensor(
            f"{prefix}.self_attn.k_norm.weight"
        )

    if rest in {
        "mlp.linear_fc1.layer_norm_weight",
        "pre_mlp_layernorm.weight",
    }:
        return reader.get_tensor(
            f"{prefix}.post_attention_layernorm.weight"
        )
    if rest == "mlp.linear_fc1.weight":
        gate = reader.get_tensor(f"{prefix}.mlp.gate_proj.weight")
        up = reader.get_tensor(f"{prefix}.mlp.up_proj.weight")
        return torch.cat((gate, up), dim=0)
    if rest == "mlp.linear_fc2.weight":
        return reader.get_tensor(f"{prefix}.mlp.down_proj.weight")

    raise KeyError(f"Unsupported Qwen3.5 Megatron parameter {name!r}")
