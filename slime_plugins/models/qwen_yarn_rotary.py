"""YaRN rotary frequencies for dense Qwen models in Megatron Core.

Megatron's dense GPT path only exposes its Llama-3-style RoPE scaling switch;
the built-in YaRN module is coupled to Multi-Latent Attention and returns a
different interface. DeepSeek-R1-0528-Qwen3-8B is a dense Qwen3 checkpoint, so
we retain the ordinary GPT attention path and replace only its inverse
frequencies with the same YaRN formula used by vLLM.
"""

from __future__ import annotations

import math

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)


def _correction_dim(
    rotations: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> float:
    return (
        dim
        * math.log(
            original_max_position_embeddings / (rotations * 2 * math.pi)
        )
        / (2 * math.log(base))
    )


def _correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(
        _correction_dim(
            beta_fast,
            dim,
            base,
            original_max_position_embeddings,
        )
    )
    high = math.ceil(
        _correction_dim(
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
    )
    return max(low, 0), min(high, dim - 1)


def _linear_ramp(low: int, high: int, size: int, *, device) -> torch.Tensor:
    if low == high:
        high += 0.001
    values = (torch.arange(size, dtype=torch.float32, device=device) - low) / (
        high - low
    )
    return values.clamp(0, 1)


class QwenYarnRotaryEmbedding(RotaryEmbedding):
    """Dense-GPT-compatible YaRN whose forward still returns angle tensors."""

    def __init__(
        self,
        *,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool,
        rotary_base: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        attention_factor: float,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        use_cpu_initialization: bool = False,
        cp_group=None,
    ) -> None:
        if scaling_factor < 1:
            raise ValueError("Qwen YaRN scaling_factor must be at least 1")
        if original_max_position_embeddings <= 0:
            raise ValueError(
                "Qwen YaRN original_max_position_embeddings must be positive"
            )
        if attention_factor <= 0:
            raise ValueError("Qwen YaRN attention_factor must be positive")
        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=None,
            rotary_base=rotary_base,
            rope_scaling=False,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
        )

        # vLLM multiplies both cos and sin by
        # yarn_get_mscale(factor) * attn_factor. The locked DeepSeek checkpoint
        # chooses attn_factor as the exact reciprocal, so the net multiplier is
        # one and the standard dense-GPT apply_rotary interface is sufficient.
        net_mscale = (
            1.0 + 0.1 * math.log(float(scaling_factor))
        ) * float(attention_factor)
        if not math.isclose(net_mscale, 1.0, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                "dense Qwen YaRN currently requires unit net mscale; "
                f"got {net_mscale}"
            )

        dim = self.inv_freq.numel() * 2
        extrapolation = self.inv_freq
        interpolation = extrapolation / float(scaling_factor)
        low, high = _correction_range(
            beta_fast,
            beta_slow,
            dim,
            float(rotary_base),
            original_max_position_embeddings,
        )
        extrapolation_mask = 1.0 - _linear_ramp(
            low,
            high,
            dim // 2,
            device=extrapolation.device,
        )
        self.inv_freq = (
            interpolation * (1.0 - extrapolation_mask)
            + extrapolation * extrapolation_mask
        )
