# Adapted from THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219.

import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except ImportError:
    pass

from .hf_attention import HuggingfaceAttention, _load_hf_config
from .qwen_gdn_backend import get_chunk_gated_delta_rule


logger = logging.getLogger(__name__)


def _diagnostic_rank_matches() -> bool:
    if os.environ.get("SLIME_DIAG_NONFINITE_GRADS", "0") != "1":
        return False
    if not torch.distributed.is_initialized():
        return False
    requested_rank = int(
        os.environ.get("SLIME_DIAG_NONFINITE_GRADS_RANK", "-1")
    )
    rank = torch.distributed.get_rank()
    return requested_rank < 0 or rank == requested_rank


def _diagnose_nonfinite(
    layer_idx: int,
    stage: str,
    **tensors: torch.Tensor,
) -> None:
    if not _diagnostic_rank_matches():
        return
    rank = torch.distributed.get_rank()
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        finite_values = tensor[finite]
        finite_abs_max = (
            float(finite_values.abs().max().item())
            if finite_values.numel()
            else None
        )
        logger.error(
            "QWEN35_GDN_DIAG_NONFINITE rank=%s layer=%s stage=%s "
            "tensor=%s shape=%s dtype=%s nan=%s posinf=%s neginf=%s "
            "finite_abs_max=%s",
            rank,
            layer_idx,
            stage,
            name,
            tuple(tensor.shape),
            tensor.dtype,
            int(torch.isnan(tensor).sum().item()),
            int(torch.isposinf(tensor).sum().item()),
            int(torch.isneginf(tensor).sum().item()),
            finite_abs_max,
        )


def _attach_gradient_diagnostic(
    tensor: torch.Tensor,
    layer_idx: int,
    stage: str,
) -> None:
    if not _diagnostic_rank_matches() or not tensor.requires_grad:
        return

    def diagnose(gradient: torch.Tensor) -> torch.Tensor:
        _diagnose_nonfinite(layer_idx, f"grad_{stage}", gradient=gradient)
        return gradient

    tensor.register_hook(diagnose)


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


class Qwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5 GatedDeltaNet with packed variable-length support."""

    def __init__(self, config, layer_idx: int, args=None):
        super().__init__()
        self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla")
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule(self.gdn_backend)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        projection_size_qkv = self.key_dim * 2 + self.value_dim
        projection_size_z = self.value_dim
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, projection_size_qkv, bias=False
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size, projection_size_z, bias=False
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        a = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(a))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=torch.cuda.current_device(),
            dtype=(
                config.dtype
                if config.dtype is not None
                else torch.get_default_dtype()
            ),
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        _diagnose_nonfinite(
            self.layer_idx,
            "input",
            hidden_states=hidden_states,
        )
        _attach_gradient_diagnostic(
            hidden_states,
            self.layer_idx,
            "hidden_states",
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        _diagnose_nonfinite(
            self.layer_idx,
            "projections",
            mixed_qkv=mixed_qkv,
            z=z,
            b=b,
            a=a,
        )
        _attach_gradient_diagnostic(
            mixed_qkv,
            self.layer_idx,
            "mixed_qkv_pre_conv",
        )
        _attach_gradient_diagnostic(z, self.layer_idx, "z")
        _attach_gradient_diagnostic(b, self.layer_idx, "b")
        _attach_gradient_diagnostic(a, self.layer_idx, "a")

        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )
        _diagnose_nonfinite(
            self.layer_idx,
            "post_conv",
            mixed_qkv=mixed_qkv,
        )
        _attach_gradient_diagnostic(
            mixed_qkv,
            self.layer_idx,
            "mixed_qkv_post_conv",
        )

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch_size, seq_len, -1, self.head_k_dim
        )
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        _diagnose_nonfinite(
            self.layer_idx,
            "gdn_inputs",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
        )
        _attach_gradient_diagnostic(query, self.layer_idx, "query")
        _attach_gradient_diagnostic(key, self.layer_idx, "key")
        _attach_gradient_diagnostic(value, self.layer_idx, "value")
        _attach_gradient_diagnostic(g, self.layer_idx, "g")
        _attach_gradient_diagnostic(beta, self.layer_idx, "beta")
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if self.gdn_backend == "flashqla":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            g = g.contiguous()
            beta = beta.contiguous()

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        _diagnose_nonfinite(
            self.layer_idx,
            "post_gdn",
            core_attn_out=core_attn_out,
        )
        _attach_gradient_diagnostic(
            core_attn_out,
            self.layer_idx,
            "core_attn_out_pre_norm",
        )

        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(
            -1, core_attn_out.shape[-1]
        )
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        _diagnose_nonfinite(
            self.layer_idx,
            "post_norm",
            core_attn_out=core_attn_out,
        )
        _attach_gradient_diagnostic(
            core_attn_out,
            self.layer_idx,
            "core_attn_out_post_norm",
        )
        core_attn_out = core_attn_out.reshape(z_shape)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        _diagnose_nonfinite(
            self.layer_idx,
            "output",
            output=output,
        )
        _attach_gradient_diagnostic(output, self.layer_idx, "output")
        return output


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        model_comm_pgs=None,
        name: str | None = None,
    ):
        # MCore 0.14 passes ``model_comm_pgs``; current upstream Slime calls
        # the same compatibility slot ``pg_collection``.  The HF-backed GDN
        # block uses global MCore process groups, so either object is only
        # retained for constructor compatibility.
        process_groups = (
            pg_collection
            if pg_collection is not None
            else model_comm_pgs
        )
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            process_groups,
        )
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"
        self.linear_attn = Qwen3_5GatedDeltaNet(
            self.hf_config, self.hf_layer_idx, args=args
        )

        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import (
                Qwen3NextRMSNorm,
            )

            self.input_layernorm = Qwen3NextRMSNorm(
                self.hf_config.hidden_size,
                eps=self.hf_config.rms_norm_eps,
            )
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(
                self.hf_config.hidden_size,
                eps=self.hf_config.rms_norm_eps,
            )

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        return self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )


def get_qwen3_5_spec(args, config, vp_stage):
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert (
        config.pipeline_model_parallel_layout is None
    ), "not support this at the moment"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    text_config = _get_text_config(_load_hf_config(args.hf_checkpoint))
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        num_layers = text_config.num_hidden_layers
        text_config.layer_types = [
            (
                "full_attention"
                if (layer_idx + 1) % interval == 0
                else "linear_attention"
            )
            for layer_idx in range(num_layers)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(
                transformer_layer_spec.layer_specs[layer_id]
            )
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
