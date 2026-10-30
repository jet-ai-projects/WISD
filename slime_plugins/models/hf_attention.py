import json
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule


def _load_hf_config(checkpoint_path):
    """Load an HF config, with a JSON fallback for a newer model type."""
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    except (ValueError, KeyError):
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path) as config_file:
            config_dict = json.load(config_file)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        def fix_dtype(config):
            if "torch_dtype" in config:
                config["torch_dtype"] = dtype_map.get(
                    config["torch_dtype"], config["torch_dtype"]
                )
            if "dtype" in config:
                config["dtype"] = dtype_map.get(
                    config["dtype"], config["dtype"]
                )

        fix_dtype(config_dict)
        namespace = type("HFConfig", (), config_dict)()
        if "text_config" in config_dict:
            fix_dtype(config_dict["text_config"])
            namespace.text_config = type(
                "TextConfig", (), config_dict["text_config"]
            )()
        return namespace


class _AllGatherForDuplicatedComputation(torch.autograd.Function):
    """All-gather whose backward returns only the local gradient slice."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        out = [
            torch.empty_like(x)
            for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(out, x.contiguous(), group=group)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        return grads[ctx.rank], None


class HuggingfaceAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.args = args
        self.config = config
        # Note that megatron layer_number starts at 1
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.hf_config = _load_hf_config(args.hf_checkpoint)
        # hardcode to fa2 at the moment.
        self.hf_config._attn_implementation = "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        if self.args.sequence_parallel:
            try:
                hidden_states = (
                    tensor_parallel.gather_from_sequence_parallel_region(
                        hidden_states,
                        tensor_parallel_output_grad=False,
                        group=mpu.get_tensor_model_parallel_group(),
                    )
                )
            except TypeError:
                # Older MCore images do not expose tensor_parallel_output_grad.
                # TP=1 is the locked Qwen3.5 recipe, where both semantics are
                # identical; reject a silently wrong TP>1 fallback.
                if mpu.get_tensor_model_parallel_world_size() != 1:
                    raise
                hidden_states = (
                    tensor_parallel.gather_from_sequence_parallel_region(
                        hidden_states,
                        group=mpu.get_tensor_model_parallel_group(),
                    )
                )

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            hidden_states_list = _AllGatherForDuplicatedComputation.apply(
                hidden_states,
                mpu.get_context_parallel_group(),
            )

            # TODO: preprocess this for each batch to prevent tolist in the training step
            whole_hidden_states_list = []

            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        output = self.hf_forward(hidden_states, packed_seq_params)
        bias = None

        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        if mpu.get_context_parallel_world_size() > 1:
            cp_rank = mpu.get_context_parallel_rank()
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                chunks = torch.chunk(seq, 2 * cp_size, dim=0)
                output_list.append(chunks[cp_rank])
                output_list.append(chunks[2 * cp_size - 1 - cp_rank])
            output = torch.cat(output_list, dim=0)

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, bias

    @abstractmethod
    def hf_forward(self, hidden_states, packed_seq_params):
        """Huggingface forward function"""
