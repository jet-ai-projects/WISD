import logging

from megatron.training.arguments import parse_args, validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # Keep explicit model/runtime limits. Clobbering these values here made
    # long-context checkpoints silently train with a 4K position limit even
    # when their model recipe requested a larger context.
    if args.seq_length is None:
        args.seq_length = 4096
    if args.max_position_embeddings is None:
        args.max_position_embeddings = args.seq_length
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
