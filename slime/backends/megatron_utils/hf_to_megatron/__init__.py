from pathlib import Path

from transformers import AutoConfig

from .common import load_model_hf_weights
from .qwen3_5 import qwen3_5_hf_tensor

_LOADERS = {
    "qwen3_5": qwen3_5_hf_tensor,
    # Our training artifact is the official language-model sub-checkpoint.  It
    # keeps the canonical model.language_model tensor names while its config
    # advertises the text architecture directly.
    "qwen3_5_text": qwen3_5_hf_tensor,
}


def supports_hf_weight_loading(path: str | Path) -> bool:
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config.model_type in _LOADERS


def load_hf_weights(args, model, path: str | Path) -> None:
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    try:
        get_hf_tensor = _LOADERS[config.model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported HuggingFace model type: {config.model_type}"
        ) from exc
    load_model_hf_weights(args, model, path, config, get_hf_tensor)
