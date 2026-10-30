"""Training-free constructions for the frozen weak scorer.

The model-level adapter targets the ordinary Qwen3-4B architecture. The
matrix primitives can be used by explicit adapters for other architectures.
"""
from ._qwen3 import MODULE_SCOPES
from .rtn import apply_quantization, quantize_weight
from .pruning import apply_pruning, transform_weight

VARIANTS = (
    "rtn3_g128", "rtn4_g128", "rtn4_g32", "rtn6_g128",
    "group_prune_g128_tau1of6", "group_prune_g128_tau1of6_rownorm",
)


def apply(model, variant, *, module_scope="all"):
    """Weaken a freshly loaded Qwen3-4B scorer in place, preserving dtype."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown weak-anchor variant: {variant}")
    if variant.startswith("rtn"):
        return apply_quantization(model, variant, module_scope=module_scope)
    return apply_pruning(model, variant, module_scope=module_scope)
