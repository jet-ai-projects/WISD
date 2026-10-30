"""RTN QDQ weak anchors; group-wise symmetric round-to-nearest quantization."""
from ._qwen3 import (discover_targets, fingerprint, scope_specification,
                    model_parameter_dtype, transform_row_blocks, transform_precision)

VARIANTS = {"rtn3_g128": (3, 128), "rtn4_g32": (4, 32), "rtn6_g128": (6, 128),
            "rtn4_g128": (4, 128)}

def specification(variant, *, module_scope="all"):
    bits, group_size = VARIANTS[variant]
    qmax = 2 ** (bits - 1) - 1
    return {
        "kind": "symmetric_weight_only_rtn_qdq", "variant": variant,
        "bits": bits, "group_size": group_size, "integer_range": [-qmax, qmax],
        "rounding": "ties-to-even", "group_axis": "consecutive input columns within each row",
        "scale": "max(abs(group))/qmax", "dtype": "float32",
        **scope_specification(module_scope),
        "preserve_bias_norm_embedding_head": True, "calibration": False,
    }


def quantize_weight(weight, *, bits, group_size):
    import torch
    if (type(bits) is not int or bits not in (3, 4, 6)
            or type(group_size) is not int or group_size not in (32, 128)
            or weight.ndim != 2 or weight.dtype != torch.float32
            or min(weight.shape) <= 0 or weight.shape[1] % group_size
            or not bool(torch.isfinite(weight).all())):
        raise ValueError("RTN requires finite FP32 matrices, supported bits and divisible input groups")
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        groups = weight.reshape(weight.shape[0], -1, group_size)
        maxima = groups.abs().amax(-1, keepdim=True)
        scale = maxima / qmax
        nonzero = maxima != 0
        if not bool(((~nonzero) | (scale > 0)).all()):
            raise ValueError("Nonzero FP32 group scale underflow")
        safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
        result = (groups / safe_scale).round().clamp(-qmax, qmax) * scale
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite quantized weight")
        return result.reshape_as(weight)


def apply_quantization(model, variant, *, require_original=True, module_scope="all"):
    import torch
    spec = specification(variant, module_scope=module_scope)
    if getattr(model, "_weight_weak_transformed", False):
        raise ValueError("Transform the original model exactly once")
    bits, group_size = VARIANTS[variant]
    dtype = model_parameter_dtype(model)
    targets = discover_targets(model, require_original=require_original, expected_dtype=dtype,
                               module_scope=module_scope)
    target_ids = {id(w) for w in targets.values()}
    excluded = {n: w for n, w in model.named_parameters() if id(w) not in target_ids}
    before = fingerprint(excluded)
    config = model.config.to_dict()
    aliases = {n: id(w) for n, w in model.named_parameters(remove_duplicate=False)}
    with torch.no_grad():
        for weight in targets.values():
            transform_row_blocks(
                weight, lambda block: quantize_weight(block, bits=bits, group_size=group_size))
    if (before != fingerprint(excluded) or config != model.config.to_dict()
            or aliases != {n: id(w) for n, w in model.named_parameters(remove_duplicate=False)}):
        raise ValueError("Quantization changed excluded parameters, config, or sharing")
    model._weight_weak_transformed = True
    actual_targets = {
        "module_scope": module_scope, "target_names": list(targets),
        "target_modules": len(targets), "target_parameter_count": sum(w.numel() for w in targets.values()),
    }
    return {
        "specification": {**spec, **actual_targets, **transform_precision(dtype)},
        **actual_targets,
        "target_weights_sha256": fingerprint(targets), "excluded_parameters_sha256": before,
        "excluded_parameters_unchanged": True, "applications": 1,
    }
