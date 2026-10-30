"""Group-threshold weak anchors; unchanged FP32 arithmetic."""
from ._qwen3 import (discover_targets, fingerprint, scope_specification,
                    model_parameter_dtype, transform_row_blocks, transform_precision)

PRUNING_VARIANTS = ("group_prune_g128_tau1of6", "group_prune_g128_tau1of6_rownorm")

def specification(variant, *, module_scope="all"):
    if variant not in PRUNING_VARIANTS:
        raise ValueError("Only the two fixed group-threshold pruning variants")
    normalized = variant.endswith("_rownorm")
    return {
        "kind": "group_relative_threshold_pruning", "variant": variant,
        "group_size": 128, "group_axis": "consecutive input columns within each output row",
        "threshold": "max(abs(group)) / 6 in FP32",
        "keep_rule": "abs(weight) > threshold; equality is deleted",
        "unscaled_nonzero_values": "original FP32 weights, never rounded",
        "formula": ("N_W(W * (abs(W) > maxabs_g128(W)/6))" if normalized
                    else "W * (abs(W) > maxabs_g128(W)/6)"),
        "dtype": "float32", "calibration": False, "calibration_required": False,
        **scope_specification(module_scope),
        "preserve_bias_norm_embedding_head": True, "activation_quantization": False,
        "low_bit_kernel": False, "zero_rows": "remain exactly zero",
        "row_norm_preserved": normalized,
        "norm": "whole output row; torch.linalg.vector_norm(ord=2,dim=1,dtype=float32)" if normalized else None,
        "norm_tolerance": 2e-6 if normalized else None,
        "arithmetic": "Canonical contiguous FP32 rows, no BF16 recast; finite/nonzero threshold and norm checks",
        "rtn3_mask_equivalence_claim": False,
        "threshold_boundary_note": "Direct FP32 maxabs/6 threshold, not a call to RTN3 round/dequantize",
    }


def transform_weight(weight, variant):
    import torch
    spec = specification(variant)
    if (weight.ndim != 2 or weight.dtype != torch.float32 or min(weight.shape) <= 0
            or weight.shape[1] % 128 or not bool(torch.isfinite(weight).all())):
        raise ValueError("Require finite nonempty FP32 matrix with input dimension divisible128")
    with torch.no_grad():
        weight = weight.contiguous()
        groups = weight.reshape(weight.shape[0], -1, 128)
        maxima = groups.abs().amax(dim=-1, keepdim=True)
        threshold = maxima / 6.
        if not bool(((maxima == 0) | (threshold > 0)).all()):
            raise ValueError("Nonzero-group FP32 threshold underflow")
        mask = groups.abs() > threshold
        result = torch.where(mask, groups, torch.zeros_like(groups)).reshape_as(weight)
        zero = (weight == 0).all(dim=1)
        if spec["row_norm_preserved"]:
            before = torch.linalg.vector_norm(weight, ord=2, dim=1, dtype=torch.float32)
            after = torch.linalg.vector_norm(result, ord=2, dim=1, dtype=torch.float32)
            if (not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all())
                    or not bool((zero | ((before > 0) & (after > 0))).all())):
                raise ValueError("Nonzero-row FP32 norm underflow or nonfinite row norm")
            safe = torch.where(zero, torch.ones_like(after), after)
            scale = torch.where(zero, torch.ones_like(before), before / safe)
            result = result * scale[:, None]
            final = torch.linalg.vector_norm(result, ord=2, dim=1, dtype=torch.float32)
            divisor = torch.where(zero, torch.ones_like(before), before)
            error = (final - before).abs() / divisor
            if not bool((error <= spec["norm_tolerance"]).all()):
                raise ValueError("FP32 row-norm relative error exceeds fixed tolerance")
        if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
            raise ValueError("Pruned weights must remain finite FP32")
        if not bool((result[zero] == 0).all()):
            raise ValueError("All-zero original row changed")
        if not bool(((result != 0).reshape_as(groups) == mask).all()):
            raise ValueError("Normalization changed the fixed pruning support")
        return result


def apply_pruning(model, variant, *, calibration=None, require_original=True, module_scope="all"):
    import torch
    spec = specification(variant, module_scope=module_scope)
    if calibration is not None:
        raise ValueError("Group-relative pruning must not consume calibration")
    if getattr(model, "_weight_weak_transformed", False):
        raise ValueError("Apply the weak transform to the original model exactly once")
    dtype = model_parameter_dtype(model)
    targets = discover_targets(model, require_original=require_original, expected_dtype=dtype,
                               module_scope=module_scope)
    ids = {id(weight) for weight in targets.values()}
    excluded = {name: weight for name, weight in model.named_parameters() if id(weight) not in ids}
    before_excluded = fingerprint(excluded)
    aliases = {name: id(weight) for name, weight in model.named_parameters(remove_duplicate=False)}
    config = model.config.to_dict()
    zeros, original_zeros, maximum_error, maximum_forward_error = 0, 0, 0., 0.
    with torch.no_grad():
        for weight in targets.values():
            stats = transform_row_blocks(weight, lambda block: transform_weight(block, variant))
            original_zeros += stats["original_zero_weights"]
            zeros += stats["final_zero_weights"]
            maximum_error = max(maximum_error, stats["maximum_fp32_row_norm_relative_error"])
            maximum_forward_error = max(maximum_forward_error, stats["maximum_forward_row_norm_relative_error"])
    if (fingerprint(excluded) != before_excluded or config != model.config.to_dict()
            or aliases != {name: id(weight) for name, weight in model.named_parameters(remove_duplicate=False)}):
        raise ValueError("Pruning changed excluded parameters, architecture or sharing")
    model._weight_weak_transformed = True
    actual_targets = {
        "module_scope": module_scope, "target_names": list(targets), "target_modules": len(targets),
        "target_parameter_count": sum(weight.numel() for weight in targets.values()),
    }
    return {
        "specification": {**spec, **actual_targets, **transform_precision(dtype),
                          "arithmetic": "FP32 complete-row blocks; explicit cast back to forward dtype"},
        **actual_targets,
        "target_weights_sha256": fingerprint(targets),
        "excluded_parameters_sha256": before_excluded,
        "excluded_parameters_unchanged": True, "applications": 1, "activation_hooks": 0,
        "original_zero_weights": original_zeros, "final_zero_weights": zeros,
        "maximum_fp32_row_norm_relative_error": maximum_error if spec["row_norm_preserved"] else None,
        "maximum_forward_row_norm_relative_error": maximum_forward_error if spec["row_norm_preserved"] else None,
        "weak_checkpoint_saved": False,
    }
