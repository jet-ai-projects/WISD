"""Original Qwen3-4B target inventory and exact-byte fingerprint."""
import hashlib

GROUP_SIZE = 128
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ORIGINAL_PARAMETER_COUNT = 3633315840
MODULE_SCOPES = ("all", "mlp", "attention")
_SCOPE_PROJECTIONS = {
    "all": PROJECTIONS, "mlp": PROJECTIONS[4:], "attention": PROJECTIONS[:4],
}
_ORIGINAL_SCOPE_PARAMETER_COUNTS = {
    "all": ORIGINAL_PARAMETER_COUNT, "mlp": 2689597440, "attention": 943718400,
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def scope_specification(module_scope):
    """Expected selected inventory for original Qwen3-4B, independent of a loaded model."""
    require(isinstance(module_scope, str) and module_scope in MODULE_SCOPES,
            "Unknown module scope; expected all, mlp, or attention")
    projections = _SCOPE_PROJECTIONS[module_scope]
    names = [f"model.layers.{index}.{path}.weight"
             for index in range(36) for path in projections]
    parameter_count = _ORIGINAL_SCOPE_PARAMETER_COUNTS[module_scope]
    return {
        "module_scope": module_scope, "projections": list(projections), "layers": list(range(36)),
        "target_names": names, "target_modules": len(names), "target_parameter_count": parameter_count,
        "original_qwen3_target_modules": len(names),
        "original_qwen3_target_parameter_count": parameter_count,
    }


def discover_targets(model, *, require_original=True, expected_dtype=None, module_scope="all"):
    """Validate all252 decoder Linears and aliases before selecting a module scope."""
    import torch
    scope = scope_specification(module_scope)
    expected_dtype = torch.float32 if expected_dtype is None else expected_dtype
    require(expected_dtype in (torch.float32, torch.bfloat16, torch.float16),
            "Unsupported transform input dtype")
    config, layers = model.config, model.model.layers
    require(config.num_hidden_layers == 36 and len(layers) == 36
            and len({id(layer) for layer in layers}) == 36, "Need36 distinct original decoder blocks")
    hidden, intermediate = config.hidden_size, config.intermediate_size
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden // heads)
    require(all(type(value) is int and value > 0 for value in (hidden, intermediate, heads, kv_heads, head_dim)),
            "Invalid original architecture dimensions")
    if require_original:
        require((hidden, intermediate, heads, kv_heads, head_dim, config.vocab_size) ==
                (2560, 9728, 32, 8, 128, 151936) and config.tie_word_embeddings is True,
                "Wrong original Qwen3-4B shape or sharing config")
        require(model.lm_head.weight is model.model.embed_tokens.weight, "Original tied embedding/head was split")
    dimensions = {
        "self_attn.q_proj": (heads * head_dim, hidden), "self_attn.k_proj": (kv_heads * head_dim, hidden),
        "self_attn.v_proj": (kv_heads * head_dim, hidden), "self_attn.o_proj": (hidden, heads * head_dim),
        "mlp.gate_proj": (intermediate, hidden), "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    targets = {}
    for index, layer in enumerate(layers):
        actual = {name for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)}
        require(actual == set(PROJECTIONS), "Missing or additional decoder Linear target")
        for path in PROJECTIONS:
            module = layer.get_submodule(path)
            require(tuple(module.weight.shape) == dimensions[path] and module.weight.dtype == expected_dtype
                    and module.in_features == dimensions[path][1] and module.out_features == dimensions[path][0]
                    and module.in_features % GROUP_SIZE == 0, "Projection dimensions/dtype differ from config")
            targets[f"model.layers.{index}.{path}.weight"] = module.weight
    require(len(targets) == len({id(value) for value in targets.values()}) == 252, "Targets must be252 distinct weights")
    expected_names = set(targets)
    actual_aliases = {name for name, value in model.named_parameters(remove_duplicate=False)
                      if id(value) in {id(target) for target in targets.values()}}
    require(actual_aliases == expected_names, "Target weight unexpectedly aliases another parameter")
    count = sum(value.numel() for value in targets.values())
    require(not require_original or count == ORIGINAL_PARAMETER_COUNT, "Wrong original quantized parameter count")
    selected_names = set(scope["target_names"])
    return {name: weight for name, weight in targets.items() if name in selected_names}


def fingerprint(parameters):
    """Streaming exact-byte fingerprint used alongside version/alias invariants, not model serialization."""
    digest = hashlib.sha256()
    for name, value in parameters.items():
        tensor = value.detach().cpu().contiguous()
        # NumPy has no native bfloat16 dtype. Preserve the existing FP32
        # fingerprint convention and hash BF16 storage bytes without a cast.
        if str(tensor.dtype) == "torch.bfloat16":
            import torch
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), "bfloat16")).encode())
            digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
            continue
        array = tensor.numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def model_parameter_dtype(model):
    """One explicit forward dtype; buffers such as RoPE frequencies may stay FP32."""
    import torch
    dtypes = {value.dtype for value in model.parameters() if value.is_floating_point()}
    require(len(dtypes) == 1 and next(iter(dtypes)) in
            (torch.float32, torch.bfloat16, torch.float16),
            "Expected one floating model-parameter dtype")
    return next(iter(dtypes))


def transform_row_blocks(weight, transform, *, rows_per_block=32):
    """Run the unchanged FP32 primitive on complete rows, then cast back in-place.

    The model and every Parameter retain their original dtype and identity.
    At most rows_per_block full rows are temporarily promoted to FP32.
    """
    import torch
    require(type(rows_per_block) is int and rows_per_block > 0, "Invalid row block")
    dtype = weight.dtype
    maximum_fp32_norm_error, maximum_forward_norm_error = 0., 0.
    original_zeros, final_zeros = 0, 0
    with torch.no_grad():
        for start in range(0, weight.shape[0], rows_per_block):
            block = weight[start:start + rows_per_block]
            source = block.float()
            before_norm = torch.linalg.vector_norm(source, ord=2, dim=1)
            output = transform(source)
            require(output.dtype == torch.float32 and output.shape == source.shape,
                    "Primitive must return a same-shape FP32 row block")
            stored = output.to(dtype)
            require(bool(torch.isfinite(stored).all()), "Transform cast produced nonfinite weights")
            divisor = torch.where(before_norm > 0, before_norm, torch.ones_like(before_norm))
            fp32_error = (torch.linalg.vector_norm(output, ord=2, dim=1) - before_norm).abs() / divisor
            forward_error = (torch.linalg.vector_norm(stored.float(), ord=2, dim=1) - before_norm).abs() / divisor
            maximum_fp32_norm_error = max(maximum_fp32_norm_error, float(fp32_error.max()))
            maximum_forward_norm_error = max(maximum_forward_norm_error, float(forward_error.max()))
            original_zeros += int((source == 0).sum())
            final_zeros += int((stored == 0).sum())
            block.copy_(stored)
    return {
        "original_zero_weights": original_zeros, "final_zero_weights": final_zeros,
        "maximum_fp32_row_norm_relative_error": maximum_fp32_norm_error,
        "maximum_forward_row_norm_relative_error": maximum_forward_norm_error,
    }


def transform_precision(dtype):
    name = str(dtype).removeprefix("torch.")
    return {
        "transform_input_dtype": name, "transform_arithmetic_dtype": "float32",
        "transform_rows_per_block": 32, "transform_writeback_dtype": name,
        "forward_parameter_dtype": name,
        "weight_cast_order": "Load native checkpoint in forward dtype; promote row block to FP32, transform, cast back",
        "fp32_forward_equivalence_claim": False,
    }
