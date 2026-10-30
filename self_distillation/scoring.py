"""Checkpoint, precision, and cache contracts for original/weak scoring."""

import json
import os
from pathlib import Path
import tempfile

from .anchors import MODULE_SCOPES, VARIANTS


def validate_original_identity(model, revision, assets, *, tokenizer_hash=None):
    """Match the requested original model and observed tokenizer to the lock."""
    model_path = Path(model).expanduser().resolve()
    for role in ("student", "post_teacher", "pre_teacher"):
        locked = assets["models"][role]
        if (Path(locked["path"]).expanduser().resolve() != model_path
                or locked["revision"] != revision):
            raise ValueError("WISD requires all roles to share the original checkpoint and revision")
        if locked["tokenizer_hash"] != assets["tokenizer_hash"]:
            raise ValueError("WISD requires identical token IDs across all model roles")
    if tokenizer_hash is not None and tokenizer_hash != assets["tokenizer_hash"]:
        raise ValueError("Loaded tokenizer differs from the original checkpoint asset lock")


def validate_request(args, assets):
    """The original checkpoint supplies all roles, and only pre is weakened."""
    if args.dtype not in ("bfloat16", "float16"):
        raise ValueError("Scorer forwards require bfloat16 or float16")
    if args.overwrite:
        raise ValueError("Use a fresh scoring output directory")
    if args.score_field not in ("post_teacher_log_probs", "pre_teacher_log_probs",
                               "student_ref_sampled_log_probs"):
        raise ValueError("WISD scoring preserves the fixed rollout support")
    if args.attn_implementation not in (None, "sdpa"):
        raise ValueError("Original and weak scorers use the same SDPA math backend")
    if args.weak_anchor not in (None, *VARIANTS):
        raise ValueError("Unknown weak-anchor variant")
    if args.weak_anchor and args.score_field != "pre_teacher_log_probs":
        raise ValueError("Apply weakening only when scoring pre_teacher_log_probs")
    if args.weak_anchor_module_scope not in MODULE_SCOPES:
        raise ValueError("Unknown decoder projection scope")
    if not args.weak_anchor and args.weak_anchor_module_scope != "all":
        raise ValueError("A partial scope requires a weak anchor")
    source, destination = args.input.resolve(), args.output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Input and output must be separate, non-nested paths")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("Require 0 <= rank < world_size")
    if args.row_batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be positive")
    validate_original_identity(args.model, args.model_revision, assets)


def precision_identity(args, vocab_size):
    return {
        "forward_parameter_dtype": args.dtype,
        "normalization_dtype": "float32",
        "normalization_vocab_size": int(vocab_size),
        "score_storage_dtype": "float32",
        "attention_backend": "sdpa_math",
        "weak_anchor_variant": args.weak_anchor,
        "weak_anchor_module_scope": args.weak_anchor_module_scope if args.weak_anchor else None,
        "transform_arithmetic_dtype": "float32" if args.weak_anchor else None,
    }


def validate_pair_precision(metadata, precision):
    """Use matching full-vocabulary normalization for the two frozen scorers."""
    for role in ("post", "pre"):
        if metadata.get(role + "_teacher_log_probs") is None:
            continue
        field = role + "_score_precision_json"
        if not metadata.get(field):
            raise ValueError("Cached anchor scores lack precision metadata. Score a fresh pair.")
        previous = json.loads(metadata[field])
        for key in ("forward_parameter_dtype", "normalization_dtype",
                    "normalization_vocab_size", "attention_backend"):
            if previous.get(key) != precision[key]:
                raise ValueError(f"Original and weak scoring differ in {key}")


def validate_row(metadata, assets, *, weak=False):
    if metadata.get("tokenizer_hash") != assets["tokenizer_hash"]:
        raise ValueError("Trajectory tokenizer differs from the original checkpoint")
    original_revision = assets["models"]["student"]["revision"]
    if metadata.get("student_revision") != original_revision:
        raise ValueError("Trajectory student_revision differs from the original checkpoint")
    original_path = Path(assets["models"]["student"]["path"]).expanduser().resolve()
    if (not metadata.get("student_model_path")
            or Path(metadata["student_model_path"]).expanduser().resolve() != original_path):
        raise ValueError("Trajectory student_model_path differs from the original checkpoint")
    if (metadata.get("support_model_revision") is not None
            and metadata["support_model_revision"] != original_revision):
        raise ValueError("Cached support belongs to a different original model")
    for role in ("post_teacher", "pre_teacher"):
        if (metadata.get(role + "_log_probs") is not None
                and metadata.get(role + "_revision") != assets["models"][role]["revision"]):
            raise ValueError(f"Cached {role} scores belong to a different original checkpoint")
    if not metadata.get("prompt_tokens") or not metadata.get("response_tokens"):
        raise ValueError("Scoring needs nonempty cached prompt and response tokens")
    candidates = metadata.get("candidate_ids")
    if (not candidates or len(candidates) != len(metadata["response_tokens"])
            or any(len(row) != 16 or len(set(row)) != 16 for row in candidates)):
        raise ValueError("WISD uses 16 distinct original-model candidates per response token")
    if weak and any(metadata.get(key) is None for key in (
            "post_teacher_log_probs", "behavior_topk_log_probs",
            "behavior_sampled_log_probs")):
        raise ValueError("Score the clean original before the weak scorer")
    if weak and metadata.get("pre_teacher_log_probs") is not None:
        raise ValueError("Input already contains weak scores. Start from the clean-score cache.")


def admit_output(args, assets, precision):
    """Workers agree on one output identity before creating their own shards."""
    identity = {
        "schema": "wisd_scoring_v1",
        "model": str(Path(args.model).expanduser().resolve()),
        "model_revision": args.model_revision,
        "input": str(args.input.resolve()),
        "score_field": args.score_field,
        "world_size": args.world_size,
        "tokenizer_hash": assets["tokenizer_hash"],
        "precision": precision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "scoring_identity.json"
    payload = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output_dir, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_text() != payload:
                raise ValueError("Output directory belongs to a different scoring configuration")
    finally:
        temporary.unlink()
    return identity
