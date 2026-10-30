"""Offline Direct-OPD data validation and loss implementation.

The offline variant freezes response prefixes and candidate IDs in the data,
but recomputes the current student's probability over that cached support at
every optimization step. Teacher policy shifts and initial-student sampled
token log-probabilities are read from the sealed dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from slime.utils.types import RolloutBatch, Sample


SCHEMA_VERSION = "offline_direct_opd_v1"
ADAPTIVE_KL_STATE_VERSION = "offline_direct_opd_adaptive_kl_v1"
DIRECT_OPD_SAMPLE_FIELDS = (
    "candidate_ids",
    "behavior_topk_log_probs",
    "post_teacher_log_probs",
    "pre_teacher_log_probs",
    "student_ref_sampled_log_probs",
)
DIRECT_OPD_SEALED_PAYLOAD_FIELDS = frozenset(
    (
        *DIRECT_OPD_SAMPLE_FIELDS,
        "prompt_tokens",
        "response_tokens",
        "loss_mask",
        "behavior_sampled_log_probs",
        "response",
    )
)


class OfflineDirectOPDDataError(ValueError):
    """Raised when an offline Direct-OPD row violates the sealed schema."""


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA256 digest without loading a shard into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_teacher_mixture(
    model: Mapping[str, Any], *, label: str, declared_rows: int
) -> None:
    revision = model.get("revision")
    components = model.get("components")
    selection = model.get("selection")
    if not isinstance(revision, str) or len(revision) != 64:
        raise OfflineDirectOPDDataError(f"{label} mixture revision must be a 64-character digest")
    if not isinstance(components, list) or len(components) < 2:
        raise OfflineDirectOPDDataError(f"{label} mixture must declare at least two components")
    names: set[str] = set()
    component_rows = 0
    component_weight = 0.0
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise OfflineDirectOPDDataError(f"{label} mixture component {index} must be an object")
        name = component.get("name")
        weight = component.get("weight")
        rows = component.get("rows")
        component_model = component.get("model")
        source_digest = component.get("source_manifest_sha256")
        if not isinstance(name, str) or not name or name in names:
            raise OfflineDirectOPDDataError(
                f"{label} mixture component {index} has a missing or duplicate name"
            )
        names.add(name)
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise OfflineDirectOPDDataError(f"{label} mixture component {index} weight must be positive")
        if not isinstance(rows, int) or rows <= 0:
            raise OfflineDirectOPDDataError(f"{label} mixture component {index} rows must be positive")
        if not isinstance(component_model, Mapping) or not component_model.get("revision"):
            raise OfflineDirectOPDDataError(
                f"{label} mixture component {index} must lock a model revision"
            )
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise OfflineDirectOPDDataError(
                f"{label} mixture component {index} has an invalid source manifest digest"
            )
        component_rows += rows
        component_weight += float(weight)
    if component_rows != declared_rows:
        raise OfflineDirectOPDDataError(
            f"{label} mixture component rows={component_rows} do not match shard rows={declared_rows}"
        )
    if not math.isclose(component_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise OfflineDirectOPDDataError(
            f"{label} mixture weights sum to {component_weight}, expected 1.0"
        )
    if not isinstance(selection, Mapping):
        raise OfflineDirectOPDDataError(f"{label} mixture must declare its selection policy")
    if selection.get("unit") != "prompt_group":
        raise OfflineDirectOPDDataError(
            f"{label} mixture selection unit must be prompt_group"
        )
    if selection.get("group_size") != 4:
        raise OfflineDirectOPDDataError(f"{label} mixture group_size must be 4")
    strategy = selection.get("strategy")
    if strategy == "round_robin":
        return
    if strategy != "deterministic_proportional_interleave_with_cyclic_reuse":
        raise OfflineDirectOPDDataError(
            f"{label} mixture selection has unsupported strategy {strategy!r}"
        )
    if selection.get("prompt_exclusive") is not True:
        raise OfflineDirectOPDDataError(
            f"{label} weighted mixture selection must be prompt-exclusive"
        )
    if selection.get("exact_ratio_every_rollout") is not True:
        raise OfflineDirectOPDDataError(
            f"{label} weighted mixture selection must use an exact ratio every rollout"
        )
    rollout_batch_size = selection.get("rollout_batch_size")
    if (
        not isinstance(rollout_batch_size, int)
        or isinstance(rollout_batch_size, bool)
        or rollout_batch_size <= 0
        or rollout_batch_size % selection["group_size"]
    ):
        raise OfflineDirectOPDDataError(
            f"{label} weighted mixture rollout_batch_size must be positive and group-aligned"
        )
    rows_per_component = selection.get("rows_per_component_per_rollout")
    if not isinstance(rows_per_component, Mapping) or set(rows_per_component) != names:
        raise OfflineDirectOPDDataError(
            f"{label} weighted mixture must declare per-rollout rows for every component"
        )
    per_rollout_total = 0
    rollout_steps: set[int] = set()
    component_by_name = {
        str(component["name"]): component for component in components
    }
    for name, rows in rows_per_component.items():
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows <= 0
            or rows % selection["group_size"]
        ):
            raise OfflineDirectOPDDataError(
                f"{label} weighted mixture rows for {name!r} must be positive and group-aligned"
            )
        component = component_by_name[str(name)]
        component_rows = int(component["rows"])
        if component_rows % rows:
            raise OfflineDirectOPDDataError(
                f"{label} weighted mixture rows for {name!r} do not span complete rollouts"
            )
        rollout_steps.add(component_rows // rows)
        expected_weight = rows / rollout_batch_size
        if not math.isclose(
            float(component["weight"]),
            expected_weight,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise OfflineDirectOPDDataError(
                f"{label} weighted mixture weight for {name!r} does not match its per-rollout rows"
            )
        per_rollout_total += rows
    if per_rollout_total != rollout_batch_size:
        raise OfflineDirectOPDDataError(
            f"{label} weighted mixture per-component rows do not sum to rollout_batch_size"
        )
    if len(rollout_steps) != 1:
        raise OfflineDirectOPDDataError(
            f"{label} weighted mixture components do not span the same number of rollouts"
        )


def validate_sealed_manifest(path: str | Path, *, expected_top_k: int | None = None) -> dict[str, Any]:
    """Validate a sealed dataset manifest and every declared shard checksum."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Offline Direct-OPD manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise OfflineDirectOPDDataError(
            f"manifest schema_version must be {SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )
    if manifest.get("sealed") is not True:
        raise OfflineDirectOPDDataError("manifest must have sealed=true before training")
    top_k = manifest.get("top_k")
    if not isinstance(top_k, int) or top_k <= 0:
        raise OfflineDirectOPDDataError("manifest top_k must be a positive integer")
    if expected_top_k is not None and top_k != expected_top_k:
        raise OfflineDirectOPDDataError(f"manifest has top_k={top_k}, expected {expected_top_k}")
    if manifest.get("score_storage_dtype") != "float32":
        raise OfflineDirectOPDDataError(
            "manifest score_storage_dtype must be 'float32'; refusing an untyped or lossy cache"
        )
    if manifest.get("token_storage_dtype") != "int32":
        raise OfflineDirectOPDDataError("manifest token_storage_dtype must be 'int32'")
    if manifest.get("loss_mask_storage_dtype") != "bool":
        raise OfflineDirectOPDDataError("manifest loss_mask_storage_dtype must be 'bool'")

    for field in (
        "student_model",
        "post_teacher_model",
        "pre_teacher_model",
        "tokenizer_hash",
        "generation_config",
        "source_dataset_sha256",
    ):
        if manifest.get(field) in (None, "", {}):
            raise OfflineDirectOPDDataError(f"manifest is missing required field {field!r}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise OfflineDirectOPDDataError("manifest shards must be a non-empty list")
    declared_rows = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise OfflineDirectOPDDataError(f"manifest shard {index} must be an object")
        relative_path = shard.get("path")
        expected_digest = shard.get("sha256")
        row_count = shard.get("rows")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise OfflineDirectOPDDataError(f"manifest shard {index} path must be relative")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise OfflineDirectOPDDataError(f"manifest shard {index} has an invalid sha256")
        if not isinstance(row_count, int) or row_count <= 0:
            raise OfflineDirectOPDDataError(f"manifest shard {index} rows must be positive")
        shard_path = manifest_path.parent / relative_path
        if not shard_path.is_file():
            raise FileNotFoundError(f"Offline Direct-OPD shard does not exist: {shard_path}")
        actual_digest = file_sha256(shard_path)
        if actual_digest != expected_digest:
            raise OfflineDirectOPDDataError(
                f"checksum mismatch for {shard_path}: expected {expected_digest}, got {actual_digest}"
            )
        declared_rows += row_count
    if manifest.get("total_rows") != declared_rows:
        raise OfflineDirectOPDDataError(
            f"manifest total_rows={manifest.get('total_rows')} does not match shard rows={declared_rows}"
        )

    post_teacher_model = manifest["post_teacher_model"]
    if isinstance(post_teacher_model, Mapping) and post_teacher_model.get("model_type") == "mixture":
        _validate_teacher_mixture(post_teacher_model, label="post-teacher", declared_rows=declared_rows)
    elif (
        isinstance(post_teacher_model, Mapping)
        and post_teacher_model.get("model_type") == "accuracy_priority_composition"
    ):
        revision = post_teacher_model.get("revision")
        if not isinstance(revision, str) or len(revision) != 64:
            raise OfflineDirectOPDDataError(
                "accuracy-priority composition revision must be a 64-character digest"
            )
        if post_teacher_model.get("composition_schema_version") != (
            "offline_direct_opd_accuracy_priority_v1"
        ):
            raise OfflineDirectOPDDataError(
                "accuracy-priority composition has an unsupported schema version"
            )
        if post_teacher_model.get("conflict_rule") != (
            "behavior_weighted_centered_topk_plus_other_dot"
        ):
            raise OfflineDirectOPDDataError(
                "accuracy-priority composition has an unsupported conflict rule"
            )
        if post_teacher_model.get("conflict_fallback") != "accuracy":
            raise OfflineDirectOPDDataError(
                "accuracy-priority composition must fall back to accuracy on conflicts"
            )
        if post_teacher_model.get("aligned_rule") != "convex_post_logprob_interpolation":
            raise OfflineDirectOPDDataError(
                "accuracy-priority composition has an unsupported aligned-state rule"
            )
        efficiency_weight = post_teacher_model.get("efficiency_weight")
        if not isinstance(efficiency_weight, (int, float)) or not 0 <= efficiency_weight <= 1:
            raise OfflineDirectOPDDataError(
                "accuracy-priority efficiency_weight must be in [0,1]"
            )
        for name in ("accuracy_model", "efficiency_model"):
            model = post_teacher_model.get(name)
            if not isinstance(model, Mapping) or not model.get("revision"):
                raise OfflineDirectOPDDataError(
                    f"accuracy-priority composition must lock {name} revision"
                )
        if (
            post_teacher_model["accuracy_model"]["revision"]
            == post_teacher_model["efficiency_model"]["revision"]
        ):
            raise OfflineDirectOPDDataError(
                "accuracy-priority composition requires distinct anchor revisions"
            )
        for name in ("accuracy_manifest_sha256", "efficiency_manifest_sha256"):
            digest = post_teacher_model.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise OfflineDirectOPDDataError(
                    f"accuracy-priority composition has an invalid {name}"
                )
    pre_teacher_model = manifest["pre_teacher_model"]
    if isinstance(pre_teacher_model, Mapping) and pre_teacher_model.get("model_type") == "mixture":
        _validate_teacher_mixture(pre_teacher_model, label="pre-teacher", declared_rows=declared_rows)
        if not (
            isinstance(post_teacher_model, Mapping)
            and post_teacher_model.get("model_type") == "mixture"
        ):
            raise OfflineDirectOPDDataError(
                "pre-teacher mixture requires a matching post-teacher mixture"
            )
        pre_components = pre_teacher_model["components"]
        post_components = post_teacher_model["components"]
        pre_signature = [(item["name"], item["rows"], item["weight"]) for item in pre_components]
        post_signature = [(item["name"], item["rows"], item["weight"]) for item in post_components]
        if pre_signature != post_signature or pre_teacher_model["selection"] != post_teacher_model["selection"]:
            raise OfflineDirectOPDDataError(
                "pre/post teacher mixtures must use aligned components and selection"
            )
    return manifest


def _to_python_list(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise OfflineDirectOPDDataError(f"{field} must be a list, got {type(value).__name__}")
    return list(value)


def direct_opd_cpu_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    """Materialize one sealed field as a numeric CPU tensor.

    pandas with the PyArrow dtype backend represents nested parquet list
    columns as ``numpy.ndarray(dtype=object)`` values whose entries are
    fixed-width numeric arrays.  Ray serializes those object arrays slowly and
    PyTorch cannot consume them directly.  Stack the array once in the
    RolloutManager so Ray can share the resulting numeric tensor with both CP
    ranks instead of repeating the conversion in every training actor.
    """

    try:
        return torch.as_tensor(value, dtype=dtype)
    except TypeError:
        try:
            import numpy as np
        except ImportError:
            raise
        if not isinstance(value, np.ndarray) or value.dtype != np.object_:
            raise
        numpy_dtype = np.int64 if dtype == torch.long else np.float32
        try:
            numeric_value = np.asarray(np.stack(value), dtype=numpy_dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "sealed Direct-OPD object array must contain fixed-shape numeric values"
            ) from exc
        return torch.as_tensor(numeric_value, dtype=dtype)


def _validate_score_matrix(value: Any, *, field: str, response_length: int, top_k: int) -> list[list[float]]:
    rows = _to_python_list(value, field=field)
    if len(rows) != response_length:
        raise OfflineDirectOPDDataError(
            f"{field} length {len(rows)} does not match response length {response_length}"
        )

    normalized = []
    for token_index, row in enumerate(rows):
        row = _to_python_list(row, field=f"{field}[{token_index}]")
        if len(row) != top_k:
            raise OfflineDirectOPDDataError(
                f"{field}[{token_index}] has K={len(row)}, expected K={top_k}"
            )
        values = [float(item) for item in row]
        if not all(math.isfinite(item) for item in values):
            raise OfflineDirectOPDDataError(f"{field}[{token_index}] contains a non-finite value")
        if any(item > 1e-5 for item in values):
            raise OfflineDirectOPDDataError(f"{field}[{token_index}] contains a positive log-probability")
        normalized.append(values)
    return normalized


def validate_offline_direct_opd_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_top_k: int | None = None,
    require_provenance: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one ``offline_direct_opd_v1`` metadata object."""

    if not isinstance(metadata, Mapping):
        raise OfflineDirectOPDDataError(f"metadata must be a mapping, got {type(metadata).__name__}")
    if not metadata.get("is_offline_direct_opd", False):
        raise OfflineDirectOPDDataError("metadata.is_offline_direct_opd must be true")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise OfflineDirectOPDDataError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {metadata.get('schema_version')!r}"
        )

    response_tokens = [int(token) for token in _to_python_list(metadata.get("response_tokens"), field="response_tokens")]
    if not response_tokens:
        raise OfflineDirectOPDDataError("response_tokens must not be empty")
    if any(token < 0 for token in response_tokens):
        raise OfflineDirectOPDDataError("response_tokens contains a negative token ID")
    response_length = len(response_tokens)
    declared_response_length = metadata.get("response_length")
    if declared_response_length is not None and int(declared_response_length) != response_length:
        raise OfflineDirectOPDDataError(
            f"response_length={declared_response_length} does not match response_tokens length {response_length}"
        )

    prompt_tokens = [int(token) for token in _to_python_list(metadata.get("prompt_tokens"), field="prompt_tokens")]
    if not prompt_tokens:
        raise OfflineDirectOPDDataError("prompt_tokens must not be empty")
    if any(token < 0 for token in prompt_tokens):
        raise OfflineDirectOPDDataError("prompt_tokens contains a negative token ID")

    loss_mask = [int(item) for item in _to_python_list(metadata.get("loss_mask"), field="loss_mask")]
    if len(loss_mask) != response_length:
        raise OfflineDirectOPDDataError(
            f"loss_mask length {len(loss_mask)} does not match response length {response_length}"
        )
    if any(item not in (0, 1) for item in loss_mask):
        raise OfflineDirectOPDDataError("loss_mask values must be 0 or 1")
    if not any(loss_mask):
        raise OfflineDirectOPDDataError("loss_mask must contain at least one trainable token")

    candidate_rows = _to_python_list(metadata.get("candidate_ids"), field="candidate_ids")
    if len(candidate_rows) != response_length:
        raise OfflineDirectOPDDataError(
            f"candidate_ids length {len(candidate_rows)} does not match response length {response_length}"
        )
    if not candidate_rows:
        raise OfflineDirectOPDDataError("candidate_ids must not be empty")
    first_row = _to_python_list(candidate_rows[0], field="candidate_ids[0]")
    top_k = len(first_row)
    if top_k <= 0:
        raise OfflineDirectOPDDataError("candidate_ids must have K > 0")
    if expected_top_k is not None and top_k != expected_top_k:
        raise OfflineDirectOPDDataError(f"candidate_ids has K={top_k}, expected K={expected_top_k}")

    candidate_ids = []
    for token_index, row in enumerate(candidate_rows):
        row = [int(item) for item in _to_python_list(row, field=f"candidate_ids[{token_index}]")]
        if len(row) != top_k:
            raise OfflineDirectOPDDataError(
                f"candidate_ids[{token_index}] has K={len(row)}, expected K={top_k}"
            )
        if any(item < 0 for item in row):
            raise OfflineDirectOPDDataError(f"candidate_ids[{token_index}] contains a negative token ID")
        if len(set(row)) != top_k:
            raise OfflineDirectOPDDataError(f"candidate_ids[{token_index}] contains duplicate token IDs")
        candidate_ids.append(row)

    normalized: dict[str, Any] = dict(metadata)
    normalized.update(
        {
            "response_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "loss_mask": loss_mask,
            "candidate_ids": candidate_ids,
            "behavior_topk_log_probs": _validate_score_matrix(
                metadata.get("behavior_topk_log_probs"),
                field="behavior_topk_log_probs",
                response_length=response_length,
                top_k=top_k,
            ),
            "post_teacher_log_probs": _validate_score_matrix(
                metadata.get("post_teacher_log_probs"),
                field="post_teacher_log_probs",
                response_length=response_length,
                top_k=top_k,
            ),
            "pre_teacher_log_probs": _validate_score_matrix(
                metadata.get("pre_teacher_log_probs"),
                field="pre_teacher_log_probs",
                response_length=response_length,
                top_k=top_k,
            ),
        }
    )
    for token_index, behavior_row in enumerate(normalized["behavior_topk_log_probs"]):
        if any(left < right for left, right in zip(behavior_row, behavior_row[1:], strict=False)):
            raise OfflineDirectOPDDataError(
                f"behavior_topk_log_probs[{token_index}] must be sorted in descending order"
            )

    for field in ("student_ref_sampled_log_probs", "behavior_sampled_log_probs"):
        values = [float(item) for item in _to_python_list(metadata.get(field), field=field)]
        if len(values) != response_length:
            raise OfflineDirectOPDDataError(
                f"{field} length {len(values)} does not match response length {response_length}"
            )
        if not all(math.isfinite(item) for item in values):
            raise OfflineDirectOPDDataError(f"{field} contains a non-finite value")
        if any(item > 1e-5 for item in values):
            raise OfflineDirectOPDDataError(f"{field} contains a positive log-probability")
        normalized[field] = values

    if require_provenance:
        for field in (
            "sample_id",
            "prompt_id",
            "group_id",
            "response_id",
            "generation_seed",
            "generation_config_hash",
            "student_revision",
            "post_teacher_revision",
            "pre_teacher_revision",
            "tokenizer_hash",
            "response_length",
            "finish_reason",
        ):
            if metadata.get(field) in (None, ""):
                raise OfflineDirectOPDDataError(f"missing required provenance field {field!r}")

    return normalized


def attach_offline_direct_opd_fields(sample: Sample, *, expected_top_k: int | None = None) -> None:
    """Validate metadata and copy tensor-like fields onto a rollout ``Sample``."""

    normalized = validate_offline_direct_opd_metadata(sample.metadata, expected_top_k=expected_top_k)
    sample.metadata = normalized
    for field in DIRECT_OPD_SAMPLE_FIELDS:
        setattr(sample, field, normalized[field])
    sample.rollout_log_probs = normalized["behavior_sampled_log_probs"]


def _validate_trusted_sealed_metadata_shape(
    metadata: Mapping[str, Any],
    *,
    expected_top_k: int | None,
) -> Mapping[str, Any]:
    """Check cheap row-shape invariants after the sealed shard hashes were verified.

    ``validate_sealed_manifest`` hashes every parquet shard before training.
    Those immutable shards have already passed the exhaustive row validator
    during sealing, so repeating all finite/range/sort/duplicate checks for
    hundreds of millions of cached candidate values in every rollout is both
    redundant and expensive.  This gate retains the structural checks needed
    for safe tensor construction without recasting the nested Python lists.
    """

    if not isinstance(metadata, Mapping):
        raise OfflineDirectOPDDataError(f"metadata must be a mapping, got {type(metadata).__name__}")
    if not metadata.get("is_offline_direct_opd", False):
        raise OfflineDirectOPDDataError("metadata.is_offline_direct_opd must be true")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise OfflineDirectOPDDataError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {metadata.get('schema_version')!r}"
        )

    def sealed_sequence_length(value: Any, *, field: str) -> int:
        # pandas' Arrow dtype backend materializes nested parquet list fields
        # as numpy/Arrow array-like values rather than Python lists.  Avoid
        # ``tolist()`` here: the large [response_length, K] matrices must stay
        # zero-copy on the trusted sealed fast path.
        if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
            raise OfflineDirectOPDDataError(
                f"sealed {field} must be a non-empty sequence, got {type(value).__name__}"
            )
        try:
            length = len(value)
        except (TypeError, ValueError) as exc:
            raise OfflineDirectOPDDataError(
                f"sealed {field} must be a non-empty sequence, got {type(value).__name__}"
            ) from exc
        if length <= 0:
            raise OfflineDirectOPDDataError(f"sealed {field} must be a non-empty sequence")
        return length

    response_tokens = metadata.get("response_tokens")
    prompt_tokens = metadata.get("prompt_tokens")
    response_length = sealed_sequence_length(response_tokens, field="response_tokens")
    sealed_sequence_length(prompt_tokens, field="prompt_tokens")
    if int(metadata.get("response_length", response_length)) != response_length:
        raise OfflineDirectOPDDataError("sealed response_length does not match response_tokens")

    for field in (
        "loss_mask",
        "candidate_ids",
        "behavior_topk_log_probs",
        "post_teacher_log_probs",
        "pre_teacher_log_probs",
        "student_ref_sampled_log_probs",
        "behavior_sampled_log_probs",
    ):
        values = metadata.get(field)
        if sealed_sequence_length(values, field=field) != response_length:
            raise OfflineDirectOPDDataError(
                f"sealed {field} length does not match response length {response_length}"
            )
    first_candidates = metadata["candidate_ids"][0]
    top_k = sealed_sequence_length(first_candidates, field="candidate_ids[0]")
    if expected_top_k is not None and top_k != expected_top_k:
        raise OfflineDirectOPDDataError(
            f"sealed candidate_ids has K={top_k}, expected K={expected_top_k}"
        )
    return metadata


def hydrate_offline_direct_opd_sample(
    sample: Sample,
    tokenizer,
    *,
    expected_top_k: int | None = None,
    trusted_sealed: bool = False,
) -> Sample:
    """Build a completed fixed-rollout sample without importing an inference engine."""

    if trusted_sealed:
        normalized = _validate_trusted_sealed_metadata_shape(
            sample.metadata,
            expected_top_k=expected_top_k,
        )
    else:
        normalized = validate_offline_direct_opd_metadata(
            sample.metadata,
            expected_top_k=expected_top_k,
        )
    # Only normalize the small/one-dimensional fields required by Sample and
    # CP slicing.  Keep the large [response_length, K] sealed matrices in
    # their pandas/NumPy representation so hydration does not copy hundreds
    # of millions of values.
    cached_prompt_tokens = [
        int(token)
        for token in _to_python_list(normalized["prompt_tokens"], field="prompt_tokens")
    ]
    response_tokens = [
        int(token)
        for token in _to_python_list(normalized["response_tokens"], field="response_tokens")
    ]
    loss_mask = [
        int(item)
        for item in _to_python_list(normalized["loss_mask"], field="loss_mask")
    ]
    behavior_sampled_log_probs = [
        float(item)
        for item in _to_python_list(
            normalized["behavior_sampled_log_probs"],
            field="behavior_sampled_log_probs",
        )
    ]
    observed_prompt_tokens = sample.prompt
    if isinstance(observed_prompt_tokens, str):
        observed_prompt_tokens = tokenizer.encode(observed_prompt_tokens, add_special_tokens=False)
    elif hasattr(observed_prompt_tokens, "tolist"):
        observed_prompt_tokens = observed_prompt_tokens.tolist()
    if not isinstance(observed_prompt_tokens, (list, tuple)):
        raise OfflineDirectOPDDataError(
            "offline Direct-OPD prompt must be a string or token list, "
            f"got {type(observed_prompt_tokens).__name__}"
        )
    observed_prompt_tokens = [int(token) for token in observed_prompt_tokens]
    if observed_prompt_tokens != cached_prompt_tokens:
        raise OfflineDirectOPDDataError(
            "prompt tokenization differs from the sealed prompt_tokens; refusing to train on shifted prefixes"
        )

    sample.tokens = direct_opd_cpu_tensor(
        cached_prompt_tokens + response_tokens,
        dtype=torch.long,
    )
    sample.loss_mask = direct_opd_cpu_tensor(loss_mask, dtype=torch.int32)
    sample.response_length = len(response_tokens)
    sample.response = normalized.get("response", "")
    sample.status = Sample.Status.COMPLETED
    for field, dtype in (
        ("candidate_ids", torch.long),
        ("behavior_topk_log_probs", torch.float32),
        ("post_teacher_log_probs", torch.float32),
        ("pre_teacher_log_probs", torch.float32),
        ("student_ref_sampled_log_probs", torch.float32),
    ):
        setattr(
            sample,
            field,
            direct_opd_cpu_tensor(normalized[field], dtype=dtype),
        )
    sample.rollout_log_probs = direct_opd_cpu_tensor(
        behavior_sampled_log_probs,
        dtype=torch.float32,
    )
    # The large sealed arrays now live in explicit top-level Sample fields.
    # Keeping the original object arrays in metadata would make Ray serialize
    # the entire payload a second time.  Preserve flags and provenance only.
    sample.metadata = {
        key: value
        for key, value in normalized.items()
        if key not in DIRECT_OPD_SEALED_PAYLOAD_FIELDS
    }
    return sample


def low_var_kl(current_log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
    """The k3/low-variance sampled-token KL estimator used by slime/verl."""

    log_ratio = torch.clamp(
        reference_log_probs.float() - current_log_probs.float(),
        min=-20.0,
        max=20.0,
    )
    return torch.clamp(log_ratio.exp() - 1.0 - log_ratio, min=-10.0, max=10.0)


def update_kl_loss_coef_from_reward(
    current_coef: float,
    reward_mean: float,
    eps: float,
    min_coef: float,
    max_coef: float,
) -> float:
    """Apply Direct-OPD's repository-head sign controller exactly once."""

    values = {
        "current_coef": current_coef,
        "reward_mean": reward_mean,
        "eps": eps,
        "min_coef": min_coef,
        "max_coef": max_coef,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"adaptive KL inputs must be finite, got {values}")
    if not 0 <= eps < 1:
        raise ValueError(f"adaptive KL eps must satisfy 0 <= eps < 1, got {eps}")
    if min_coef < 0 or max_coef < min_coef:
        raise ValueError(f"invalid adaptive KL bounds [{min_coef}, {max_coef}]")

    if reward_mean < 0:
        new_coef = current_coef * (1 - eps)
    elif reward_mean > 0:
        new_coef = current_coef * (1 + eps)
    else:
        new_coef = current_coef
    return max(min_coef, min(max_coef, new_coef))


def adaptive_kl_state_path(save_dir: str | Path, rollout_id: int) -> Path:
    """Return the state file tied to one zero-based saved rollout ID."""

    return Path(save_dir) / "offline_direct_opd_kl_state" / f"iter_{rollout_id:07d}.json"


def save_adaptive_kl_state(args: Namespace, rollout_id: int) -> Path:
    """Atomically persist the adaptive coefficient alongside a model checkpoint."""

    if args.save is None:
        raise ValueError("adaptive Offline Direct-OPD requires --save")
    state_path = adaptive_kl_state_path(args.save, rollout_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ADAPTIVE_KL_STATE_VERSION,
        "rollout_id": int(rollout_id),
        "kl_coef": float(args.offline_direct_opd_kl_coef),
        "eps": float(args.offline_direct_opd_kl_eps),
        "min_coef": float(args.offline_direct_opd_kl_min),
        "max_coef": float(args.offline_direct_opd_kl_max),
    }
    temporary_path = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, state_path)
    return state_path


def load_adaptive_kl_state(args: Namespace, rollout_id: int) -> float:
    """Restore and validate the coefficient matching a resumed checkpoint."""

    state_path = adaptive_kl_state_path(args.load, rollout_id)
    if not state_path.is_file():
        raise FileNotFoundError(
            "adaptive Offline Direct-OPD checkpoint is missing its controller state: "
            f"{state_path}"
        )
    with state_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != ADAPTIVE_KL_STATE_VERSION:
        raise ValueError(f"unsupported adaptive KL state schema in {state_path}")
    if payload.get("rollout_id") != rollout_id:
        raise ValueError(
            f"adaptive KL state rollout_id={payload.get('rollout_id')} does not match checkpoint {rollout_id}"
        )
    expected = {
        "eps": float(args.offline_direct_opd_kl_eps),
        "min_coef": float(args.offline_direct_opd_kl_min),
        "max_coef": float(args.offline_direct_opd_kl_max),
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"adaptive KL controller config changed across resume: saved={observed}, current={expected}"
        )
    coefficient = float(payload["kl_coef"])
    if not math.isfinite(coefficient) or not expected["min_coef"] <= coefficient <= expected["max_coef"]:
        raise ValueError(f"invalid saved adaptive KL coefficient {coefficient} in {state_path}")
    return coefficient


def offline_direct_opd_terms(
    candidate_log_probs: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    post_teacher_log_probs: torch.Tensor,
    pre_teacher_log_probs: torch.Tensor,
    student_ref_sampled_log_probs: torch.Tensor,
    behavior_topk_log_probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-token loss and diagnostic terms without reducing over tokens."""

    if candidate_log_probs.ndim != 2:
        raise ValueError(f"candidate_log_probs must have shape [T,K], got {candidate_log_probs.shape}")
    expected_matrix_shape = candidate_log_probs.shape
    for name, value in (
        ("post_teacher_log_probs", post_teacher_log_probs),
        ("pre_teacher_log_probs", pre_teacher_log_probs),
    ):
        if value.shape != expected_matrix_shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected_matrix_shape}")
    token_count = candidate_log_probs.shape[0]
    for name, value in (
        ("sampled_log_probs", sampled_log_probs),
        ("student_ref_sampled_log_probs", student_ref_sampled_log_probs),
    ):
        if value.shape != (token_count,):
            raise ValueError(f"{name} has shape {value.shape}, expected {(token_count,)}")
    if behavior_topk_log_probs is not None and behavior_topk_log_probs.shape != expected_matrix_shape:
        raise ValueError(
            f"behavior_topk_log_probs has shape {behavior_topk_log_probs.shape}, expected {expected_matrix_shape}"
        )

    current_candidate_log_probs = candidate_log_probs.float()
    candidate_weights = torch.softmax(current_candidate_log_probs.detach(), dim=-1)
    teacher_delta = (post_teacher_log_probs.float() - pre_teacher_log_probs.float()).detach()
    candidate_rewards = (candidate_weights * teacher_delta).detach()
    # Match Direct-OPD's on-policy Top-K PPO surrogate exactly at each offline
    # update: old_log_prob is the detached current log-probability, hence the
    # forward ratio is one while its gradient is d ratio / d log_prob = one.
    # This has the same analytical gradient as -reward * log_prob, but also
    # preserves the official step-zero loss scalar and metric reducer.
    on_policy_ratio = torch.exp(
        current_candidate_log_probs - current_candidate_log_probs.detach()
    )
    direct_loss = -(candidate_rewards * on_policy_ratio).sum(dim=-1)
    kl = low_var_kl(sampled_log_probs, student_ref_sampled_log_probs)

    terms = {
        "direct_loss": direct_loss,
        "kl_loss": kl,
        "weighted_reward_mean": candidate_rewards.mean(dim=-1),
        "weighted_reward_token_mean": candidate_rewards.sum(dim=-1),
        "cached_support_mass": current_candidate_log_probs.logsumexp(dim=-1).exp(),
        "positive_reward_mass": torch.where(
            teacher_delta > 0,
            candidate_weights,
            torch.zeros_like(candidate_weights),
        ).sum(dim=-1),
    }
    if behavior_topk_log_probs is not None:
        terms["behavior_topk_abs_diff"] = (
            current_candidate_log_probs.detach() - behavior_topk_log_probs.float()
        ).abs().mean(dim=-1)
    return terms


def _topk_plus_other_distribution(log_probs: torch.Tensor) -> torch.Tensor:
    """Collapse a full-vocabulary distribution into cached Top-K plus ``other``.

    ``log_probs`` must be full-vocabulary log-probabilities gathered at the
    cached candidate IDs, rather than logits normalized only within Top-K.
    The extra bucket preserves the probability mass outside the cache so a
    target loss cannot silently force all mass onto the cached candidates.
    """

    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must have shape [T,K], got {log_probs.shape}")
    candidate_probs = log_probs.float().exp()
    other_probs = (1.0 - candidate_probs.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    distribution = torch.cat((candidate_probs, other_probs), dim=-1)
    return distribution / distribution.sum(dim=-1, keepdim=True)


def offline_direct_opd_tilted_target_terms(
    candidate_log_probs: torch.Tensor,
    behavior_topk_log_probs: torch.Tensor,
    post_teacher_log_probs: torch.Tensor,
    pre_teacher_log_probs: torch.Tensor,
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Build a self-correcting offline target from the cached policy shift.

    On every frozen prefix we form the KL-regularized policy-improvement
    target ``q ∝ pi_behavior * exp(delta / alpha)``.  Cached candidates are
    represented individually and all remaining vocabulary mass is represented
    by one zero-reward ``other`` bucket.  The squared log-probability residual
    is the Rao-Blackwellized analogue of Lightning OPD on this fixed target:
    its gradient matches the local reward gradient at initialization and
    vanishes when the current policy reaches the target.
    """

    if not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError(f"tilted-target alpha must be positive and finite, got {alpha}")
    if candidate_log_probs.ndim != 2:
        raise ValueError(f"candidate_log_probs must have shape [T,K], got {candidate_log_probs.shape}")
    expected_shape = candidate_log_probs.shape
    for name, value in (
        ("behavior_topk_log_probs", behavior_topk_log_probs),
        ("post_teacher_log_probs", post_teacher_log_probs),
        ("pre_teacher_log_probs", pre_teacher_log_probs),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected_shape}")

    current_candidate_log_probs = candidate_log_probs.float()
    behavior_candidate_log_probs = behavior_topk_log_probs.float().detach()
    teacher_delta = (post_teacher_log_probs.float() - pre_teacher_log_probs.float()).detach()

    current_distribution = _topk_plus_other_distribution(current_candidate_log_probs)
    behavior_distribution = _topk_plus_other_distribution(behavior_candidate_log_probs).detach()
    zero_other_reward = torch.zeros_like(teacher_delta[..., :1])
    bucket_rewards = torch.cat((teacher_delta, zero_other_reward), dim=-1)
    target_logits = behavior_distribution.clamp_min(1e-30).log() + bucket_rewards / float(alpha)
    target_log_probs = target_logits.log_softmax(dim=-1).detach()
    target_distribution = target_log_probs.exp()
    current_log_probs = current_distribution.clamp_min(1e-30).log()

    log_residual = current_log_probs - target_log_probs
    # Direct-OPD renormalizes its candidate weights within Top-K.  Dividing by
    # the cached Top-K support mass makes the step-zero log-probability
    # gradient exactly equal to that official surrogate (including its effect
    # on the non-candidate vocabulary), while retaining q as the zero-gradient
    # fixed point.
    behavior_support_mass = behavior_distribution[..., :-1].sum(dim=-1)
    tilted_target_loss = 0.5 * float(alpha) * (
        behavior_distribution * log_residual.square()
    ).sum(dim=-1) / behavior_support_mass.clamp_min(1e-8)

    candidate_weights = torch.softmax(current_candidate_log_probs.detach(), dim=-1)
    candidate_rewards = (candidate_weights * teacher_delta).detach()
    detached_current = current_distribution.detach()
    return {
        "tilted_target_loss": tilted_target_loss,
        "target_forward_kl": (
            target_distribution * (target_log_probs - current_log_probs.detach())
        ).sum(dim=-1),
        "target_reverse_kl": (
            detached_current * (current_log_probs.detach() - target_log_probs)
        ).sum(dim=-1),
        "target_tv": 0.5 * (detached_current - target_distribution).abs().sum(dim=-1),
        "target_entropy": -(target_distribution * target_log_probs).sum(dim=-1),
        "weighted_reward_mean": candidate_rewards.mean(dim=-1),
        "weighted_reward_token_mean": candidate_rewards.sum(dim=-1),
        "cached_support_mass": current_candidate_log_probs.detach().logsumexp(dim=-1).exp(),
        "positive_reward_mass": torch.where(
            teacher_delta > 0,
            candidate_weights,
            torch.zeros_like(candidate_weights),
        ).sum(dim=-1),
        "behavior_topk_abs_diff": (
            current_candidate_log_probs.detach() - behavior_candidate_log_probs
        ).abs().mean(dim=-1),
    }


def offline_direct_opd_reward_sum_and_count(
    candidate_log_probs: torch.Tensor,
    post_teacher_log_probs: torch.Tensor,
    pre_teacher_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the exact numerator/count of ``weighted_reward_mean``."""

    if candidate_log_probs.shape != post_teacher_log_probs.shape:
        raise ValueError("candidate and post-teacher log-probability shapes differ")
    if candidate_log_probs.shape != pre_teacher_log_probs.shape:
        raise ValueError("candidate and pre-teacher log-probability shapes differ")
    if candidate_log_probs.ndim != 2 or loss_mask.shape != (candidate_log_probs.shape[0],):
        raise ValueError("expected candidate scores [T,K] and loss mask [T]")
    candidate_weights = torch.softmax(candidate_log_probs.detach().float(), dim=-1)
    teacher_delta = (post_teacher_log_probs.float() - pre_teacher_log_probs.float()).detach()
    candidate_rewards = candidate_weights * teacher_delta
    mask = loss_mask.to(device=candidate_rewards.device, dtype=torch.bool)
    reward_sum = candidate_rewards[mask].double().sum()
    reward_count = mask.sum().double() * candidate_rewards.shape[-1]
    return reward_sum, reward_count


def compute_offline_direct_opd_objective(
    candidate_log_probs: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    post_teacher_log_probs: torch.Tensor,
    pre_teacher_log_probs: torch.Tensor,
    student_ref_sampled_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    kl_coef: float,
    behavior_topk_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the token-mean offline Direct-OPD objective for tests and parity checks."""

    terms = offline_direct_opd_terms(
        candidate_log_probs,
        sampled_log_probs,
        post_teacher_log_probs,
        pre_teacher_log_probs,
        student_ref_sampled_log_probs,
        behavior_topk_log_probs,
    )
    mask = loss_mask.to(device=candidate_log_probs.device, dtype=torch.float32)
    if mask.shape != (candidate_log_probs.shape[0],):
        raise ValueError(f"loss_mask has shape {mask.shape}, expected {(candidate_log_probs.shape[0],)}")
    denominator = mask.sum().clamp_min(1.0)

    reduced = {name: (value * mask).sum() / denominator for name, value in terms.items()}
    total_loss = reduced["direct_loss"] + float(kl_coef) * reduced["kl_loss"]
    metrics = {"loss": total_loss.detach(), **{name: value.detach() for name, value in reduced.items()}}
    return total_loss, metrics


class _VocabParallelCandidateLogProbs(torch.autograd.Function):
    """Exact arbitrary-candidate log-probs with token-chunked recomputation.

    Keeping a full-vocabulary softmax for backward is prohibitively expensive
    for long responses.  For example, 12K Qwen3 tokens over a 151,936-token
    vocabulary need more than 7 GiB in fp32 for one tensor.  The forward pass
    therefore stores only the input logits, per-token log-normalizers, and the
    small candidate index tensors.  Backward reconstructs softmax rows in
    bounded token chunks while writing directly into the required input
    gradient.
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        tp_group,
        chunk_size: int,
        action_vocab_size: int,
    ):
        if vocab_parallel_logits.ndim != 2 or candidate_ids.ndim != 2:
            raise ValueError(
                "vocab_parallel_logits and candidate_ids must have shapes [T,V_local] and [T,K], "
                f"got {vocab_parallel_logits.shape} and {candidate_ids.shape}"
            )
        if vocab_parallel_logits.shape[0] != candidate_ids.shape[0]:
            raise ValueError("candidate and logit token dimensions do not match")
        if chunk_size <= 0:
            raise ValueError(f"candidate log-prob chunk size must be positive, got {chunk_size}")

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size(tp_group) if distributed else 1
        rank = dist.get_rank(tp_group) if distributed else 0
        local_vocab_size = vocab_parallel_logits.shape[-1]
        vocab_start = rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size
        global_vocab_size = local_vocab_size * world_size
        if action_vocab_size <= 0 or action_vocab_size > global_vocab_size:
            raise ValueError(
                f"action vocabulary must lie in [1,{global_vocab_size}], "
                f"got {action_vocab_size}"
            )
        local_action_vocab_size = max(
            0,
            min(local_vocab_size, action_vocab_size - vocab_start),
        )
        if candidate_ids.numel() and (
            candidate_ids.min().item() < 0
            or candidate_ids.max().item() >= action_vocab_size
        ):
            raise ValueError(
                f"candidate token ID falls outside normalized action vocabulary "
                f"[0,{action_vocab_size})"
            )

        token_count = vocab_parallel_logits.shape[0]
        if local_action_vocab_size:
            logits_max = torch.cat(
                [
                    vocab_parallel_logits[
                        start : start + chunk_size,
                        :local_action_vocab_size,
                    ]
                    .float()
                    .max(dim=-1, keepdim=True)
                    .values
                    for start in range(0, token_count, chunk_size)
                ],
                dim=0,
            )
        else:
            logits_max = torch.full(
                (token_count, 1),
                float("-inf"),
                device=vocab_parallel_logits.device,
                dtype=torch.float32,
            )
        if distributed:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        if local_action_vocab_size:
            sum_exp = torch.cat(
                [
                    (
                        vocab_parallel_logits[
                            start : start + chunk_size,
                            :local_action_vocab_size,
                        ].float()
                        - logits_max[start : start + chunk_size]
                    )
                    .exp_()
                    .sum(dim=-1, keepdim=True)
                    for start in range(0, token_count, chunk_size)
                ],
                dim=0,
            )
        else:
            sum_exp = torch.zeros_like(logits_max)
        if distributed:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        target_mask = (candidate_ids < vocab_start) | (candidate_ids >= vocab_end)
        local_target_ids = (candidate_ids - vocab_start).masked_fill(target_mask, 0)
        target_logits = vocab_parallel_logits.gather(dim=-1, index=local_target_ids).float()
        target_logits = target_logits.masked_fill(target_mask, 0.0)
        if distributed:
            dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)
        log_normalizer = logits_max + sum_exp.log()
        log_probs = target_logits - log_normalizer

        ctx.save_for_backward(
            vocab_parallel_logits,
            log_normalizer,
            local_target_ids,
            target_mask,
        )
        ctx.chunk_size = chunk_size
        ctx.input_dtype = vocab_parallel_logits.dtype
        ctx.local_action_vocab_size = local_action_vocab_size
        return log_probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            vocab_parallel_logits,
            log_normalizer,
            local_target_ids,
            target_mask,
        ) = ctx.saved_tensors
        grad_logits = torch.zeros_like(vocab_parallel_logits)
        for start in range(0, vocab_parallel_logits.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, vocab_parallel_logits.shape[0])
            grad_output_chunk = grad_output[start:end].float()
            valid_width = ctx.local_action_vocab_size
            if valid_width:
                grad_chunk = (
                    vocab_parallel_logits[start:end, :valid_width].float()
                    - log_normalizer[start:end]
                ).exp_()
                grad_chunk.mul_(-grad_output_chunk.sum(dim=-1, keepdim=True))
                local_ids = local_target_ids[start:end]
                local_mask = target_mask[start:end] | (local_ids >= valid_width)
                safe_local_ids = local_ids.masked_fill(local_mask, 0)
                grad_chunk.scatter_add_(
                    dim=-1,
                    index=safe_local_ids,
                    src=grad_output_chunk.masked_fill(local_mask, 0.0),
                )
                grad_logits[start:end, :valid_width].copy_(
                    grad_chunk.to(ctx.input_dtype)
                )
        return grad_logits, None, None, None, None


def vocab_parallel_candidate_log_probs(
    vocab_parallel_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    tp_group=None,
    *,
    chunk_size: int = 256,
    action_vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute exact ``[T,K]`` log-probs without gathering or saving the vocabulary."""

    if action_vocab_size is None:
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size(tp_group) if distributed else 1
        action_vocab_size = vocab_parallel_logits.shape[-1] * world_size
    return _VocabParallelCandidateLogProbs.apply(
        vocab_parallel_logits,
        candidate_ids.long(),
        tp_group,
        int(chunk_size),
        int(action_vocab_size),
    )


def _candidate_log_prob_chunk_size(args: Namespace) -> int:
    """Use Slime's shared log-prob knob, with a memory-safe Direct-OPD default."""

    configured = int(getattr(args, "log_probs_chunk_size", -1))
    return configured if configured > 0 else 256


def dense_candidate_log_probs(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    action_vocab_size: int | None = None,
) -> torch.Tensor:
    """Gather full-vocabulary candidate log-probabilities for HF/FSDP.

    Unlike a Top-K-only softmax, this normalizes over the complete model
    vocabulary so the cached support mass and the aggregate ``other`` bucket
    retain their intended probability semantics.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [T,V], got {logits.shape}")
    if candidate_ids.ndim != 2:
        raise ValueError(
            f"candidate_ids must have shape [T,K], got {candidate_ids.shape}"
        )
    if logits.shape[0] != candidate_ids.shape[0]:
        raise ValueError(
            f"logits/candidate length mismatch: {logits.shape[0]} != "
            f"{candidate_ids.shape[0]}"
        )
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    if action_vocab_size is None:
        action_vocab_size = logits.shape[-1]
    if action_vocab_size <= 0 or action_vocab_size > logits.shape[-1]:
        raise ValueError(
            f"action vocabulary must lie in [1,{logits.shape[-1]}], "
            f"got {action_vocab_size}"
        )
    if candidate_ids.numel() and (
        int(candidate_ids.min()) < 0
        or int(candidate_ids.max()) >= action_vocab_size
    ):
        raise ValueError(
            f"candidate IDs must lie in [0,{action_vocab_size}), got "
            f"[{int(candidate_ids.min())},{int(candidate_ids.max())}]"
        )
    scaled_logits = logits[..., :action_vocab_size].float() / float(temperature)
    candidate_ids = candidate_ids.to(device=scaled_logits.device, dtype=torch.long)
    return scaled_logits.gather(
        dim=-1,
        index=candidate_ids,
    ) - scaled_logits.logsumexp(dim=-1, keepdim=True)


@torch.no_grad()
def get_offline_direct_opd_reward_stats(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    batch: RolloutBatch,
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Return numerator/count for the official whole-rollout adaptive metric."""

    del with_entropy
    if not non_loss_data:
        raise ValueError("adaptive KL metric must be computed in a forward-only pass")

    from megatron.core import mpu

    from slime.backends.megatron_utils.loss import get_responses

    if mpu.get_context_parallel_world_size() != 1:
        raise ValueError("adaptive Offline Direct-OPD currently requires --context-parallel-size 1")

    reward_sum = torch.zeros((), dtype=torch.float64, device=logits.device)
    reward_count = torch.zeros((), dtype=torch.float64, device=logits.device)
    response_chunks = get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    tp_group = mpu.get_tensor_model_parallel_group()
    for sample_index, (logits_chunk, _) in enumerate(response_chunks):
        candidate_ids = batch["candidate_ids"][sample_index].to(device=logits_chunk.device, dtype=torch.long)
        candidate_log_probs = vocab_parallel_candidate_log_probs(
            logits_chunk,
            candidate_ids,
            tp_group,
            chunk_size=_candidate_log_prob_chunk_size(args),
            action_vocab_size=int(
                getattr(
                    args,
                    "offline_direct_opd_action_vocab_size",
                    getattr(args, "padded_vocab_size", logits_chunk.shape[-1]),
                )
            ),
        )
        sample_reward_sum, sample_reward_count = offline_direct_opd_reward_sum_and_count(
            candidate_log_probs,
            batch["post_teacher_log_probs"][sample_index].to(logits_chunk.device),
            batch["pre_teacher_log_probs"][sample_index].to(logits_chunk.device),
            batch["loss_masks"][sample_index].to(logits_chunk.device),
        )
        reward_sum += sample_reward_sum
        reward_count += sample_reward_count

    return {
        "offline_direct_opd_reward_sum": [reward_sum],
        "offline_direct_opd_reward_count": [reward_count],
    }


def megatron_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Megatron ``custom_loss`` entry point for Offline Direct-OPD."""

    from megatron.core import mpu

    from slime.backends.megatron_utils.loss import get_responses
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    if not args.calculate_per_token_loss:
        raise ValueError("Offline Direct-OPD requires --calculate-per-token-loss for token-mean reduction")

    required_fields = DIRECT_OPD_SAMPLE_FIELDS
    missing = [field for field in required_fields if batch.get(field) is None]
    if missing:
        raise KeyError(f"Offline Direct-OPD batch is missing fields: {', '.join(missing)}")

    response_chunks = list(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
        )
    )
    token_terms: dict[str, list[torch.Tensor]] = {}
    tp_group = mpu.get_tensor_model_parallel_group()

    loss_mode = getattr(args, "offline_direct_opd_loss_mode", "policy_gradient")
    for sample_index, (logits_chunk, sampled_tokens) in enumerate(response_chunks):
        candidate_ids = batch["candidate_ids"][sample_index].to(device=logits_chunk.device, dtype=torch.long)
        if candidate_ids.shape[0] != logits_chunk.shape[0]:
            raise ValueError(
                f"sample {sample_index}: candidate length {candidate_ids.shape[0]} != logits length {logits_chunk.shape[0]}"
            )
        candidate_log_probs = vocab_parallel_candidate_log_probs(
            logits_chunk,
            candidate_ids,
            tp_group,
            chunk_size=_candidate_log_prob_chunk_size(args),
            action_vocab_size=int(
                getattr(
                    args,
                    "offline_direct_opd_action_vocab_size",
                    getattr(args, "padded_vocab_size", logits_chunk.shape[-1]),
                )
            ),
        )
        if loss_mode == "policy_gradient":
            sampled_log_probs, _ = calculate_log_probs_and_entropy(
                logits_chunk,
                sampled_tokens,
                tp_group,
                with_entropy=False,
                chunk_size=args.log_probs_chunk_size,
            )
            terms = offline_direct_opd_terms(
                candidate_log_probs,
                sampled_log_probs.squeeze(-1),
                batch["post_teacher_log_probs"][sample_index].to(logits_chunk.device),
                batch["pre_teacher_log_probs"][sample_index].to(logits_chunk.device),
                batch["student_ref_sampled_log_probs"][sample_index].to(logits_chunk.device),
                batch["behavior_topk_log_probs"][sample_index].to(logits_chunk.device),
            )
        elif loss_mode == "tilted_target":
            terms = offline_direct_opd_tilted_target_terms(
                candidate_log_probs,
                batch["behavior_topk_log_probs"][sample_index].to(logits_chunk.device),
                batch["post_teacher_log_probs"][sample_index].to(logits_chunk.device),
                batch["pre_teacher_log_probs"][sample_index].to(logits_chunk.device),
                alpha=float(args.offline_direct_opd_kl_coef),
            )
        else:
            raise ValueError(f"unknown Offline Direct-OPD loss mode {loss_mode!r}")
        for name, value in terms.items():
            token_terms.setdefault(name, []).append(value)

    reduced = {name: sum_of_sample_mean(torch.cat(values, dim=0)) for name, values in token_terms.items()}
    alpha = float(args.offline_direct_opd_kl_coef)
    if loss_mode == "policy_gradient":
        loss = reduced["direct_loss"] + alpha * reduced["kl_loss"]
        log_keys = (
            "direct_loss",
            "kl_loss",
            "weighted_reward_mean",
            "weighted_reward_token_mean",
            "cached_support_mass",
            "positive_reward_mass",
            "behavior_topk_abs_diff",
        )
        coefficient_name = "kl_coef"
    else:
        loss = reduced["tilted_target_loss"]
        log_keys = (
            "tilted_target_loss",
            "target_forward_kl",
            "target_reverse_kl",
            "target_tv",
            "target_entropy",
            "weighted_reward_mean",
            "weighted_reward_token_mean",
            "cached_support_mass",
            "positive_reward_mass",
            "behavior_topk_abs_diff",
        )
        coefficient_name = "tilt_alpha"
    # Match the reducer input shape used by every logged per-token term.  A
    # rollout batch can contain multiple response chunks with different
    # lengths, so using only the first chunk would report a sample-dependent
    # coefficient under ``sum_of_sample_mean``.
    coefficient_tokens = torch.full_like(
        torch.cat(next(iter(token_terms.values())), dim=0),
        alpha,
    )
    log = {"loss": loss.detach()}
    log.update({name: reduced[name].detach() for name in log_keys})
    log[coefficient_name] = sum_of_sample_mean(coefficient_tokens).detach()
    return loss, log
