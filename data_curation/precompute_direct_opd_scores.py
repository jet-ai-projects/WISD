#!/usr/bin/env python3
"""Score cached original-model actions with the frozen original or weak model."""

import argparse
import json
import math
import os
import sys
from bisect import bisect_left
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import OFFLINE_STORAGE_TYPES, atomic_output, canonical_hash, parquet_paths
from data_curation.composition import flatten_metadata, replace_metadata_fields
from data_curation.prepare_direct_opd_assets import TOKEN_PROJECTION_MODE, resolve_model_vocab_size
from self_distillation.anchors import MODULE_SCOPES, VARIANTS
from self_distillation.scoring import (
    admit_output, precision_identity, validate_original_identity,
    validate_pair_precision, validate_request, validate_row,
)

SCORE_FIELDS = {
    "post_teacher_log_probs": "post_teacher_revision",
    "pre_teacher_log_probs": "pre_teacher_revision",
    "student_ref_sampled_log_probs": "student_revision",
    "student_support_topk": "student_revision",
}
REQUIRED_SCORED_FIELDS = tuple(field for field in SCORE_FIELDS if field != "student_support_topk")


def normalize_legacy_vllm_yarn_config(config):
    """Convert vLLM's YaRN multiplier to Transformers' complete rotary scale."""
    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, Mapping):
        return False
    if params.get("rope_type", params.get("type")) != "yarn" or "attn_factor" not in params:
        return False
    params = dict(params)
    params["attention_factor"] = (1.0 + 0.1 * math.log(float(params["factor"]))) * float(params.pop("attn_factor"))
    config.rope_parameters = params
    return True


def cast_offline_direct_opd_storage(table):
    """Store token IDs as Int32, scores as Float32, and masks as bool."""
    if "metadata" not in table.column_names:
        return table
    metadata = flatten_metadata(table)
    replacements = {
        field.name: metadata.field(field.name).cast(OFFLINE_STORAGE_TYPES[field.name])
        for field in metadata.type
        if field.name in OFFLINE_STORAGE_TYPES and not pa.types.is_null(field.type)
    }
    return replace_metadata_fields(table, replacements)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--asset-lock", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-field", required=True, choices=REQUIRED_SCORED_FIELDS)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--row-batch-size", type=int, default=8)
    parser.add_argument("--attn-implementation", choices=["sdpa"], default="sdpa")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--weak-anchor", choices=VARIANTS,
                        help="Transform only the frozen pre scorer loaded from the original checkpoint")
    parser.add_argument("--weak-anchor-module-scope", choices=MODULE_SCOPES, default="all")
    return parser.parse_args()


def write_scored_shard(source_path, destination, *, score_rows, row_batch_size, storage_row_group_size=128):
    """Keep Python batches small while buffering larger Arrow row groups."""
    writer = None
    written = pending_rows = 0
    pending = []

    def flush():
        nonlocal writer, pending_rows
        if pending:
            table = pa.concat_tables(pending)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=storage_row_group_size)
            pending.clear()
            pending_rows = 0

    with atomic_output(destination) as temporary:
        try:
            for batch in pq.ParquetFile(source_path).iter_batches(batch_size=row_batch_size):
                rows = batch.to_pylist()
                score_rows(rows, written)
                pending.append(cast_offline_direct_opd_storage(pa.Table.from_pylist(rows)))
                pending_rows += len(rows)
                written += len(rows)
                if pending_rows >= storage_row_group_size:
                    flush()
            flush()
        finally:
            if writer is not None:
                writer.close()
    return written


def resolve_normalization_vocab_size(compatibility, *, score_role, actual_vocab_size):
    """Drop padded vocabulary rows before computing log probabilities."""
    if compatibility.get("mode") == TOKEN_PROJECTION_MODE:
        return int(compatibility.get("normalization_vocab_sizes", {}).get(score_role, actual_vocab_size))
    return int(compatibility.get("normalization_vocab_size") or compatibility["model_vocab_size"])


class ExactTokenStringProjection:
    """Map atomic actions exactly; expand observed context in ByteLevel symbols."""

    def __init__(self, student_tokenizer, teacher_tokenizer):
        student_vocab = student_tokenizer.get_vocab()
        teacher_vocab = teacher_tokenizer.get_vocab()
        self.teacher_tokenizer = teacher_tokenizer
        self.symbols = {int(token_id): symbol for symbol, token_id in student_vocab.items()}
        self.added_ids = set(getattr(student_tokenizer, "get_added_vocab", lambda: {})().values())
        self.atomic_ids = {
            token_id: int(teacher_vocab[symbol])
            for token_id, symbol in self.symbols.items()
            if symbol in teacher_vocab
        }
        self.context_cache = {token_id: [target] for token_id, target in self.atomic_ids.items()}

    def context_ids(self, student_id):
        student_id = int(student_id)
        if student_id not in self.context_cache:
            symbol = self.symbols[student_id]
            # Added tokens contain literal text, ordinary BPE tokens contain
            # byte symbols that must not be decoded as individual UTF-8 strings.
            if student_id in self.added_ids:
                ids = self.teacher_tokenizer.encode(symbol, add_special_tokens=False)
            else:
                ids = [piece.id for piece in self.teacher_tokenizer.backend_tokenizer.model.tokenize(symbol)]
            self.context_cache[student_id] = ids
        return self.context_cache[student_id]

    def prepare_row(self, metadata):
        prompt = [target for token in metadata["prompt_tokens"] for target in self.context_ids(token)]
        response = [self.context_ids(token) for token in metadata["response_tokens"]]
        context = prompt + [target for tokens in response for target in tokens]
        indices = []
        boundary = len(prompt)
        for tokens in response:
            indices.append(boundary - 1)
            boundary += len(tokens)
        targets, mask = [], []
        for candidates in metadata["candidate_ids"]:
            mapped = [self.atomic_ids.get(int(token)) for token in candidates]
            mask.append(all(token is not None for token in mapped))
            # Zero is only a tensor-shape placeholder for masked actions.
            targets.append([0 if token is None else token for token in mapped])
        return context, targets, indices, mask


def _prepare_scoring_row(metadata, score_field, token_projection=None):
    if token_projection is not None:
        context, targets, indices, mask = token_projection.prepare_row(metadata)
        metadata["token_projection_valid_mask"] = mask
    else:
        prompt = [int(token) for token in metadata["prompt_tokens"]]
        response = [int(token) for token in metadata["response_tokens"]]
        context = prompt + response[:-1]
        indices = list(range(len(prompt) - 1, len(context)))
        targets = (
            [[token] for token in response]
            if score_field == "student_ref_sampled_log_probs"
            else metadata["candidate_ids"]
        )
    return context[: indices[-1] + 1], targets, indices


def _score_logits(logits, targets, response_tokens, score_field, support_width):
    import torch

    if score_field == "student_support_topk":
        normalizer = logits.logsumexp(dim=-1, keepdim=True)
        support_logits, candidate_ids = logits.topk(support_width, dim=-1)
        response_ids = torch.tensor(response_tokens, dtype=torch.long, device=logits.device).unsqueeze(-1)
        sampled = logits.gather(dim=-1, index=response_ids) - normalizer
        return {
            "candidate_ids": candidate_ids.cpu().tolist(),
            "behavior_topk_log_probs": (support_logits - normalizer).cpu().tolist(),
            "student_ref_sampled_log_probs": sampled.squeeze(-1).cpu().tolist(),
        }
    target_ids = torch.tensor(targets, dtype=torch.long, device=logits.device)
    scores = logits.log_softmax(dim=-1).gather(dim=-1, index=target_ids)
    return (scores[:, 0] if score_field == "student_ref_sampled_log_probs" else scores).cpu().tolist()


def score_sequence(
    model,
    metadata,
    score_field,
    *,
    device,
    chunk_size,
    normalization_vocab_size=None,
    token_projection=None,
    _prepared=None,
):
    """Stream cached logits at ordinary or sparse cross-tokenizer boundaries."""
    import torch

    context, targets, indices = (
        _prepare_scoring_row(metadata, score_field, token_projection) if _prepared is None else _prepared
    )
    width = len(metadata["candidate_ids"][0]) if score_field == "student_support_topk" else 0
    past_key_values = None
    chunks, cursor = [], 0
    with torch.inference_mode():
        for start in range(0, len(context), chunk_size):
            chunk = context[start : start + chunk_size]
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            end = bisect_left(indices, start + len(chunk), lo=cursor)
            if end > cursor:
                local_indices = torch.tensor(
                    [index - start for index in indices[cursor:end]], dtype=torch.long, device=device
                )
                logits = output.logits[0].index_select(0, local_indices).float()[..., :normalization_vocab_size]
                chunks.append(
                    _score_logits(
                        logits,
                        targets[cursor:end],
                        metadata["response_tokens"][cursor:end],
                        score_field,
                        width,
                    )
                )
                del logits
            cursor = end
            del output, input_ids
    if score_field == "student_support_topk":
        return {key: [value for chunk in chunks for value in chunk[key]] for key in chunks[0]}
    return [value for chunk in chunks for value in chunk]


def score_sequence_batch(
    model,
    metadata_rows,
    score_field,
    *,
    device,
    normalization_vocab_size=None,
    token_projection=None,
    chunk_size=None,
):
    """Batch short contexts; stream long ones using the same prepared rows."""
    import torch

    prepared = [_prepare_scoring_row(row, score_field, token_projection) for row in metadata_rows]
    max_length = max(len(context) for context, _, _ in prepared)
    if chunk_size is not None and max_length > chunk_size:
        return [
            score_sequence(
                model, row, score_field, device=device, chunk_size=chunk_size,
                normalization_vocab_size=normalization_vocab_size, _prepared=sequence,
            )
            for row, sequence in zip(metadata_rows, prepared)
        ]
    input_ids = torch.zeros((len(prepared), max_length), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, (context, _, _) in enumerate(prepared):
        input_ids[row_index, : len(context)] = torch.tensor(context, dtype=torch.long, device=device)
        attention_mask[row_index, : len(context)] = 1
    scores = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        for row_index, (_, targets, indices) in enumerate(prepared):
            metadata = metadata_rows[row_index]
            logits = output.logits[row_index, indices].float()[..., :normalization_vocab_size]
            width = len(metadata["candidate_ids"][0]) if score_field == "student_support_topk" else 0
            scores.append(_score_logits(logits, targets, metadata["response_tokens"], score_field, width))
    return scores


def main():
    args = parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if args.device is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    asset_lock = json.loads(args.asset_lock.read_text())
    validate_request(args, asset_lock)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    validate_original_identity(args.model, args.model_revision, asset_lock,
                               tokenizer_hash=canonical_hash(tokenizer.get_vocab()))
    revision_field = SCORE_FIELDS[args.score_field]
    score_role = revision_field.removesuffix("_revision")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    compatibility = asset_lock["token_id_compatibility"]
    # The original and weakened copies have the same output head. Normalize
    # across every model logit before gathering the fixed candidate IDs.
    vocab_size = resolve_model_vocab_size(config)
    if compatibility.get("normalization_vocab_size") != vocab_size:
        raise ValueError("Scoring metadata must specify the full original-model vocabulary")
    projection = None
    if compatibility.get("mode") == TOKEN_PROJECTION_MODE and score_role != "student":
        projection = ExactTokenStringProjection(
            AutoTokenizer.from_pretrained(asset_lock["models"]["student"]["path"], trust_remote_code=True),
            AutoTokenizer.from_pretrained(args.model, trust_remote_code=True),
        )
    normalize_legacy_vllm_yarn_config(config)
    model_kwargs = dict(config=config, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype))
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    precision = precision_identity(args, vocab_size)
    admit_output(args, asset_lock, precision)
    shards = parquet_paths(args.input)
    if not shards:
        raise FileNotFoundError(f"No input Parquet shards: {args.input}")
    assigned_shards = shards[args.rank :: args.world_size]
    if not assigned_shards:
        print(f"Rank {args.rank} has no assigned shards", flush=True)
        return
    for source in assigned_shards:
        if (args.output_dir / source.name).exists():
            raise FileExistsError(args.output_dir / source.name)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(args.device)
    model.eval()
    model.requires_grad_(False)
    from self_distillation.anchors._qwen3 import model_parameter_dtype
    if model_parameter_dtype(model) != getattr(torch, args.dtype):
        raise ValueError("Loaded scorer parameters do not match the requested forward dtype")
    weak_report = None
    if args.weak_anchor:
        from self_distillation.anchors import apply
        weak_report = apply(model, args.weak_anchor, module_scope=args.weak_anchor_module_scope)
        with (args.output_dir / f"weak_anchor.rank{args.rank}.json").open("x") as stream:
            json.dump(weak_report, stream, indent=2)

    def score_rows(rows, first_row_index):
        metadata_rows = [row["metadata"] for row in rows]
        for metadata in metadata_rows:
            validate_row(metadata, asset_lock, weak=bool(args.weak_anchor))
            validate_pair_precision(metadata, precision)
        scores = score_sequence_batch(
            model, metadata_rows, args.score_field, device=args.device,
            chunk_size=args.chunk_size, normalization_vocab_size=vocab_size, token_projection=projection,
        )
        for metadata, scores_for_row in zip(metadata_rows, scores):
            if args.score_field == "student_support_topk":
                rollout_id = metadata.get("rollout_sample_id", metadata["sample_id"])
                metadata["rollout_sample_id"] = rollout_id
                metadata["sample_id"] = canonical_hash(
                    {
                        "rollout_sample_id": rollout_id,
                        "student_revision": args.model_revision,
                    }
                )
                metadata.update(scores_for_row)
                metadata["behavior_sampled_log_probs"] = list(scores_for_row["student_ref_sampled_log_probs"])
                metadata["support_model_role"] = "student"
                metadata["support_model_revision"] = args.model_revision
            else:
                metadata[args.score_field] = scores_for_row
            if projection is not None:
                metadata["loss_mask"] = [
                    bool(original) and bool(valid)
                    for original, valid in zip(metadata["loss_mask"], metadata["token_projection_valid_mask"])
                ]
            metadata[revision_field] = args.model_revision
            role = {"post_teacher_log_probs": "post", "pre_teacher_log_probs": "pre"}.get(
                args.score_field, "student")
            metadata[role + "_score_precision_json"] = json.dumps(precision, sort_keys=True)
            if weak_report is not None:
                metadata["pre_weight_weak_variant"] = args.weak_anchor
                metadata["pre_weight_weak_spec_json"] = json.dumps(weak_report["specification"], sort_keys=True)
                metadata["pre_weight_weak_weights_sha256"] = weak_report["target_weights_sha256"]
            complete = all(metadata.get(field) is not None for field in REQUIRED_SCORED_FIELDS)
            metadata["is_offline_direct_opd"] = complete
            metadata["offline_direct_opd_stage"] = "fully_scored" if complete else f"scored_{args.score_field}"

    for source in assigned_shards:
        destination = args.output_dir / source.name
        count = write_scored_shard(
            source,
            destination,
            score_rows=score_rows,
            row_batch_size=args.row_batch_size,
        )
        print(f"Wrote {count:,} rows to {destination}", flush=True)


if __name__ == "__main__":
    main()
