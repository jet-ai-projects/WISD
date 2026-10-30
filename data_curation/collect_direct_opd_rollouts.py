#!/usr/bin/env python3
"""Collect a fixed student rollout set with top-K and sampled-token log probabilities."""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import canonical_hash, read_manifest, write_parquet
from self_distillation.scoring import validate_original_identity


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--asset-lock", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--label-key", default="reward_model")
    parser.add_argument("--prompt-id-key")
    parser.add_argument("--max-prompts", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--responses-per-prompt", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--compilation-mode", type=int, default=0)
    parser.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_prompts(tokenizer, args):
    rows = (
        pq.read_table(args.input).to_pylist()
        if args.input.suffix == ".parquet"
        else [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    )
    prepared = []
    for index, row in enumerate(rows):
        prompt = row[args.prompt_key]
        if not isinstance(prompt, str):
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= args.max_prompt_length:
            label = row.get(args.label_key, "")
            if isinstance(label, dict):
                label = label["ground_truth"]
            prompt_id = row[args.prompt_id_key] if args.prompt_id_key is not None else index
            prepared.append((prompt_id, prompt, tokens, str(label)))
        if len(prepared) == args.max_prompts:
            break
    return prepared


def parse_token_logprobs(entry, chosen_token_id, top_k):
    scores = {int(token): float(getattr(value, "logprob", value)) for token, value in entry.items()}
    candidates = sorted(scores, key=lambda token: (-scores[token], token))[:top_k]
    return candidates, [scores[token] for token in candidates], scores[chosen_token_id]


def response_metadata(completion, prompt_id, response_id, prompt_tokens, common):
    tokens = [int(token) for token in completion.token_ids]
    candidates, scores, sampled = zip(
        *(
            parse_token_logprobs(entry, token, common["generation_config"]["top_k"])
            for token, entry in zip(tokens, completion.logprobs)
        )
    )
    return {
        **common,
        "sample_id": canonical_hash(
            {
                "student_revision": common["student_revision"],
                "prompt_id": prompt_id,
                "response_index": response_id,
                "generation_config_hash": common["generation_config_hash"],
            }
        ),
        "prompt_id": prompt_id,
        "group_id": prompt_id,
        "response_id": response_id,
        "prompt_tokens": prompt_tokens,
        "response_tokens": tokens,
        "loss_mask": [1] * len(tokens),
        "response": completion.text,
        "response_length": len(tokens),
        "candidate_ids": list(candidates),
        "behavior_topk_log_probs": list(scores),
        "behavior_sampled_log_probs": list(sampled),
        "finish_reason": str(completion.finish_reason),
    }


def main():
    args = parse_args()
    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("Require 0 <= rank < world_size")
    assets = read_manifest(args.asset_lock)
    validate_original_identity(args.model, args.model_revision, assets)
    # A new run owns its rank's shards; never silently replace an existing run.
    existing = sorted(args.output_dir.glob(f"rollouts-r{args.rank:05d}-*.parquet"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{args.output_dir}: use a new output directory or --overwrite")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    validate_original_identity(args.model, args.model_revision, assets,
                               tokenizer_hash=canonical_hash(tokenizer.get_vocab()))
    prepared = prepare_prompts(tokenizer, args)
    if len(prepared) != args.max_prompts:
        raise ValueError(f"Requested {args.max_prompts} prompts, but {len(prepared)} remain after length filtering")
    # Contiguous partitions keep source order when shards are sorted.
    per_rank = (len(prepared) + args.world_size - 1) // args.world_size
    prepared = prepared[args.rank * per_rank : (args.rank + 1) * per_rank]
    if not prepared:
        return
    for path in existing:
        path.unlink()
    import vllm
    from vllm import LLM, SamplingParams
    config = {
        name: getattr(args, name)
        for name in (
            "temperature",
            "top_p",
            "max_response_length",
            "max_prompt_length",
            "responses_per_prompt",
            "top_k",
            "seed",
            "enable_thinking",
            "dtype",
            "tensor_parallel_size",
            "compilation_mode",
            "cudagraph_mode",
            "max_num_seqs",
            "language_model_only",
            "gdn_prefill_backend",
        )
    }
    config.update(parallel_world_size=args.world_size, vllm_version=vllm.__version__)
    common = {
        "is_offline_direct_opd": False,
        "offline_direct_opd_stage": "rollout_collected",
        "schema_version": "offline_direct_opd_v1",
        "student_ref_sampled_log_probs": None,
        "post_teacher_log_probs": None,
        "pre_teacher_log_probs": None,
        "generation_seed": args.seed,
        "generation_config_hash": canonical_hash(config),
        "generation_config": config,
        "student_revision": args.model_revision,
        "student_model_path": str(Path(args.model).expanduser().resolve()),
        "post_teacher_revision": None,
        "pre_teacher_revision": None,
        "tokenizer_hash": assets["tokenizer_hash"],
    }
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_response_length,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_prompt_length + args.max_response_length,
        max_num_seqs=args.max_num_seqs,
        language_model_only=args.language_model_only,
        gdn_prefill_backend=args.gdn_prefill_backend,
        compilation_config={"mode": args.compilation_mode, "cudagraph_mode": args.cudagraph_mode},
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.responses_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
        logprobs=args.top_k,
        seed=args.seed,
    )
    buffer = []
    shard_index = 0

    def flush():
        nonlocal shard_index
        path = args.output_dir / f"rollouts-r{args.rank:05d}-{shard_index:05d}.parquet"
        write_parquet(pa.Table.from_pylist(buffer), path, compression="zstd")
        print(f"Wrote {path.name}: {len(buffer)} rows", flush=True)
        buffer.clear()
        shard_index += 1

    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start : start + args.batch_size]
        outputs = llm.generate([{"prompt_token_ids": row[2]} for row in batch], sampling, use_tqdm=False)
        for output, (prompt_id, prompt, tokens, label) in zip(outputs, batch):
            for response_id, completion in enumerate(output.outputs):
                buffer.append(
                    {
                        "prompt": prompt,
                        "label": label,
                        "metadata": response_metadata(completion, prompt_id, response_id, tokens, common),
                    }
                )
            # Keep each prompt's responses together, including in the final shard.
            if len(buffer) >= args.shard_size:
                flush()
    if buffer:
        flush()


if __name__ == "__main__":
    main()
