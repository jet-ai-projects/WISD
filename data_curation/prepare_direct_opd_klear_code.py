#!/usr/bin/env python3
"""Select length-filtered, decontaminated CodeSub prompts for Direct-OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_parquet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lcb-lock", required=True, type=Path, help="Benchmark cache paths and start date")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=3_200)
    parser.add_argument("--max-prompt-length", type=int, default=4_096)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--tokenizer-batch-size", type=int, default=16)
    parser.add_argument("--arrow-batch-size", type=int, default=1)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.80)
    parser.add_argument("--near-duplicate-overlap", type=float, default=0.90)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_tokens(text):
    text = re.sub(
        r"^\s*solve\s+the\s+following\s+coding\s+problem\s+using\s+the\s+programming" r"\s+language\s+python\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    for suffix in ("now solve the problem and return the code.", "now solve the problem and return the code"):
        if text.lower().rstrip().endswith(suffix):
            text = text[: len(text.rstrip()) - len(suffix)]
            break
    return tuple(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower()))


def shingles(tokens, size):
    if len(tokens) < size:
        return frozenset({tokens}) if tokens else frozenset()
    return frozenset(tokens[index : index + size] for index in range(len(tokens) - size + 1))


def load_lcb_tokens(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(config["start_date"])
    tokens = []
    for source in config["source_files"]:
        with Path(source["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if datetime.fromisoformat(row["contest_date"]) >= start:
                    tokens.append(
                        normalized_tokens(
                            f"{row.get('question_title', '')}\n{row['question_content']}\n"
                            f"{row.get('starter_code', '')}"
                        )
                    )
    return tokens


def contaminated_indices(rows, benchmark, *, shingle_size, jaccard_threshold, overlap_threshold):
    """Exclude exact matches or shingle overlap using an inverted benchmark index."""
    exact = set(benchmark)
    benchmark_shingles = [shingles(tokens, shingle_size) for tokens in benchmark]
    inverted = defaultdict(list)
    for index, items in enumerate(benchmark_shingles):
        for item in items:
            inverted[item].append(index)
    excluded = set()
    for index, row in enumerate(rows):
        tokens = row["normalized_tokens"]
        if tokens in exact:
            excluded.add(index)
            continue
        items = shingles(tokens, shingle_size)
        shared = Counter(match for item in items for match in inverted.get(item, ()))
        for match, intersection in shared.items():
            other = benchmark_shingles[match]
            jaccard = intersection / (len(items) + len(other) - intersection)
            overlap = intersection / min(len(items), len(other))
            if jaccard >= jaccard_threshold or overlap >= overlap_threshold:
                excluded.add(index)
                break
    return excluded


def prepare_rows(paths, tokenizer, *, tokenizer_batch_size, arrow_batch_size):
    rows, pending = [], []

    def flush():
        if not pending:
            return
        rendered = [
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            for row in pending
        ]
        lengths = tokenizer(rendered, add_special_tokens=False, padding=False, truncation=False, return_length=True)[
            "length"
        ]
        for row, length in zip(pending, lengths, strict=True):
            row["prompt_token_length"] = int(length)
        rows.extend(pending)
        pending.clear()

    source_id = 0
    for path in paths:
        batches = pq.ParquetFile(path).iter_batches(
            columns=["prompt", "reward_model"], batch_size=arrow_batch_size, use_threads=False
        )
        for batch in batches:
            for source in batch.to_pylist():
                content = source["prompt"][0]["content"]
                digest = hashlib.sha256(source["reward_model"]["ground_truth"].encode("utf-8")).hexdigest()
                pending.append(
                    {
                        "prompt": [{"content": content, "role": "user"}],
                        "label": f"sha256:{digest}",
                        "source_prompt_id": source_id,
                        "normalized_tokens": normalized_tokens(content),
                    }
                )
                source_id += 1
                if len(pending) >= tokenizer_batch_size:
                    flush()
    flush()
    return rows


def select_rows(rows, contaminated, *, max_prompt_length, num_prompts, seed):
    """Filter first, keep the first normalized duplicate, then shuffle in source order."""
    seen, eligible = set(), []
    for index, row in enumerate(rows):
        if row["prompt_token_length"] > max_prompt_length or index in contaminated:
            continue
        tokens = row["normalized_tokens"]
        if tokens not in seen:
            seen.add(tokens)
            eligible.append(row)
    random.Random(seed).shuffle(eligible)
    return [{key: row[key] for key in ("prompt", "label", "source_prompt_id")} for row in eligible[:num_prompts]]


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = prepare_rows(
        sorted(args.input_dir.glob("*.parquet")),
        tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
        arrow_batch_size=args.arrow_batch_size,
    )
    contaminated = contaminated_indices(
        rows,
        load_lcb_tokens(args.lcb_lock),
        shingle_size=args.shingle_size,
        jaccard_threshold=args.near_duplicate_jaccard,
        overlap_threshold=args.near_duplicate_overlap,
    )
    selected = select_rows(
        rows,
        contaminated,
        max_prompt_length=args.max_prompt_length,
        num_prompts=args.num_prompts,
        seed=args.selection_seed,
    )
    write_parquet(pa.Table.from_pylist(selected), args.output, compression="zstd")
    print(f"Wrote {len(selected):,} prompts to {args.output}")


if __name__ == "__main__":
    main()
