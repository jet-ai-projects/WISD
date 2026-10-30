#!/usr/bin/env python3
"""Convert Skywork-OR1 math prompts to the DAPO training format."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_parquet

PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def convert_row(row):
    question = row["prompt"][0]["content"].strip()
    return {
        "data_source": "math_dapo",
        "prompt": [{"content": f"{PROMPT_PREFIX}{question}{PROMPT_SUFFIX}", "role": "user"}],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": json.loads(row["reward_model"]["ground_truth"])[0],
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": {
            "index": row["extra_info"].get("index"),
            "model_difficulty": row["extra_info"]["model_difficulty"],
            "original_data_source": row["data_source"],
            "prompt_style": "dapo_original",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    rows = [convert_row(row) for row in pq.read_table(args.input).to_pylist()]
    write_parquet(pa.Table.from_pylist(rows), args.output, compression="snappy")
    print(f"Wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
