#!/usr/bin/env python3
"""Summarize sample-averaged accuracy (Avg@n), with optional response lengths."""

import argparse
import json
from pathlib import Path
from statistics import fmean


def summarize_code(rows, repeats=4):
    if not rows:
        raise ValueError("No evaluated questions")
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    scores, lengths = [], []
    for row in rows:
        grades, tokens = row["graded_list"], row["response_tokens"]
        if len(grades) != repeats or len(tokens) != repeats:
            raise ValueError(f"Expected {repeats} responses per question")
        if any(type(grade) not in (int, bool) or grade not in (0, 1) for grade in grades):
            raise ValueError("Grades must be booleans or 0/1 integers")
        if any(type(length) is not int or length < 0 for length in tokens):
            raise ValueError("Token counts must be nonnegative integers")
        scores.extend(int(grade) for grade in grades)
        lengths.extend(tokens)
    return {
        "accuracy": fmean(scores), "mean_response_tokens": fmean(lengths),
        "problems": len(rows), "samples": len(scores), "repeats": repeats,
    }


def summarize_math(rows, tokenizer=None, repeats=32):
    if not rows:
        raise ValueError("No math samples")
    ids = [row["doc_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate math document IDs")
    scores, lengths = [], []
    for row in rows:
        texts = row["resps"][0]
        score = row["accuracy"]
        if len(texts) != repeats or not 0 <= score <= 1:
            raise ValueError("Invalid repetition count or accuracy")
        scores.append(score)
        if tokenizer is not None:
            lengths.extend(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
    result = {
        "accuracy": fmean(scores), "problems": len(rows),
        "samples": len(rows) * repeats, "repeats": repeats,
    }
    if tokenizer is not None:
        result["mean_response_tokens"] = fmean(lengths)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-samples", type=Path, required=True)
    parser.add_argument("--tokenizer", help="Optionally report mean response length")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with args.math_samples.open() as handle:
        rows = [json.loads(line) for line in handle]
    result = summarize_math(rows, tokenizer, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
