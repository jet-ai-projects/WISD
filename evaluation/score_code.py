#!/usr/bin/env python3
"""Run the upstream LCB checker INSIDE an isolated, disposable container."""

import argparse
import json
from pathlib import Path

from evaluation.livecodebench_v6.lock_dataset import load_locked_rows
from evaluation.summarize import summarize_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--acknowledge-isolated-execution", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_isolated_execution:
        parser.error("Run inside a network-disabled disposable container; see docs/evaluation.md")
    if args.output.exists():
        raise FileExistsError(args.output)
    lock, rows = load_locked_rows(args.dataset_lock)
    generated = json.loads(args.generation.read_text())
    if [row["question_id"] for row in generated] != lock["question_ids"]:
        raise ValueError("Generation questions do not match the dataset lock")
    if args.repeats < 1 or any(len(row["code_list"]) != args.repeats for row in generated):
        raise ValueError("Unexpected number of code candidates per problem")

    # Both dataset test deserialization and model code execution occur here,
    # never in the GPU generation process.
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results

    problems = [CodeGenerationProblem(**row) for row in rows]
    metrics = codegen_metrics(
        [problem.get_evaluation_sample() for problem in problems],
        [row["code_list"] for row in generated],
        num_process_evaluate=args.workers,
        timeout=6,
    )
    grades = extract_instance_results(metrics[1])
    evaluated = [
        {**row, "graded_list": result}
        for row, result in zip(generated, grades, strict=True)
    ]
    summary = summarize_code(evaluated, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "summary": summary, "rows": evaluated,
        "release": lock["release_version"], "timeout_seconds": 6,
    }, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
