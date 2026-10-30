#!/usr/bin/env python3
"""Generate four code responses per problem; never execute generated code."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from evaluation.livecodebench_v6.lock_dataset import load_locked_rows, sha256_file
from evaluation.prompts import render_code_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat-format", choices=("codeqwen", "native"), default="codeqwen")
    parser.add_argument("--chat-template-kwargs", default="{}", help="Native chat template options as JSON")
    parser.add_argument("--llm-kwargs", default="{}", help="Additional vLLM engine arguments as JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_new_tokens < 1 or args.max_model_len <= args.max_new_tokens or args.repeats < 1:
        parser.error("Use a positive sample count and leave context space for the prompt")
    lock, rows = load_locked_rows(args.dataset_lock)

    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    from vllm import LLM, SamplingParams

    style = LMStyle.CodeQwenInstruct
    engine_kwargs = dict(
        model=args.model, tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, dtype="bfloat16",
        gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=args.seed,
    )
    extra = json.loads(args.llm_kwargs)
    if set(extra) & set(engine_kwargs):
        parser.error("--llm-kwargs must not override named CLI options or fixed engine defaults")
    engine_kwargs.update(extra)
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    template_kwargs = json.loads(args.chat_template_kwargs)
    prompts = [render_code_prompt(row, tokenizer, args.chat_format, template_kwargs) for row in rows]
    prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if max(prompt_lengths) + args.max_new_tokens > args.max_model_len:
        raise ValueError("Context window must accommodate the longest prompt and full response budget")
    outputs = llm.generate(prompts, SamplingParams(
        n=args.repeats, temperature=0.6, top_p=0.95, top_k=-1,
        max_tokens=args.max_new_tokens, seed=args.seed, stop=[],
    ))
    generations = []
    for row, output in zip(rows, outputs, strict=True):
        texts = [sample.text for sample in output.outputs]
        if len(texts) != args.repeats:
            raise ValueError("Unexpected number of responses per problem")
        generations.append({
            "question_id": row["question_id"],
            "output_list": texts,
            "code_list": [extract_code(text, style) for text in texts],
            "response_tokens": [
                len(sample.token_ids) for sample in output.outputs
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generations, ensure_ascii=False) + "\n")
    args.output.with_suffix(".protocol.json").write_text(json.dumps({
        "model": args.model, "release": lock["release_version"],
        "problem_count": len(rows), "samples_per_problem": args.repeats,
        "temperature": 0.6, "top_p": 0.95, "top_k": -1, "seed": args.seed,
        "max_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
        "prompt_style": "CodeQwenInstruct", "chat_format": args.chat_format,
        "chat_template_kwargs": template_kwargs, "llm_kwargs": engine_kwargs,
        "dataset_lock_sha256": sha256_file(args.dataset_lock),
        "vllm_version": version("vllm"), "transformers_version": version("transformers"),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
