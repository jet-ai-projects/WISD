"""Portable single-node launch configuration for cached WISD training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen3-4B": ("megatron", "qwen3-4B", None),
    "qwen3.5-4B": ("megatron", "qwen3.5-4B", None),
    "olmo3-7b-think": ("fsdp", None, None),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-type", choices=MODELS, default=os.getenv("MODEL_TYPE", "qwen3-4B"))
    result.add_argument("--student", default=os.getenv("STUDENT_MODEL"))
    result.add_argument("--data", default=os.getenv("DATA_DIR"))
    result.add_argument("--manifest", default=os.getenv("MANIFEST"))
    result.add_argument("--save", default=os.getenv("SAVE_DIR", "checkpoints/wisd"))
    result.add_argument("--load", default=os.getenv("LOAD_DIR"))
    result.add_argument("--num-gpus", type=int, default=int(os.getenv("NUM_GPUS", "8")))
    result.add_argument("--alpha", type=float, default=float(os.getenv("ALPHA", "2.0")))
    result.add_argument("--lr", type=float, default=float(os.getenv("LR", "1e-6")))
    result.add_argument("--rollout-batch-size", type=int, default=int(os.getenv("ROLLOUT_BATCH_SIZE", "256")))
    result.add_argument("--global-batch-size", type=int, default=int(os.getenv("GLOBAL_BATCH_SIZE", "64")))
    budget = result.add_mutually_exclusive_group()
    budget.add_argument("--num-rollout", type=int, help="Number of cached batches; defaults to 50 (200 optimizer updates).")
    budget.add_argument("--optimizer-steps", type=int, help="Specify an exact optimizer-update budget instead.")
    result.add_argument("--save-interval", type=int, default=int(os.getenv("SAVE_INTERVAL", "50")))
    result.add_argument("--max-tokens-per-gpu", type=int, default=int(os.getenv("MAX_TOKENS_PER_GPU", "4096")))
    result.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TP_SIZE", "1")))
    result.add_argument("--context-parallel-size", type=int, default=int(os.getenv("CP_SIZE", "1")))
    result.add_argument("--sequence-parallel", action="store_true")
    result.add_argument("--top-k", type=int, default=16)
    result.add_argument("--responses-per-prompt", type=int, default=int(os.getenv("RESPONSES_PER_PROMPT", "4")))
    default_prompt_length = "4096" if os.getenv("TASK", "math") == "code" else "1024"
    result.add_argument("--max-prompt-length", type=int,
                        default=int(os.getenv("MAX_PROMPT_LENGTH", default_prompt_length)))
    result.add_argument("--max-response-length", type=int, default=int(os.getenv("MAX_RESPONSE_LENGTH", "2048")))
    result.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "1.0")))
    result.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "1.0")))
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--rollout-seed", type=int, default=42)
    result.add_argument("--loss-mode", choices=["tilted_target", "policy_gradient"], default="tilted_target")
    result.add_argument("--ray-address", default=os.getenv("RAY_JOB_ADDRESS"),
                        help="Existing Ray dashboard address; otherwise start a local Ray head.")
    result.add_argument("--dashboard-port", type=int, default=8265)
    result.add_argument("--dry-run", action="store_true", help="Print commands without starting Ray or loading models.")
    return result


def training_budget(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve exact optimizer-update and cached-batch counts."""
    batch, global_batch = args.rollout_batch_size, args.global_batch_size
    if batch <= 0 or global_batch <= 0 or batch % global_batch:
        raise ValueError("rollout-batch-size must be a positive multiple of global-batch-size")
    updates_per_batch = batch // global_batch
    if args.num_rollout is not None:
        if args.num_rollout <= 0:
            raise ValueError("num-rollout must be positive")
        return args.num_rollout, args.num_rollout * updates_per_batch
    if args.optimizer_steps is None:
        return 50, 50 * updates_per_batch
    optimizer_steps = args.optimizer_steps
    if optimizer_steps <= 0 or optimizer_steps % updates_per_batch:
        raise ValueError(f"optimizer-steps must be a positive multiple of {updates_per_batch}")
    return optimizer_steps // updates_per_batch, optimizer_steps


def build_train_args(args: argparse.Namespace) -> list[str]:
    if not args.student or not args.data:
        raise ValueError("provide --student and --data, or STUDENT_MODEL and DATA_DIR")
    for name in ("alpha", "lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("num_gpus", "tensor_parallel_size", "context_parallel_size", "save_interval",
                 "max_tokens_per_gpu", "top_k", "responses_per_prompt"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    backend, _, _ = MODELS[args.model_type]
    parallel_size = args.tensor_parallel_size * args.context_parallel_size
    if args.num_gpus % parallel_size:
        raise ValueError("tensor-parallel-size * context-parallel-size must divide num-gpus")
    if args.global_batch_size % (args.num_gpus // parallel_size):
        raise ValueError("global-batch-size must be divisible by data-parallel size")
    if backend == "fsdp" and (parallel_size != 1 or args.sequence_parallel):
        raise ValueError("the OLMo FSDP recipe uses pure data parallelism (TP=CP=1)")
    if args.sequence_parallel and args.tensor_parallel_size == 1:
        raise ValueError("sequence-parallel requires tensor-parallel-size > 1")
    num_rollout, _ = training_budget(args)
    student = str(Path(args.student).expanduser().resolve())
    data = str(Path(args.data).expanduser().resolve())
    save = str(Path(args.save).expanduser().resolve())
    manifest = str(Path(args.manifest).expanduser().resolve()) if args.manifest else str(Path(data) / "manifest.json")
    options = [
        "--hf-checkpoint", student,
        "--save", save, "--save-interval", str(args.save_interval),
        "--prompt-data", data, "--input-key", "prompt", "--label-key", "label",
        "--num-rollout", str(num_rollout),
        "--rollout-batch-size", str(args.rollout_batch_size),
        "--global-batch-size", str(args.global_batch_size),
        "--n-samples-per-prompt", "1",
        "--rollout-max-prompt-len", str(args.max_prompt_length),
        "--rollout-max-response-len", str(args.max_response_length),
        "--rollout-temperature", str(args.temperature), "--rollout-top-p", str(args.top_p),
        "--rollout-seed", str(args.rollout_seed),
        "--custom-rm-path", "slime.rollout.on_policy_distillation.reward_func",
        "--custom-reward-post-process-path", "slime.rollout.on_policy_distillation.post_process_rewards",
        "--offline-direct-opd", "--offline-direct-opd-manifest", manifest,
        "--offline-direct-opd-responses-per-prompt", str(args.responses_per_prompt),
        "--direct-opd-top-k", str(args.top_k),
        "--disable-compute-advantages-and-returns", "--loss-type", "custom_loss",
        "--custom-loss-function-path", "slime.rollout.offline_direct_opd.megatron_loss",
        "--calculate-per-token-loss", "--offline-direct-opd-loss-mode", args.loss_mode,
        "--offline-direct-opd-kl-coef", str(args.alpha), "--entropy-coef", "0.0",
        "--optimizer", "adam", "--lr", str(args.lr), "--lr-decay-style", "constant",
        "--weight-decay", "0.01", "--adam-beta1", "0.9", "--adam-beta2", "0.999", "--clip-grad", "1.0",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(args.max_tokens_per_gpu),
        "--context-parallel-size", str(args.context_parallel_size),
        "--seed", str(args.seed), "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(args.num_gpus),
        "--rollout-num-gpus", "0", "--rollout-num-gpus-per-engine", "1",
    ]
    if args.load:
        options.extend(["--load", str(Path(args.load).expanduser().resolve())])
    elif backend == "megatron":
        raise ValueError("Megatron requires --load / LOAD_DIR pointing to a converted checkpoint")
    if backend == "fsdp":
        options.extend(["--train-backend", "fsdp", "--attn-implementation", "flash_attention_2",
                        "--gradient-checkpointing", "--adam-eps", "1e-8"])
    else:
        sequence_length = max(4096, args.max_prompt_length + args.max_response_length)
        options.extend([
            "--seq-length", str(sequence_length), "--max-position-embeddings", str(sequence_length),
            "--tensor-model-parallel-size", str(args.tensor_parallel_size),
            "--pipeline-model-parallel-size", "1", "--expert-model-parallel-size", "1",
            "--expert-tensor-parallel-size", "1", "--recompute-granularity", "full",
            "--recompute-method", "uniform", "--recompute-num-layers", "1",
            "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
            "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
            "--attention-backend", "flash",
        ])
        if args.sequence_parallel:
            options.append("--sequence-parallel")
    return options


def model_args(model_type: str) -> list[str]:
    """Load the same architecture flags used for checkpoint conversion."""
    _, model_file, rotary_base = MODELS[model_type]
    if model_file is None:
        return []
    environment = os.environ.copy()
    if rotary_base is not None:
        environment["MODEL_ARGS_ROTARY_BASE"] = rotary_base
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
         "model-config", str(REPO_ROOT / "configs" / "models" / f"{model_file}.sh")],
        check=True, capture_output=True, env=environment,
    )
    return [value.decode() for value in result.stdout.split(b"\0") if value]


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        train_args = build_train_args(args)
        architecture_args = model_args(args.model_type)
    except ValueError as error:
        argument_parser.error(str(error))
    num_rollout, updates = training_budget(args)
    print(f"Budget: {num_rollout} cached batches, {updates} optimizer updates, "
          f"{num_rollout * args.rollout_batch_size} trajectory rows.", flush=True)
    environment = os.environ.copy()
    python_paths = [str(REPO_ROOT)]
    if environment.get("MEGATRON_PATH"):
        python_paths.append(environment["MEGATRON_PATH"])
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    worker_environment = {"PYTHONPATH": environment["PYTHONPATH"]}
    if MODELS[args.model_type][0] == "megatron":
        worker_environment["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    runtime = json.dumps({"env_vars": worker_environment})
    address = args.ray_address or f"http://127.0.0.1:{args.dashboard_port}"
    commands = []
    if not args.ray_address:
        commands.append(["ray", "start", "--head", "--node-ip-address", "127.0.0.1",
                         "--num-gpus", str(args.num_gpus), "--disable-usage-stats",
                         "--dashboard-host", "127.0.0.1", "--dashboard-port", str(args.dashboard_port)])
    commands.append(["ray", "job", "submit", f"--address={address}", f"--runtime-env-json={runtime}",
                     "--", sys.executable, str(REPO_ROOT / "train.py"), *architecture_args, *train_args])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)


if __name__ == "__main__":
    main()
