"""CPU-only tests for the release launcher, without loading training backends."""

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wisd_train", ROOT / "configs/wisd/train.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments(*extra):
    return MODULE.parser().parse_args([
        "--student", "/tmp/student", "--data", "/tmp/cache",
        "--save", "/tmp/run", "--load", "/tmp/initial", *extra,
    ])


def test_default_budget_is_fifty_cached_batches_and_two_hundred_updates():
    assert MODULE.training_budget(arguments()) == (50, 200)


def test_exact_optimizer_budget_is_not_rollout_count():
    assert MODULE.training_budget(arguments("--optimizer-steps", "100")) == (25, 100)


def test_explicit_rollout_budget_is_not_optimizer_step_count():
    assert MODULE.training_budget(arguments("--num-rollout", "100")) == (100, 400)


def test_non_integral_optimizer_budget_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        MODULE.training_budget(arguments("--optimizer-steps", "101"))


def test_wisd_target_flags_and_no_live_rollout_gpus():
    result = MODULE.build_train_args(arguments())
    assert result[result.index("--offline-direct-opd-loss-mode") + 1] == "tilted_target"
    assert result[result.index("--offline-direct-opd-kl-coef") + 1] == "2.0"
    assert result[result.index("--rollout-num-gpus") + 1] == "0"
    assert result[result.index("--num-rollout") + 1] == "50"
    assert result[result.index("--custom-loss-function-path") + 1] == "slime.rollout.offline_direct_opd.megatron_loss"


def test_naive_ablation_uses_the_existing_runtime_loss():
    result = MODULE.build_train_args(arguments("--loss-mode", "policy_gradient"))
    assert result[result.index("--offline-direct-opd-loss-mode") + 1] == "policy_gradient"


def test_fsdp_uses_native_model_and_no_megatron_flags():
    args = arguments("--model-type", "olmo3-7b-think")
    args.load = None
    result = MODULE.build_train_args(args)
    assert result[result.index("--train-backend") + 1] == "fsdp"
    assert "--load" not in result
    assert "--tensor-model-parallel-size" not in result
    assert "--gradient-checkpointing" in result


def test_megatron_initialization_requires_a_checkpoint():
    args = arguments()
    args.load = None
    with pytest.raises(ValueError, match="requires --load"):
        MODULE.build_train_args(args)


def test_fsdp_rejects_model_parallel_configuration():
    with pytest.raises(ValueError, match="pure data parallelism"):
        MODULE.build_train_args(arguments("--model-type", "olmo3-7b-think", "--context-parallel-size", "2"))


@pytest.mark.parametrize("alpha", ["0", "-1", "nan", "inf"])
def test_alpha_must_be_finite_positive(alpha):
    with pytest.raises(ValueError, match="positive and finite"):
        MODULE.build_train_args(arguments("--alpha", alpha))


def test_qwen35_keeps_its_native_architecture():
    assert MODULE.MODELS["qwen3.5-4B"] == ("megatron", "qwen3.5-4B", None)


def test_code_generation_settings_match_pipeline(monkeypatch):
    monkeypatch.setenv("TASK", "code")
    monkeypatch.setenv("MAX_RESPONSE_LENGTH", "2048")
    args = arguments()
    assert args.max_prompt_length == 4096
    assert args.max_response_length == 2048
    options = MODULE.build_train_args(args)
    assert options[options.index("--seq-length") + 1] == "6144"
    assert options[options.index("--max-position-embeddings") + 1] == "6144"
    assert options[options.index("--rollout-max-response-len") + 1] == "2048"


def test_sequence_parallel_requires_actual_tensor_parallelism():
    with pytest.raises(ValueError, match="requires tensor"):
        MODULE.build_train_args(arguments("--sequence-parallel"))


def test_dry_run_has_no_workspace_upload_and_binds_local_dashboard():
    process = subprocess.run([
        sys.executable, str(ROOT / "configs/wisd/train.py"),
        "--student", "/tmp/student with spaces", "--data", "/tmp/cache with spaces",
        "--load", "/tmp/initial checkpoint", "--model-type", "qwen3-4B",
        "--dry-run",
    ], check=True, capture_output=True, text=True)
    lines = process.stdout.splitlines()
    ray_start = shlex.split(next(line for line in lines if line.startswith("ray start")))
    assert ray_start[ray_start.index("--dashboard-host") + 1] == "127.0.0.1"
    ray_submit = shlex.split(next(line for line in lines if line.startswith("ray job submit")))
    runtime_arg = next(arg for arg in ray_submit if arg.startswith("--runtime-env-json="))
    runtime = json.loads(runtime_arg.split("=", 1)[1])
    assert set(runtime) == {"env_vars"}
    assert "--working-dir" not in ray_submit
    assert "/tmp/student with spaces" in ray_submit
