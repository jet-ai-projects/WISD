"""CPU-only coverage of the public data-pipeline entry points."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline_env(tmp_path):
    calls_path = tmp_path / "calls.jsonl"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['PIPELINE_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args and args[0].endswith(('lock_dataset.py', 'prepare_direct_opd_assets.py')):\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text('{}')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    run_dir = tmp_path / "run with spaces"
    run_dir.mkdir()
    source_dataset = run_dir / "prompts.parquet"
    source_dataset.write_bytes(b"mock parquet; never loaded")
    (run_dir / "rollouts").mkdir()
    environment = os.environ.copy()
    for name in (
        "TASK",
        "RUN_DIR",
        "SOURCE_INPUT",
        "SOURCE_DATASET",
        "ROLLOUT_DIR",
        "STUDENT_MODEL",
        "STUDENT_REVISION",
        "WEAK_ANCHOR",
        "ANCHOR_DIR",
        "ASSET_LOCK",
        "LCB_LOCK",
        "LCB_RAW_DIR",
        "DATA_DIR",
        "NUM_PROMPTS",
        "RESPONSES_PER_PROMPT",
        "MAX_PROMPT_LENGTH",
        "MAX_RESPONSE_LENGTH",
        "RANK",
        "WORLD_SIZE",
        "DTYPE",
        "ENABLE_THINKING",
        "LANGUAGE_MODEL_ONLY",
        "TEMPERATURE",
        "TOP_P",
        "SEED",
        "COLLECT_BATCH_SIZE",
        "TP_SIZE",
        "GPU_MEMORY_UTILIZATION",
        "MAX_NUM_SEQS",
        "SHARD_SIZE",
        "SCORE_DEVICE",
        "SCORE_BATCH_SIZE",
        "SCORE_CHUNK_SIZE",
    ):
        environment.pop(name, None)
    environment.update(
        PYTHON=str(fake_python),
        PIPELINE_CALLS=str(calls_path),
        TASK="math",
        RUN_DIR=str(run_dir),
        NUM_PROMPTS="3200",
    )
    for index, name in enumerate(
        ("STUDENT_MODEL",),
        start=1,
    ):
        model = tmp_path / "models" / f"{index:040x}"
        model.mkdir(parents=True)
        environment[name] = str(model)
    return environment, calls_path


def run_script(name, environment, *arguments, check=True):
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / name), *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


def read_calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def assert_supported_flags(calls):
    """Every generated flag must exist in the retained core CLI."""
    for call in calls:
        source = ROOT / call[0]
        tree = ast.parse(source.read_text())
        known_flags = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        }
        # argparse.BooleanOptionalAction creates --no-... implicitly.
        known_flags |= {
            "--no-" + flag[2:] for flag in known_flags if flag in {"--enable-thinking", "--language-model-only"}
        }
        unknown_flags = {value for value in call[1:] if value.startswith("--") and value not in known_flags}
        assert not unknown_flags, (source, unknown_flags)


@pytest.mark.parametrize(
    "name",
    ["prepare_data.sh", "collect_rollouts.sh", "score_anchors.sh"],
)
def test_help_requires_no_models_or_gpu(name):
    completed = run_script(name, os.environ.copy(), "--help")
    assert completed.stdout.strip()


def test_math_prepare_flags(pipeline_env):
    environment, calls_path = pipeline_env
    source = Path(environment["RUN_DIR"]) / "source.parquet"
    source.write_bytes(b"mock")
    environment["SOURCE_INPUT"] = str(source)
    run_script("prepare_data.sh", environment)
    calls = read_calls(calls_path)
    assert calls[0][0] == "data_curation/prepare_direct_opd_skywork_math.py"
    assert option(calls[0], "--input") == str(source)
    assert option(calls[0], "--output").endswith("/prompts.parquet")
    assert_supported_flags(calls)


def test_code_prepare_downloads_and_uses_decontamination_lock(pipeline_env):
    environment, calls_path = pipeline_env
    environment.update(TASK="code", SOURCE_INPUT="/mock/code-shards", LCB_RAW_DIR="/mock/lcb-raw")
    run_script("prepare_data.sh", environment)
    calls = read_calls(calls_path)
    assert calls[0][0] == "evaluation/livecodebench_v6/lock_dataset.py"
    assert option(calls[0], "--start-date") == "2025-02-01"
    assert option(calls[0], "--expected-problems") == "131"
    assert option(calls[0], "--raw-dir") == "/mock/lcb-raw"
    assert calls[1][0] == "data_curation/prepare_direct_opd_klear_code.py"
    assert option(calls[1], "--lcb-lock") == option(calls[0], "--output")
    assert option(calls[1], "--max-prompt-length") == "4096"
    assert_supported_flags(calls)


def test_code_prepare_reuses_supplied_decontamination_lock(pipeline_env):
    environment, calls_path = pipeline_env
    environment.update(TASK="code", SOURCE_INPUT="/mock/code-shards", LCB_LOCK="/mock/lcb-lock.json")
    run_script("prepare_data.sh", environment)
    calls = read_calls(calls_path)
    assert len(calls) == 1
    assert calls[0][0] == "data_curation/prepare_direct_opd_klear_code.py"
    assert option(calls[0], "--lcb-lock") == environment["LCB_LOCK"]
    assert_supported_flags(calls)


@pytest.mark.parametrize("task,prompt_limit,label", [("math", "1024", "reward_model"), ("code", "4096", "label")])
def test_collection_uses_one_student_snapshot(pipeline_env, task, prompt_limit, label):
    environment, calls_path = pipeline_env
    environment.update(TASK=task, RANK="1", WORLD_SIZE="2")
    run_script("collect_rollouts.sh", environment)
    calls = read_calls(calls_path)
    assert len(calls) == 2
    assets, collect = calls
    assert assets[0] == "data_curation/prepare_direct_opd_assets.py"
    assert option(assets, "--student") == environment["STUDENT_MODEL"]
    assert collect[0] == "data_curation/collect_direct_opd_rollouts.py"
    assert option(collect, "--model") == environment["STUDENT_MODEL"]
    assert option(collect, "--model-revision") == "0" * 39 + "1"
    assert option(collect, "--max-response-length") == "2048"
    assert option(collect, "--max-prompt-length") == prompt_limit
    assert option(collect, "--label-key") == label
    assert option(collect, "--responses-per-prompt") == "4"
    assert option(collect, "--rank") == "1"
    assert option(collect, "--world-size") == "2"
    assert_supported_flags(calls)


@pytest.mark.parametrize("variant", ["rtn3_g128", "group_prune_g128_tau1of6_rownorm"])
def test_clean_and_weak_scores_share_checkpoint_and_candidates(pipeline_env, variant):
    environment, calls_path = pipeline_env
    environment["WEAK_ANCHOR"] = variant
    run_script("score_anchors.sh", environment)
    assets, post, pre, reference, manifest = read_calls(calls_path)
    for role in ("student", "post-teacher", "pre-teacher"):
        assert option(assets, "--" + role) == environment["STUDENT_MODEL"]
    assert option(post, "--input") == str(Path(environment["RUN_DIR"]) / "rollouts")
    assert option(pre, "--input") == option(post, "--output-dir")
    assert option(reference, "--input") == option(pre, "--output-dir")
    assert option(pre, "--weak-anchor") == variant
    assert "--weak-anchor" not in post
    assert "--weak-anchor" not in reference
    assert [option(call, "--score-field") for call in (post, pre, reference)] == [
        "post_teacher_log_probs", "pre_teacher_log_probs", "student_ref_sampled_log_probs"
    ]
    for call in (post, pre, reference):
        assert option(call, "--model") == environment["STUDENT_MODEL"]
        assert option(call, "--dtype") == "bfloat16"
        assert option(call, "--attn-implementation") == "sdpa"
        assert option(call, "--asset-lock") == option(assets, "--output")
    assert option(manifest, "--input") == option(reference, "--output-dir")
    assert option(manifest, "--manifest-out") == f"{option(reference, '--output-dir')}/manifest.json"
    assert option(manifest, "--source-dataset") == str(Path(environment["RUN_DIR"]) / "prompts.parquet")
    assert option(reference, "--output-dir").endswith(f"/anchors/{variant}/final")
    assert_supported_flags([assets, post, pre, reference, manifest])


def test_scoring_rejects_float32_forward(pipeline_env):
    environment, calls_path = pipeline_env
    environment["DTYPE"] = "float32"
    completed = run_script("score_anchors.sh", environment, check=False)
    assert completed.returncode != 0
    assert "bfloat16 or float16" in completed.stderr
    assert not calls_path.exists()


def test_scoring_rejects_unknown_weak_construction(pipeline_env):
    environment, calls_path = pipeline_env
    environment["WEAK_ANCHOR"] = "unrecognized"
    completed = run_script("score_anchors.sh", environment, check=False)
    assert completed.returncode != 0
    assert "Unknown WEAK_ANCHOR" in completed.stderr
    assert not calls_path.exists()


@pytest.mark.parametrize("revision", [None, "explicit-model-revision"])
def test_named_model_directory_defaults_revision_to_basename(pipeline_env, revision):
    environment, calls_path = pipeline_env
    named_model = Path(environment["RUN_DIR"]) / "Qwen3-4B"
    named_model.mkdir()
    environment["STUDENT_MODEL"] = str(named_model)
    if revision is not None:
        environment["STUDENT_REVISION"] = revision
    run_script("collect_rollouts.sh", environment)
    assets, collect = read_calls(calls_path)
    expected_revision = revision or named_model.name
    assert option(assets, "--student-revision") == expected_revision
    assert option(collect, "--model-revision") == expected_revision
    assert option(collect, "--model") == str(named_model)
    assert_supported_flags([assets, collect])


@pytest.mark.parametrize("script,call_count", [("collect_rollouts.sh", 1), ("score_anchors.sh", 4)])
def test_existing_asset_lock_is_reused(pipeline_env, script, call_count):
    environment, calls_path = pipeline_env
    lock = Path(environment["RUN_DIR"]) / "assets.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({"models": {"student": {"path": "/different/model", "revision": "old-revision"}}})
    lock.write_text(content)
    run_script(script, environment)
    calls = read_calls(calls_path)
    assert len(calls) == call_count
    assert all(call[0] != "data_curation/prepare_direct_opd_assets.py" for call in calls)
    assert all(option(call, "--asset-lock") == str(lock) for call in calls)
    assert lock.read_text() == content
    assert_supported_flags(calls)
