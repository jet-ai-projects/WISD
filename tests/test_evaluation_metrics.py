import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.livecodebench_v6.lock_dataset import (
    load_locked_rows,
    release_files,
    sha256_file,
)
from evaluation.math_tasks.utils import identity_filter, process_results
from evaluation.summarize import summarize_code, summarize_math
from evaluation.prompts import code_messages, render_code_prompt


def test_code_reports_sample_average_not_any_correct():
    rows = [
        {"question_id": "a", "graded_list": [False, True, False, False], "response_tokens": [10, 20, 30, 40]},
        {"question_id": "b", "graded_list": [False] * 4, "response_tokens": [10] * 4},
    ]
    result = summarize_code(rows)
    assert result["accuracy"] == 0.125
    assert result["mean_response_tokens"] == 17.5
    assert result["samples"] == 8


def test_code_rejects_missing_or_invalid_grades():
    with pytest.raises(ValueError):
        summarize_code([{"question_id": "a", "graded_list": [True], "response_tokens": [10]}])
    with pytest.raises(ValueError):
        summarize_code([{"question_id": "a", "graded_list": ["False"] * 4, "response_tokens": [10] * 4}])


def test_math_grades_every_identity_filtered_response(monkeypatch):
    monkeypatch.setitem(sys.modules, "math_verify", SimpleNamespace(
        parse=lambda text: text.strip("$"), verify=lambda target, answer: target == answer,
    ))
    assert process_results({"Answer": 42}, [["wrong", "42", "wrong", "wrong"]]) == {"accuracy": 0.25}
    with pytest.raises(ValueError):
        process_results({"Answer": 42}, ["42"])


def test_identity_filter_keeps_all_repeats():
    responses = [["answer"] * 32, ["other"] * 32]
    assert identity_filter(responses, [{}, {}]) is responses


def test_math_summary_preserves_all_responses():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
    result = summarize_math([{
        "doc_id": 0, "accuracy": 0.5, "resps": [["a b", "c d e f"]],
    }], tokenizer, repeats=2)
    assert result["accuracy"] == 0.5
    assert result["mean_response_tokens"] == 3


def test_math_default_is_avg32_without_loading_tokenizer():
    result = summarize_math([{
        "doc_id": 0, "accuracy": 1 / 32, "resps": [["response"] * 32],
    }])
    assert result == {"accuracy": 1 / 32, "problems": 1, "samples": 32, "repeats": 32}


def test_code_prompt_preserves_starter_and_stdin_variants():
    starter = {"question_content": "Add two integers.", "starter_code": "def solve(a, b):"}
    messages = code_messages(starter)
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert "Question: Add two integers.\n\n" in messages[1]["content"]
    assert messages[1]["content"].endswith("```python\ndef solve(a, b):\n```\n\n")
    assert "stdin" not in messages[1]["content"]
    without = code_messages({"question_content": "Print the input."})[1]["content"]
    assert "Read the inputs from stdin solve the problem" in without
    assert without.endswith("```python\n# YOUR CODE HERE\n```\n\n")


def test_codeqwen_and_native_chat_wrappers_are_distinct():
    row = {"question_content": "Print 1."}
    raw = render_code_prompt(row, None)
    assert raw.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    assert raw.endswith("\n\n<|im_end|>\n<|im_start|>assistant\n")
    assert "<think>" not in raw
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: calls.append((args, kwargs)) or "native")
    assert render_code_prompt(row, tokenizer, "native", {"enable_thinking": True}) == "native"
    assert calls[0][1] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}


@pytest.mark.parametrize("family,expected", [
    ("qwen3", "wisd_aime25"), ("qwen36", "wisd_aime25"),
    ("qwen35", "wisd_aime25_step"), ("olmo3", "wisd_aime25_step"),
])
def test_math_entrypoint_selects_model_prompt_and_paper_context(tmp_path, family, expected):
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in {"TASKS", "MODEL_ARGS", "MAX_MODEL_LEN"}}
    env.update(PATH=str(tmp_path) + os.pathsep + env.get("PATH", ""), MODEL="model", MATH_PROMPT_FAMILY=family)
    result = subprocess.run(["bash", str(root / "evaluation/eval_math.sh")], env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert args[args.index("--tasks") + 1] == f"wisd_aime24,{expected},wisd_hmmt25"
    assert "max_model_len=49152" in args[args.index("--model_args") + 1]


def test_math_tasks_use_avg32_and_full_response_budget():
    import yaml

    class TaskLoader(yaml.SafeLoader):
        pass

    TaskLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    root = Path(__file__).resolve().parents[1] / "evaluation/math_tasks"
    tasks = {p.stem: yaml.load(p.read_text(), Loader=TaskLoader) for p in root.glob("*.yaml")}
    assert set(tasks) == {"wisd_aime24", "wisd_aime25", "wisd_aime25_step", "wisd_hmmt25"}
    for task in tasks.values():
        assert task["repeats"] == 32
        assert task["generation_kwargs"]["max_gen_toks"] == 40960
        assert task["generation_kwargs"]["top_k"] == -1
    assert tasks["wisd_aime25"]["doc_to_text"] == "Question: {{problem}}\nAnswer:"
    assert tasks["wisd_aime25_step"]["doc_to_text"] == tasks["wisd_hmmt25"]["doc_to_text"]
    assert "\\boxed{$Answer}" in tasks["wisd_hmmt25"]["doc_to_text"]


def test_dataset_lock_checks_checksum_and_question_ids(tmp_path):
    raw = tmp_path / "test6.jsonl"
    raw.write_text(json.dumps({"question_id": "a", "contest_date": "2025-02-01"}) + "\n")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "start_date": "2025-02-01", "question_ids": ["a"],
        "source_files": [{"path": "test6.jsonl", "sha256": sha256_file(raw)}],
    }))
    assert load_locked_rows(lock)[1][0]["question_id"] == "a"
    raw.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_locked_rows(lock)


def test_release_files():
    assert release_files("release_v5")[-1] == "test5.jsonl"
    assert release_files("release_v6")[-1] == "test6.jsonl"
    with pytest.raises(ValueError):
        release_files("release_latest")
