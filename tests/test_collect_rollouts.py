import hashlib
import json
import sys
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from data_curation import collect_direct_opd_rollouts as collect


def reference_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def completion():
    return SimpleNamespace(
        token_ids=[9, "3"],
        logprobs=[
            {9: SimpleNamespace(logprob=-2.0), 4: -0.5, 2: SimpleNamespace(logprob=-0.5)},
            {8: -1.5, "3": -0.2, 1: -0.9},
        ],
        text="generated response",
        finish_reason=None,
    )


def test_response_metadata_preserves_sample_id_support_and_chosen_scores():
    config = {"top_k": 2, "temperature": 1.0, "seed": 42}
    common = {
        "generation_config": config,
        "generation_config_hash": reference_hash(config),
        "student_revision": "student-版本",
    }
    metadata = collect.response_metadata(completion(), 17, 2, [10, 11], common)
    assert metadata == {
        **common,
        "sample_id": reference_hash(
            {
                "student_revision": "student-版本",
                "prompt_id": 17,
                "response_index": 2,
                "generation_config_hash": common["generation_config_hash"],
            }
        ),
        "prompt_id": 17,
        "group_id": 17,
        "response_id": 2,
        "prompt_tokens": [10, 11],
        "response_tokens": [9, 3],
        "loss_mask": [1, 1],
        "response": "generated response",
        "response_length": 2,
        "candidate_ids": [[2, 4], [3, 1]],
        "behavior_topk_log_probs": [[-0.5, -0.5], [-0.2, -0.9]],
        "behavior_sampled_log_probs": [-2.0, -0.2],
        "finish_reason": "None",
    }
    assert collect.response_metadata(completion(), 17, 3, [10, 11], common)["sample_id"] != metadata["sample_id"]


class Tokenizer:
    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, **kwargs):
        self.rendered.append((messages, kwargs))
        return "chat:" + messages[-1]["content"]

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(prompt)))

    def get_vocab(self):
        return {f'token-{index}': index for index in range(32)}


def asset_lock(tmp_path, *, model='student', revision='unknown', tokenizer_hash=None):
    tokenizer_hash = tokenizer_hash or collect.canonical_hash(Tokenizer().get_vocab())
    path = tmp_path / 'assets.json'
    path.write_text(json.dumps({'tokenizer_hash': tokenizer_hash, 'models': {
        role: {'path': model, 'revision': revision, 'tokenizer_hash': tokenizer_hash}
        for role in ('student', 'post_teacher', 'pre_teacher')}}))
    return path


def prompt_args(tmp_path, rows, **overrides):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return SimpleNamespace(
        **{
            "input": path,
            "prompt_key": "prompt",
            "label_key": "label",
            "enable_thinking": True,
            "max_prompt_length": 10,
            "prompt_id_key": None,
            "max_prompts": 2,
            **overrides,
        }
    )


def test_prepare_prompts_preserves_source_order_and_ids_after_length_filter(tmp_path):
    rows = [
        {"prompt": "too long for the prompt budget"},
        {"prompt": "ten chars!", "label": "first"},
        {"prompt": "short", "label": {"ground_truth": "second"}},
        {"prompt": "not read"},
    ]
    selected = collect.prepare_prompts(Tokenizer(), prompt_args(tmp_path, rows))
    assert selected == [
        (1, "ten chars!", list(range(10)), "first"),
        (2, "short", list(range(5)), "second"),
    ]


@pytest.mark.parametrize("key", ["stable_id", ""])
def test_prepare_prompts_uses_explicit_ids_without_sorting(tmp_path, key):
    rows = [{"prompt": "one", key: 91}, {"prompt": "two", key: 7}]
    selected = collect.prepare_prompts(Tokenizer(), prompt_args(tmp_path, rows, prompt_id_key=key))
    assert [item[0] for item in selected] == [91, 7]


def test_prepare_prompts_renders_messages(tmp_path):
    messages = [{"role": "user", "content": "hi"}]
    tokenizer = Tokenizer()
    result = collect.prepare_prompts(tokenizer, prompt_args(tmp_path, [{"prompt": messages}], enable_thinking=False))
    assert result == [(0, "chat:hi", list(range(7)), "")]
    assert tokenizer.rendered == [
        (messages, {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})
    ]


def test_rollout_main_keeps_prompt_groups_and_rank_order(tmp_path, monkeypatch):
    source = prompt_args(tmp_path, [{"prompt": str(i)} for i in range(6)]).input
    asset_path = asset_lock(tmp_path)
    output_dir = tmp_path / "output"
    generated = []

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == "float32"

        def generate(self, prompts, sampling, **kwargs):
            generated.extend(prompts)
            return [SimpleNamespace(outputs=[completion() for _ in range(sampling.n)]) for _ in prompts]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__version__="test", LLM=LLM, SamplingParams=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--model",
            "student",
            "--asset-lock",
            str(asset_path),
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--max-prompts",
            "6",
            "--responses-per-prompt",
            "2",
            "--top-k",
            "2",
            "--shard-size",
            "3",
            "--batch-size",
            "2",
            "--rank",
            "1",
            "--world-size",
            "2",
        ],
    )
    collect.main()
    shards = sorted(output_dir.glob("*.parquet"))
    assert [pq.read_metadata(path).num_rows for path in shards] == [4, 2]
    rows = [row for path in shards for row in pq.read_table(path).to_pylist()]
    assert [row["prompt"] for row in rows] == ["3", "3", "4", "4", "5", "5"]
    assert [row["metadata"]["response_id"] for row in rows] == [0, 1, 0, 1, 0, 1]
    assert len(generated) == 3


@pytest.mark.parametrize('mismatch', ['model', 'revision', 'tokenizer'])
def test_collector_rejects_identity_mismatch_before_loading_engine(tmp_path, monkeypatch, mismatch):
    source = prompt_args(tmp_path, [{'prompt': 'one'}]).input
    asset_path = asset_lock(
        tmp_path, model='other-model' if mismatch == 'model' else 'student',
        revision='other-revision' if mismatch == 'revision' else 'unknown',
        tokenizer_hash='other-tokenizer' if mismatch == 'tokenizer' else None)
    loaded = []

    def forbidden_engine(**kwargs):
        loaded.append(kwargs)
        raise AssertionError('No engine should load for mismatched assets')

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=forbidden_engine))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setattr(sys, 'argv', ['collect', '--model', 'student', '--asset-lock', str(asset_path),
                                   '--input', str(source), '--output-dir', str(tmp_path / 'out'),
                                   '--max-prompts', '1'])
    with pytest.raises(ValueError, match='original checkpoint|Loaded tokenizer'):
        collect.main()
    assert not loaded
    assert not list((tmp_path / 'out').glob('*.parquet'))


def test_collector_requires_requested_number_after_prompt_filtering(tmp_path, monkeypatch):
    source = prompt_args(tmp_path, [{'prompt': 'one'}, {'prompt': 'too long'}]).input
    asset_path = asset_lock(tmp_path)
    loaded = []

    def forbidden_engine(**kwargs):
        loaded.append(kwargs)
        raise AssertionError('No engine should load for an incomplete prompt set')

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=forbidden_engine))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setattr(sys, 'argv', ['collect', '--model', 'student', '--asset-lock', str(asset_path),
                                   '--input', str(source), '--output-dir', str(tmp_path / 'out'),
                                   '--max-prompts', '2', '--max-prompt-length', '3'])
    with pytest.raises(ValueError, match='Requested 2 prompts, but 1 remain'):
        collect.main()
    assert not loaded
