import hashlib
import json
import random
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import prepare_direct_opd_klear_code as prep


def prepared_row(index, content, length=32):
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": f"label-{index}",
        "source_prompt_id": index,
        "prompt_token_length": length,
        "normalized_tokens": prep.normalized_tokens(content),
    }


def test_codesub_wrapper_and_unicode_normalization():
    problem = "Find the sum of all numbers. Input contains N integers."
    wrapped = (
        "Solve the following coding problem using the programming language python:\n\n"
        f"{problem}\n\nNow solve the problem and return the code."
    )
    assert prep.normalized_tokens(wrapped) == prep.normalized_tokens(problem)
    assert prep.normalized_tokens("Ｆｉｎｄ THE SUM!") == ("find", "the", "sum")


def test_benchmark_exclusion_matches_pairwise_reference():
    benchmark = [
        prep.normalized_tokens("one two three four five six seven eight"),
        prep.normalized_tokens("Title tiny task"),
        (),
    ]
    rows = [
        prepared_row(0, "one two three four five six seven eight nine ten"),
        prepared_row(1, "unrelated coding problem"),
        prepared_row(2, "Title tiny task"),
        prepared_row(3, "one two three four five six"),
        prepared_row(4, "one two altered four five changed seven eight"),
    ]
    for size in (1, 3, 5):
        for jaccard, overlap in ((0.8, 0.9), (1.0, 1.0), (0.3, 0.7)):
            expected = set()
            for index, row in enumerate(rows):
                train = prep.shingles(row["normalized_tokens"], size)
                for tokens in benchmark:
                    evaluation = prep.shingles(tokens, size)
                    shared = len(train & evaluation)
                    if row["normalized_tokens"] == tokens or (
                        shared
                        and (
                            shared / len(train | evaluation) >= jaccard
                            or shared / min(len(train), len(evaluation)) >= overlap
                        )
                    ):
                        expected.add(index)
            assert (
                prep.contaminated_indices(
                    rows,
                    benchmark,
                    shingle_size=size,
                    jaccard_threshold=jaccard,
                    overlap_threshold=overlap,
                )
                == expected
            )


def test_filter_then_deduplicate_then_seeded_selection():
    rows = [
        prepared_row(0, "Length duplicate", length=101),
        prepared_row(1, "Length duplicate", length=100),
        prepared_row(2, "Benchmark duplicate"),
        prepared_row(3, "Benchmark duplicate"),
        prepared_row(4, "Unique prompt"),
        prepared_row(5, "UNIQUE PROMPT!"),
        prepared_row(6, "Another unique prompt"),
    ]
    expected_ids = [1, 3, 4, 6]
    random.Random(42).shuffle(expected_ids)
    selected = prep.select_rows(rows, {2}, max_prompt_length=100, num_prompts=3, seed=42)
    assert [row["source_prompt_id"] for row in selected] == expected_ids[:3]
    for row in selected:
        assert row == {key: rows[row["source_prompt_id"]][key] for key in ("prompt", "label", "source_prompt_id")}
    selected = prep.select_rows(rows, {2}, max_prompt_length=100, num_prompts=5, seed=42)
    assert [row["source_prompt_id"] for row in selected] == expected_ids


class StubTokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
        return "<user>" + prompt[0]["content"] + "</user><assistant><think>"

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_length": True,
        }
        return {"length": [len(text) for text in texts]}


def write_source(path):
    rows = [
        {
            "prompt": [{"role": "user", "content": f"Find solution {index}"}],
            "reward_model": {"ground_truth": '{ "inputs": ["1"], "outputs": ["2"] }'},
        }
        for index in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return rows


@pytest.mark.parametrize("tokenizer_batch_size,arrow_batch_size", [(1, 1), (2, 3), (8, 8)])
def test_batches_keep_original_prompts_raw_json_labels_and_source_order(
    tmp_path, tokenizer_batch_size, arrow_batch_size
):
    path = tmp_path / "source.parquet"
    source = write_source(path)
    rows = prep.prepare_rows(
        [path],
        StubTokenizer(),
        tokenizer_batch_size=tokenizer_batch_size,
        arrow_batch_size=arrow_batch_size,
    )
    assert [row["source_prompt_id"] for row in rows] == list(range(5))
    for row, original in zip(rows, source, strict=True):
        assert row["prompt"] == original["prompt"]
        digest = hashlib.sha256(original["reward_model"]["ground_truth"].encode()).hexdigest()
        assert row["label"] == f"sha256:{digest}"
        assert row["prompt_token_length"] == len(
            "<user>" + original["prompt"][0]["content"] + "</user><assistant><think>"
        )


def write_benchmark_config(tmp_path):
    raw = tmp_path / "lcb.jsonl"
    tasks = [
        {"contest_date": "2024-01-01", "question_content": "excluded by start date"},
        {
            "contest_date": "2025-01-01T00:00:00",
            "question_title": "Title",
            "question_content": "Completely separate benchmark question",
            "starter_code": "def solve():",
        },
    ]
    raw.write_text("\n".join(json.dumps(task) for task in tasks) + "\n")
    path = tmp_path / "lcb.json"
    path.write_text(json.dumps({"start_date": "2024-08-01", "source_files": [{"path": str(raw)}]}))
    return path


def test_benchmark_date_selection_includes_title_content_and_starter_code(tmp_path):
    config = write_benchmark_config(tmp_path)
    assert prep.load_lcb_tokens(config) == [
        prep.normalized_tokens("Title\nCompletely separate benchmark question\ndef solve():")
    ]


def test_main_writes_only_training_fields_with_deterministic_selection(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = write_source(source_dir / "anything.parquet")
    output = tmp_path / "selected.parquet"
    args = SimpleNamespace(
        input_dir=source_dir,
        tokenizer="stub",
        lcb_lock=write_benchmark_config(tmp_path),
        output=output,
        num_prompts=3,
        max_prompt_length=100,
        selection_seed=42,
        tokenizer_batch_size=2,
        arrow_batch_size=1,
        shingle_size=5,
        near_duplicate_jaccard=0.8,
        near_duplicate_overlap=0.9,
        overwrite=False,
    )
    monkeypatch.setattr(prep, "parse_args", lambda: args)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: StubTokenizer())),
    )
    prep.main()
    table = pq.read_table(output)
    assert table.column_names == ["prompt", "label", "source_prompt_id"]
    expected_ids = list(range(len(source)))
    random.Random(42).shuffle(expected_ids)
    assert table["source_prompt_id"].to_pylist() == expected_ids[:3]
    assert [row["prompt"] for row in table.to_pylist()] == [source[i]["prompt"] for i in expected_ids[:3]]
