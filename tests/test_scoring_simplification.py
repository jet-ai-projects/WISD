"""Regression checks for the shared cached/batched offline scoring core."""

import math
from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data_curation.precompute_direct_opd_scores import (
    SCORE_FIELDS,
    cast_offline_direct_opd_storage,
    normalize_legacy_vllm_yarn_config,
    score_sequence,
    score_sequence_batch,
    write_scored_shard,
)


@pytest.mark.parametrize("factor,legacy_scale", [(1.0, 1.0), (8.0, 0.8), (4.0, 1.2)])
def test_yarn_adapter_preserves_legacy_rotary_scale(factor, legacy_scale):
    config = SimpleNamespace(rope_parameters={"rope_type": "yarn", "factor": factor, "attn_factor": legacy_scale})
    normalize_legacy_vllm_yarn_config(config)
    assert config.rope_parameters == {
        "rope_type": "yarn",
        "factor": factor,
        "attention_factor": (1 + 0.1 * math.log(factor)) * legacy_scale,
    }


class PrefixModel:
    """Small causal model with a real prefix-dependent state and padded output."""

    def __call__(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True):
        previous = 0 if past_key_values is None else past_key_values
        prefixes = input_ids.cumsum(-1) + previous
        axis = torch.arange(13, dtype=torch.float32)
        logits = (prefixes[..., None] * (axis + 1) / 37).sin().to(torch.float16)
        if not use_cache:
            assert attention_mask is not None
            assert torch.all(input_ids[attention_mask == 0] == 0)
        return SimpleNamespace(logits=logits, past_key_values=prefixes[:, -1:])


ROWS = [
    {
        "prompt_tokens": [1, 4, 2],
        "response_tokens": [3, 5, 1, 2],
        "candidate_ids": [[0, 3, 8], [1, 5, 7], [1, 2, 3], [2, 4, 6]],
    },
    {
        "prompt_tokens": [2],
        "response_tokens": [6, 1],
        "candidate_ids": [[0, 6], [1, 7]],
    },
]


def expected_scores(metadata, score_field, vocab_size):
    context = metadata["prompt_tokens"] + metadata["response_tokens"][:-1]
    logits = PrefixModel()(torch.tensor([context])).logits[0].float()
    logits = logits[len(metadata["prompt_tokens"]) - 1 :, :vocab_size]
    targets = torch.tensor(metadata["candidate_ids"])
    sampled = torch.tensor(metadata["response_tokens"])[:, None]
    if score_field == "student_support_topk":
        support_logits, support_ids = logits.topk(targets.shape[1], dim=-1)
        normalizer = logits.logsumexp(-1, keepdim=True)
        return {
            "candidate_ids": support_ids.tolist(),
            "behavior_topk_log_probs": (support_logits - normalizer).tolist(),
            "student_ref_sampled_log_probs": (logits.gather(-1, sampled) - normalizer).flatten().tolist(),
        }
    if score_field == "student_ref_sampled_log_probs":
        return logits.log_softmax(-1).gather(-1, sampled).flatten().tolist()
    return logits.log_softmax(-1).gather(-1, targets).tolist()


@pytest.mark.parametrize("score_field", SCORE_FIELDS)
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 20])
@pytest.mark.parametrize("vocab_size", [None, 9])
def test_cached_and_batched_scores_match_direct_formula(score_field, chunk_size, vocab_size):
    rows = deepcopy(ROWS)
    batch = score_sequence_batch(PrefixModel(), rows, score_field, device="cpu", normalization_vocab_size=vocab_size)
    dispatched = score_sequence_batch(
        PrefixModel(), rows, score_field, device="cpu",
        normalization_vocab_size=vocab_size, chunk_size=chunk_size,
    )
    for metadata, batched, routed in zip(rows, batch, dispatched, strict=True):
        cached = score_sequence(
            PrefixModel(),
            metadata,
            score_field,
            device="cpu",
            chunk_size=chunk_size,
            normalization_vocab_size=vocab_size,
        )
        # Exact Python equality catches arithmetic changes such as replacing
        # top-K logits minus logsumexp with a differently rounded expression.
        assert cached == batched == routed == expected_scores(metadata, score_field, vocab_size)
    assert rows == ROWS


class SparseProjection:
    def prepare_row(self, metadata):
        return [1, 2, 3, 4, 5, 6, 7, 8], [[1, 2], [0, 3], [4, 5]], [2, 3, 5], [True, False, True]


@pytest.mark.parametrize("chunk_size", [4, 6])
def test_dispatch_uses_projected_context_and_omits_unused_tail(chunk_size):
    class RecordingModel(PrefixModel):
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, **kwargs):
            self.calls.append((input_ids.shape[-1], kwargs["use_cache"]))
            return super().__call__(input_ids, **kwargs)

    class CountingProjection(SparseProjection):
        def __init__(self):
            self.calls = 0

        def prepare_row(self, metadata):
            self.calls += 1
            return super().prepare_row(metadata)

    row = {"prompt_tokens": [1], "response_tokens": [2, 3, 4], "candidate_ids": [[0, 1]] * 3}
    model, projection = RecordingModel(), CountingProjection()
    scores = score_sequence_batch(
        model, [row], "post_teacher_log_probs", device="cpu",
        token_projection=projection, chunk_size=chunk_size,
    )
    assert model.calls == ([(4, True), (2, True)] if chunk_size == 4 else [(6, False)])
    assert projection.calls == 1
    assert row["token_projection_valid_mask"] == [True, False, True]
    reference = score_sequence(
        PrefixModel(), deepcopy(row), "post_teacher_log_probs", device="cpu",
        token_projection=SparseProjection(), chunk_size=20,
    )
    assert scores == [reference]


def test_storage_cast_preserves_nulls_and_field_metadata_across_chunks():
    fields = [
        pa.field("prompt_tokens", pa.list_(pa.int64()), metadata={b"kind": b"tokens"}),
        pa.field("post_teacher_log_probs", pa.null()),
        pa.field("custom", pa.string(), metadata={b"kind": b"custom"}),
    ]
    schema = pa.schema(
        [pa.field("metadata", pa.struct(fields), metadata={b"kind": b"offline"})],
        metadata={b"version": b"test"},
    )
    table = pa.Table.from_pylist(
        [{"metadata": {"prompt_tokens": [1, 2], "custom": "keep"}}, {"metadata": None}],
        schema=schema,
    )
    table = pa.concat_tables([table.slice(0, 1), table.slice(1, 1)])
    result = cast_offline_direct_opd_storage(table)
    assert result.to_pylist() == table.to_pylist()
    assert result["metadata"].combine_chunks().is_null().to_pylist() == [False, True]
    assert result.schema.metadata == schema.metadata
    field = result.schema.field("metadata")
    assert field.metadata == schema.field("metadata").metadata
    assert field.type.field("prompt_tokens") == fields[0].with_type(pa.list_(pa.int32()))
    assert field.type.field("prompt_tokens").metadata == fields[0].metadata
    assert field.type.field("post_teacher_log_probs") == fields[1]
    assert field.type.field("custom").metadata == fields[2].metadata


@pytest.mark.parametrize("score_field", ["post_teacher_log_probs", "pre_teacher_log_probs"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 20])
@pytest.mark.parametrize("vocab_size", [None, 9])
def test_projected_cached_and_padded_scores_preserve_sparse_positions_and_masks(score_field, chunk_size, vocab_size):
    metadata = {"prompt_tokens": [1], "response_tokens": [2, 3, 4], "candidate_ids": [[0, 1]] * 3}
    projection = SparseProjection()
    cached = score_sequence(
        PrefixModel(),
        metadata,
        score_field,
        device="cpu",
        chunk_size=chunk_size,
        token_projection=projection,
        normalization_vocab_size=vocab_size,
    )
    batched_row = deepcopy(metadata)
    batched = score_sequence_batch(
        PrefixModel(),
        [batched_row],
        score_field,
        device="cpu",
        token_projection=projection,
        normalization_vocab_size=vocab_size,
    )[0]
    context, targets, indices, mask = projection.prepare_row(metadata)
    logits = PrefixModel()(torch.tensor([context])).logits[0, indices].float()[..., :vocab_size]
    expected = logits.log_softmax(-1).gather(-1, torch.tensor(targets)).tolist()
    assert cached == batched == expected
    assert metadata["token_projection_valid_mask"] == batched_row["token_projection_valid_mask"] == mask


def test_failed_streaming_write_preserves_previous_output_and_cleans_temporary(tmp_path):
    source = tmp_path / "input.parquet"
    destination = tmp_path / "output.parquet"
    table = pa.Table.from_pylist([{"value": 1}, {"value": 2}])
    pq.write_table(table, source)
    pq.write_table(table.slice(0, 1), destination)
    previous = destination.read_bytes()

    def fail_second_row(rows, index):
        if index == 1:
            raise RuntimeError("scorer failed")
        rows[0]["value"] += 1

    with pytest.raises(RuntimeError, match="scorer failed"):
        write_scored_shard(
            source,
            destination,
            score_rows=fail_second_row,
            row_batch_size=1,
            storage_row_group_size=1,
        )
    assert destination.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.parquet", "output.parquet"]
