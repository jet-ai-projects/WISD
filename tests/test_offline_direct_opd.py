from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from slime.rollout.offline_direct_opd import (
    SCHEMA_VERSION,
    OfflineDirectOPDDataError,
    attach_offline_direct_opd_fields,
    compute_offline_direct_opd_objective,
    dense_candidate_log_probs,
    hydrate_offline_direct_opd_sample,
    load_adaptive_kl_state,
    low_var_kl,
    offline_direct_opd_reward_sum_and_count,
    offline_direct_opd_terms,
    offline_direct_opd_tilted_target_terms,
    save_adaptive_kl_state,
    update_kl_loss_coef_from_reward,
    validate_offline_direct_opd_metadata,
    validate_sealed_manifest,
    vocab_parallel_candidate_log_probs,
)
from slime.utils.types import Sample


def _distributed_candidate_logprob_worker(rank: int, world_size: int, init_file: str) -> None:
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(19)
        full_logits = torch.randn(2, 8, dtype=torch.float64)
        candidates = torch.tensor([[0, 4, 7], [1, 3, 6]])
        grad_output = torch.tensor([[0.2, -0.4, 0.7], [-0.1, 0.3, 0.5]], dtype=torch.float64)
        local_logits = full_logits[:, rank * 4 : (rank + 1) * 4].clone().requires_grad_(True)
        dense_logits = full_logits.clone().requires_grad_(True)

        actual = vocab_parallel_candidate_log_probs(local_logits, candidates, dist.group.WORLD)
        expected = dense_logits.log_softmax(dim=-1).gather(dim=-1, index=candidates)
        torch.testing.assert_close(actual.double(), expected)

        (actual * grad_output).sum().backward()
        (expected * grad_output).sum().backward()
        torch.testing.assert_close(
            local_logits.grad,
            dense_logits.grad[:, rank * 4 : (rank + 1) * 4],
        )
    finally:
        dist.destroy_process_group()


def _metadata(top_k: int = 2):
    candidates = [[1, 2][:top_k], [2, 3][:top_k]]
    scores = [[-0.2, -1.2][:top_k], [-0.3, -1.3][:top_k]]
    return {
        "is_offline_direct_opd": True,
        "schema_version": SCHEMA_VERSION,
        "prompt_tokens": [4, 5],
        "response_tokens": [7, 8],
        "response_length": 2,
        "loss_mask": [1, 1],
        "candidate_ids": candidates,
        "behavior_topk_log_probs": scores,
        "post_teacher_log_probs": scores,
        "pre_teacher_log_probs": [[value - 0.1 for value in row] for row in scores],
        "student_ref_sampled_log_probs": [-0.4, -0.5],
        "behavior_sampled_log_probs": [-0.4, -0.5],
        "finish_reason": "stop",
    }


def test_metadata_validation_and_sample_attachment():
    metadata = validate_offline_direct_opd_metadata(_metadata(), expected_top_k=2)
    sample = Sample(metadata=metadata)
    attach_offline_direct_opd_fields(sample, expected_top_k=2)

    assert sample.candidate_ids == [[1, 2], [2, 3]]
    assert sample.rollout_log_probs == [-0.4, -0.5]


def test_trusted_sealed_hydration_tensorizes_once_and_compacts_metadata(monkeypatch):
    import slime.rollout.offline_direct_opd as direct_opd

    metadata = _metadata()
    sample = Sample(prompt=[4, 5], metadata=metadata)

    def fail_if_full_validation_runs(*args, **kwargs):
        raise AssertionError("trusted sealed hydration must not repeat exhaustive row validation")

    monkeypatch.setattr(
        direct_opd,
        "validate_offline_direct_opd_metadata",
        fail_if_full_validation_runs,
    )
    hydrated = direct_opd.hydrate_offline_direct_opd_sample(
        sample,
        tokenizer=None,
        expected_top_k=2,
        trusted_sealed=True,
    )

    assert hydrated.status == Sample.Status.COMPLETED
    torch.testing.assert_close(
        hydrated.tokens,
        torch.tensor([4, 5, 7, 8], dtype=torch.long),
    )
    torch.testing.assert_close(
        hydrated.candidate_ids,
        torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
    )
    torch.testing.assert_close(
        hydrated.post_teacher_log_probs,
        torch.tensor([[-0.2, -1.2], [-0.3, -1.3]]),
    )
    torch.testing.assert_close(
        hydrated.rollout_log_probs,
        torch.tensor(metadata["behavior_sampled_log_probs"]),
    )
    assert hydrated.metadata["is_offline_direct_opd"] is True
    assert "candidate_ids" not in hydrated.metadata
    assert "post_teacher_log_probs" not in hydrated.metadata
    assert "response_tokens" not in hydrated.metadata


def test_trusted_sealed_hydration_accepts_numpy_backed_parquet_fields(monkeypatch):
    import slime.rollout.offline_direct_opd as direct_opd

    metadata = _metadata()
    for field in (
        "prompt_tokens",
        "response_tokens",
        "loss_mask",
        "candidate_ids",
        "behavior_topk_log_probs",
        "post_teacher_log_probs",
        "pre_teacher_log_probs",
        "student_ref_sampled_log_probs",
        "behavior_sampled_log_probs",
    ):
        metadata[field] = np.asarray(metadata[field])
    sample = Sample(prompt=np.asarray([4, 5]), metadata=metadata)

    def fail_if_full_validation_runs(*args, **kwargs):
        raise AssertionError("trusted sealed hydration must not repeat exhaustive row validation")

    monkeypatch.setattr(
        direct_opd,
        "validate_offline_direct_opd_metadata",
        fail_if_full_validation_runs,
    )
    hydrated = direct_opd.hydrate_offline_direct_opd_sample(
        sample,
        tokenizer=None,
        expected_top_k=2,
        trusted_sealed=True,
    )

    assert hydrated.status == Sample.Status.COMPLETED
    torch.testing.assert_close(
        hydrated.tokens,
        torch.tensor([4, 5, 7, 8], dtype=torch.long),
    )
    torch.testing.assert_close(
        hydrated.loss_mask,
        torch.tensor([1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        hydrated.rollout_log_probs,
        torch.tensor([-0.4, -0.5]),
    )
    torch.testing.assert_close(
        hydrated.candidate_ids,
        torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
    )
    torch.testing.assert_close(
        hydrated.post_teacher_log_probs,
        torch.tensor([[-0.2, -1.2], [-0.3, -1.3]]),
    )
    assert "candidate_ids" not in hydrated.metadata
    assert "post_teacher_log_probs" not in hydrated.metadata


def test_offline_direct_opd_data_source_shallow_copies_sealed_rows():
    from slime.rollout.data_source import RolloutDataSource

    class StubDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

    original = Sample(prompt="prompt", metadata={"large_nested_payload": [[1], [2]]})
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = SimpleNamespace(
        n_samples_per_prompt=1,
        offline_direct_opd=True,
        rollout_shuffle=False,
    )
    source.dataset = StubDataset([original])
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.sample_offset = 0

    copied = source.get_samples(1)[0][0]
    assert copied is not original
    assert copied.metadata is original.metadata
    copied.status = Sample.Status.COMPLETED
    assert original.status == Sample.Status.PENDING


def test_metadata_rejects_duplicate_candidates():
    metadata = _metadata()
    metadata["candidate_ids"][0] = [1, 1]
    with pytest.raises(OfflineDirectOPDDataError, match="duplicate"):
        validate_offline_direct_opd_metadata(metadata)


def test_dense_candidate_log_probs_uses_full_vocab_and_preserves_gradient():
    logits = torch.tensor(
        [[0.3, -0.2, 1.1, -1.7], [0.8, -0.4, 0.2, 1.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    candidates = torch.tensor([[0, 2], [1, 3]])

    actual = dense_candidate_log_probs(
        logits,
        candidates,
        temperature=0.7,
    )
    expected = (
        (reference_logits.float() / 0.7)
        .log_softmax(-1)
        .gather(
            dim=-1,
            index=candidates,
        )
    )
    torch.testing.assert_close(actual, expected)

    weights = torch.tensor([[0.2, -0.5], [0.7, 0.1]])
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_chunked_vocab_parallel_candidate_log_probs_matches_dense_gradient():
    torch.manual_seed(29)
    logits = torch.randn(7, 19, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    candidates = torch.tensor(
        [
            [0, 3, 18],
            [1, 7, 11],
            [2, 5, 13],
            [4, 8, 17],
            [6, 9, 14],
            [10, 12, 16],
            [0, 15, 18],
        ]
    )
    weights = torch.randn_like(candidates, dtype=torch.float64)

    actual = vocab_parallel_candidate_log_probs(
        logits,
        candidates,
        chunk_size=2,
    )
    expected = (
        reference_logits.float()
        .log_softmax(dim=-1)
        .gather(
            dim=-1,
            index=candidates,
        )
    )
    torch.testing.assert_close(actual, expected)

    (actual.double() * weights).sum().backward()
    (expected.double() * weights).sum().backward()
    torch.testing.assert_close(
        logits.grad,
        reference_logits.grad,
        atol=5e-7,
        rtol=5e-7,
    )


def test_fsdp_packer_preserves_offline_direct_opd_candidate_axes(monkeypatch):
    # Load the leaf module directly: importing the FSDP package also imports
    # Ray/ring-flash-attn, neither of which is needed for this CPU-only test.
    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "fsdp_utils" / "data_packing.py"
    spec = importlib.util.spec_from_file_location(
        "offline_direct_opd_data_packing_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    data_packing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_packing)
    pack_sequences = data_packing.pack_sequences
    unpack_sequences = data_packing.unpack_sequences

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    tokens = [[10, 11, 12], [20, 21, 22, 23]]
    response_lengths = [2, 2]
    candidate_ids = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
    behavior = [
        [[-0.1, -1.1], [-0.2, -1.2]],
        [[-0.3, -1.3], [-0.4, -1.4]],
    ]
    post = [
        [[-0.2, -1.2], [-0.3, -1.3]],
        [[-0.4, -1.4], [-0.5, -1.5]],
    ]
    pre = [
        [[-0.5, -1.5], [-0.6, -1.6]],
        [[-0.7, -1.7], [-0.8, -1.8]],
    ]
    sampled = [[-0.4, -0.5], [-0.6, -0.7]]

    packed = pack_sequences(
        tokens,
        [[1, 1], [1, 1]],
        [0.0, 0.0],
        [0.0, 0.0],
        response_lengths,
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        candidate_ids=candidate_ids,
        behavior_topk_log_probs=behavior,
        post_teacher_log_probs=post,
        pre_teacher_log_probs=pre,
        student_ref_sampled_log_probs=sampled,
        num_packs=1,
    )
    unpacked = unpack_sequences(packed[0])

    assert len(unpacked) == 2
    for index in range(2):
        torch.testing.assert_close(
            unpacked[index]["candidate_ids"],
            torch.tensor(candidate_ids[index]),
        )
        torch.testing.assert_close(
            unpacked[index]["behavior_topk_log_probs"],
            torch.tensor(behavior[index]),
        )
        torch.testing.assert_close(
            unpacked[index]["post_teacher_log_probs"],
            torch.tensor(post[index]),
        )
        torch.testing.assert_close(
            unpacked[index]["pre_teacher_log_probs"],
            torch.tensor(pre[index]),
        )
        torch.testing.assert_close(
            unpacked[index]["student_ref_sampled_log_probs"],
            torch.tensor(sampled[index]),
        )


def test_direct_objective_matches_formula_and_stops_teacher_gradient():
    candidate_log_probs = torch.tensor(
        [[-0.3, -1.0], [-0.6, -0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    sampled_log_probs = torch.tensor([-0.4, -0.7], dtype=torch.float64, requires_grad=True)
    post = torch.tensor([[-0.2, -0.5], [-0.1, -0.9]], dtype=torch.float64, requires_grad=True)
    pre = torch.tensor([[-0.7, -0.3], [-0.6, -1.1]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([-0.5, -0.4], dtype=torch.float64)
    mask = torch.tensor([1, 0])
    kl_coef = 1.25

    loss, metrics = compute_offline_direct_opd_objective(
        candidate_log_probs,
        sampled_log_probs,
        post,
        pre,
        reference,
        mask,
        kl_coef=kl_coef,
    )

    weights = torch.softmax(candidate_log_probs[0].detach().float(), dim=-1)
    delta = (post[0] - pre[0]).detach().float()
    expected_direct = -(weights * delta).sum()
    log_ratio = reference[0].float() - sampled_log_probs[0].float()
    expected_kl = torch.exp(log_ratio) - 1 - log_ratio
    expected = expected_direct + kl_coef * expected_kl

    torch.testing.assert_close(loss.float(), expected)
    torch.testing.assert_close(metrics["direct_loss"].float(), expected_direct)
    loss.backward()

    assert candidate_log_probs.grad is not None
    assert sampled_log_probs.grad is not None
    assert post.grad is None
    assert pre.grad is None
    assert torch.count_nonzero(candidate_log_probs.grad[1]) == 0
    assert sampled_log_probs.grad[1] == 0


def test_step_zero_matches_official_topk_ppo_loss_metrics_and_gradient():
    """Lock the offline surrogate to Direct-OPD's detached, on-policy PPO path."""

    offline_log_probs = torch.tensor(
        [[-0.2, -1.1, -2.0], [-0.4, -0.7, -1.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    official_log_probs = offline_log_probs.detach().clone().requires_grad_(True)
    post = torch.tensor(
        [[-0.1, -0.8, -1.9], [-0.2, -1.0, -1.4]],
        dtype=torch.float64,
    )
    pre = torch.tensor(
        [[-0.6, -0.5, -2.1], [-0.7, -0.8, -1.6]],
        dtype=torch.float64,
    )
    loss_mask = torch.tensor([1.0, 0.0], dtype=torch.float64)
    sampled = torch.tensor([-0.3, -0.5], dtype=torch.float64)

    terms = offline_direct_opd_terms(
        offline_log_probs,
        sampled,
        post,
        pre,
        sampled,
    )
    offline_loss = (terms["direct_loss"] * loss_mask).sum() / loss_mask.sum()

    # This is the authoritative repository formula: detached student softmax
    # weights multiply the detached post-minus-pre teacher shift, then vanilla
    # Top-K PPO sums -advantage * exp(current-old) over K. At an on-policy
    # update old is the detached current value, so the ratio is exactly one.
    official_weights = torch.softmax(official_log_probs.detach().float(), dim=-1)
    official_delta = (post.float() - pre.float()).detach()
    official_rewards = (official_weights * official_delta).detach()
    official_ratio = torch.exp(official_log_probs.float() - official_log_probs.detach().float())
    official_per_token_loss = -(official_rewards * official_ratio).sum(dim=-1)
    official_loss = (official_per_token_loss * loss_mask).sum() / loss_mask.sum()

    torch.testing.assert_close(terms["direct_loss"], official_per_token_loss)
    torch.testing.assert_close(
        terms["weighted_reward_mean"],
        official_rewards.mean(dim=-1),
    )
    torch.testing.assert_close(
        terms["weighted_reward_token_mean"],
        official_rewards.sum(dim=-1),
    )
    torch.testing.assert_close(offline_loss, official_loss)

    offline_loss.backward()
    official_loss.backward()
    torch.testing.assert_close(offline_log_probs.grad, official_log_probs.grad)
    assert torch.count_nonzero(offline_log_probs.grad[1]) == 0


def test_low_var_kl_matches_official_pre_and_post_clamps():
    current = torch.tensor([25.0, 1.0, -1.0, -25.0], requires_grad=True)
    reference = torch.zeros_like(current)

    actual = low_var_kl(current, reference)
    official_log_ratio = torch.clamp(reference - current, min=-20.0, max=20.0)
    expected = torch.clamp(
        official_log_ratio.exp() - official_log_ratio - 1.0,
        min=-10.0,
        max=10.0,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    actual.sum().backward()
    assert torch.isfinite(current.grad).all()


def test_tilted_target_matches_direct_reward_gradient_at_initialization():
    behavior = torch.log(torch.tensor([[0.55, 0.35]], dtype=torch.float64))
    current = behavior.detach().clone().requires_grad_(True)
    post = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    pre = torch.tensor([[-0.1, 0.1]], dtype=torch.float64)
    alpha = 1.25

    terms = offline_direct_opd_tilted_target_terms(
        current,
        behavior,
        post,
        pre,
        alpha=alpha,
    )
    terms["tilted_target_loss"].sum().backward()
    actual_grad = current.grad.detach().clone()

    direct_current = behavior.detach().clone().requires_grad_(True)
    delta = post - pre
    direct_loss = -(direct_current.softmax(dim=-1) * delta).detach().mul(direct_current).sum()
    direct_loss.backward()
    torch.testing.assert_close(actual_grad, direct_current.grad, rtol=1e-5, atol=1e-6)


def test_tilted_target_has_zero_loss_and_gradient_at_target():
    behavior = torch.log(torch.tensor([[0.55, 0.35], [0.4, 0.5]], dtype=torch.float64))
    post = torch.tensor([[0.4, -0.2], [-0.3, 0.2]], dtype=torch.float64)
    pre = torch.tensor([[-0.1, 0.1], [-0.2, -0.4]], dtype=torch.float64)
    alpha = 1.25

    behavior_candidates = behavior.exp()
    behavior_other = 1.0 - behavior_candidates.sum(dim=-1, keepdim=True)
    behavior_buckets = torch.cat((behavior_candidates, behavior_other), dim=-1)
    rewards = torch.cat((post - pre, torch.zeros(2, 1, dtype=torch.float64)), dim=-1)
    target = (behavior_buckets.log() + rewards / alpha).softmax(dim=-1)
    current = target[:, :-1].log().detach().clone().requires_grad_(True)

    terms = offline_direct_opd_tilted_target_terms(
        current,
        behavior,
        post,
        pre,
        alpha=alpha,
    )
    torch.testing.assert_close(
        terms["tilted_target_loss"],
        torch.zeros_like(terms["tilted_target_loss"]),
        atol=1e-7,
        rtol=0.0,
    )
    terms["tilted_target_loss"].sum().backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current.grad), atol=1e-6, rtol=0.0)


def test_vocab_parallel_candidate_log_probs_matches_dense_autograd():
    torch.manual_seed(7)
    candidates = torch.tensor([[0, 3, 5], [1, 2, 4]])
    grad = torch.randn(2, 3)
    custom_logits = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    dense_logits = custom_logits.detach().clone().requires_grad_(True)

    actual = vocab_parallel_candidate_log_probs(custom_logits, candidates)
    expected = dense_logits.log_softmax(dim=-1).gather(dim=-1, index=candidates)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)

    (actual * grad).sum().backward()
    (expected * grad).sum().backward()
    torch.testing.assert_close(custom_logits.grad, dense_logits.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED_TESTS") != "1",
    reason="set RUN_DISTRIBUTED_TESTS=1 where local Gloo sockets are permitted",
)
def test_two_rank_vocab_parallel_candidate_log_probs(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_candidate_logprob_worker,
        args=(2, str(tmp_path / "gloo-init")),
        nprocs=2,
        join=True,
    )


def test_sealed_manifest_checks_shards(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shard = data_dir / "part-00000.parquet"
    shard.write_bytes(b"synthetic parquet placeholder")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sealed": True,
        "top_k": 16,
        "student_model": {"name": "student", "revision": "abc"},
        "post_teacher_model": {"name": "post", "revision": "def"},
        "pre_teacher_model": {"name": "pre", "revision": "ghi"},
        "tokenizer_hash": "tokenizer-hash",
        "generation_config": {"temperature": 1.0, "top_p": 1.0},
        "loss_mask_storage_dtype": "bool",
        "score_storage_dtype": "float32",
        "token_storage_dtype": "int32",
        "source_dataset_sha256": "source-hash",
        "total_rows": 3,
        "shards": [{"path": shard.name, "sha256": digest, "rows": 3}],
    }
    path = data_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_sealed_manifest(path, expected_top_k=16)["total_rows"] == 3

    mixture_manifest = dict(manifest)
    mixture_manifest["post_teacher_model"] = {
        "model_type": "mixture",
        "revision": "a" * 64,
        "components": [
            {
                "name": "performance",
                "weight": 2 / 3,
                "rows": 2,
                "model": {"revision": "performance-revision"},
                "source_manifest_sha256": "b" * 64,
            },
            {
                "name": "brevity",
                "weight": 1 / 3,
                "rows": 1,
                "model": {"revision": "brevity-revision"},
                "source_manifest_sha256": "c" * 64,
            },
        ],
        "selection": {"unit": "prompt_group", "strategy": "round_robin", "group_size": 4},
    }
    mixture_path = data_dir / "mixture-manifest.json"
    mixture_path.write_text(json.dumps(mixture_manifest), encoding="utf-8")
    assert validate_sealed_manifest(mixture_path, expected_top_k=16)["total_rows"] == 3
    mixture_manifest["post_teacher_model"]["components"][0]["rows"] = 1
    mixture_path.write_text(json.dumps(mixture_manifest), encoding="utf-8")
    with pytest.raises(OfflineDirectOPDDataError, match="component rows"):
        validate_sealed_manifest(mixture_path, expected_top_k=16)

    weighted_manifest = dict(manifest)
    weighted_manifest["total_rows"] = 12
    weighted_manifest["shards"] = [{"path": shard.name, "sha256": digest, "rows": 12}]
    weighted_manifest["post_teacher_model"] = {
        "model_type": "mixture",
        "revision": "d" * 64,
        "components": [
            {
                "name": "performance",
                "weight": 2 / 3,
                "rows": 8,
                "model": {"revision": "performance-revision"},
                "source_manifest_sha256": "e" * 64,
            },
            {
                "name": "brevity",
                "weight": 1 / 3,
                "rows": 4,
                "model": {"revision": "brevity-revision"},
                "source_manifest_sha256": "f" * 64,
            },
        ],
        "selection": {
            "unit": "prompt_group",
            "strategy": "deterministic_proportional_interleave_with_cyclic_reuse",
            "group_size": 4,
            "prompt_exclusive": True,
            "exact_ratio_every_rollout": True,
            "rollout_batch_size": 12,
            "rows_per_component_per_rollout": {
                "performance": 8,
                "brevity": 4,
            },
        },
    }
    weighted_path = data_dir / "weighted-manifest.json"
    weighted_path.write_text(json.dumps(weighted_manifest), encoding="utf-8")
    assert validate_sealed_manifest(weighted_path, expected_top_k=16)["total_rows"] == 12

    weighted_manifest["post_teacher_model"]["components"][0]["weight"] = 0.5
    weighted_manifest["post_teacher_model"]["components"][1]["weight"] = 0.5
    weighted_path.write_text(json.dumps(weighted_manifest), encoding="utf-8")
    with pytest.raises(OfflineDirectOPDDataError, match="weight.*per-rollout"):
        validate_sealed_manifest(weighted_path, expected_top_k=16)

    weighted_manifest["post_teacher_model"]["components"][0]["weight"] = 2 / 3
    weighted_manifest["post_teacher_model"]["components"][1]["weight"] = 1 / 3
    weighted_manifest["post_teacher_model"]["selection"]["rows_per_component_per_rollout"]["performance"] = 4
    weighted_path.write_text(json.dumps(weighted_manifest), encoding="utf-8")
    with pytest.raises(OfflineDirectOPDDataError, match="weight.*per-rollout"):
        validate_sealed_manifest(weighted_path, expected_top_k=16)

    composition_manifest = dict(manifest)
    composition_manifest["post_teacher_model"] = {
        "model_type": "accuracy_priority_composition",
        "revision": "d" * 64,
        "composition_schema_version": "offline_direct_opd_accuracy_priority_v1",
        "conflict_rule": "behavior_weighted_centered_topk_plus_other_dot",
        "conflict_fallback": "accuracy",
        "aligned_rule": "convex_post_logprob_interpolation",
        "efficiency_weight": 0.5,
        "accuracy_model": {"revision": "accuracy-revision"},
        "efficiency_model": {"revision": "efficiency-revision"},
        "accuracy_manifest_sha256": "e" * 64,
        "efficiency_manifest_sha256": "f" * 64,
    }
    composition_path = data_dir / "composition-manifest.json"
    composition_path.write_text(json.dumps(composition_manifest), encoding="utf-8")
    assert validate_sealed_manifest(composition_path, expected_top_k=16)["total_rows"] == 3
    composition_manifest["post_teacher_model"]["efficiency_weight"] = 1.5
    composition_path.write_text(json.dumps(composition_manifest), encoding="utf-8")
    with pytest.raises(OfflineDirectOPDDataError, match="efficiency_weight"):
        validate_sealed_manifest(composition_path, expected_top_k=16)

    shard.write_bytes(b"corrupted")
    with pytest.raises(OfflineDirectOPDDataError, match="checksum mismatch"):
        validate_sealed_manifest(path, expected_top_k=16)


def test_adaptive_kl_sign_rule_clipping_and_checkpoint_state(tmp_path):
    assert update_kl_loss_coef_from_reward(1.0, 0.2, 0.01, 0.5, 2.5) == pytest.approx(1.01)
    assert update_kl_loss_coef_from_reward(1.0, -0.2, 0.01, 0.5, 2.5) == pytest.approx(0.99)
    assert update_kl_loss_coef_from_reward(1.0, 0.0, 0.01, 0.5, 2.5) == pytest.approx(1.0)
    assert update_kl_loss_coef_from_reward(2.5, 0.2, 0.01, 0.5, 2.5) == pytest.approx(2.5)
    assert update_kl_loss_coef_from_reward(0.5, -0.2, 0.01, 0.5, 2.5) == pytest.approx(0.5)

    controller_args = SimpleNamespace(
        save=str(tmp_path),
        load=str(tmp_path),
        offline_direct_opd_kl_coef=1.234,
        offline_direct_opd_kl_eps=0.01,
        offline_direct_opd_kl_min=0.5,
        offline_direct_opd_kl_max=2.5,
    )
    state_path = save_adaptive_kl_state(controller_args, rollout_id=19)
    assert state_path.name == "iter_0000019.json"
    assert load_adaptive_kl_state(controller_args, rollout_id=19) == pytest.approx(1.234)

    controller_args.offline_direct_opd_kl_eps = 0.02
    with pytest.raises(ValueError, match="changed across resume"):
        load_adaptive_kl_state(controller_args, rollout_id=19)


def test_release_checkpoint_is_initialization_not_rollout_zero(tmp_path):
    from slime.utils.checkpoint_utils import is_initialization_checkpoint

    converted = tmp_path / "converted"
    converted.mkdir()
    (converted / "latest_checkpointed_iteration.txt").write_text("release", encoding="utf-8")
    assert is_initialization_checkpoint(converted, is_megatron_checkpoint=True)

    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
    assert not is_initialization_checkpoint(resume, is_megatron_checkpoint=True)
    assert is_initialization_checkpoint(tmp_path / "hf", is_megatron_checkpoint=False)


def test_adaptive_reward_reducer_matches_official_candidate_element_mean():
    candidate_log_probs = torch.tensor([[-0.1, -1.0], [-2.0, -0.2], [-0.4, -0.5]])
    post = torch.tensor([[-0.2, -0.3], [-0.4, -0.5], [-0.6, -0.7]])
    pre = torch.tensor([[-0.6, -0.1], [-0.9, -0.8], [-0.2, -1.0]])
    mask = torch.tensor([1, 0, 1])

    reward_sum, reward_count = offline_direct_opd_reward_sum_and_count(
        candidate_log_probs,
        post,
        pre,
        mask,
    )
    weights = torch.softmax(candidate_log_probs, dim=-1)
    expected_rewards = weights * (post - pre)
    expected = expected_rewards[mask.bool()].mean()

    assert reward_count.item() == 4
    torch.testing.assert_close((reward_sum / reward_count).float(), expected)


def test_fixed_rollout_loader_attaches_direct_opd_fields():
    class Tokenizer:
        @staticmethod
        def encode(prompt, add_special_tokens=False):
            assert prompt == "prompt"
            assert add_special_tokens is False
            return [4, 5]

    sample = Sample(prompt="prompt", metadata=_metadata())
    result = hydrate_offline_direct_opd_sample(sample, Tokenizer(), expected_top_k=2)

    torch.testing.assert_close(
        result.tokens,
        torch.tensor([4, 5, 7, 8], dtype=torch.long),
    )
    assert result.response_length == 2
    torch.testing.assert_close(
        result.candidate_ids,
        torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
    )
    assert result.status is Sample.Status.COMPLETED


def test_parquet_round_trip_and_manifest(monkeypatch, tmp_path):
    from data_curation.common import canonical_hash
    from data_curation.precompute_direct_opd_scores import cast_offline_direct_opd_storage
    from data_curation.prepare_direct_opd_manifest import main

    metadata = _metadata()
    metadata.update(
        {
            "sample_id": "sample-0",
            "prompt_id": 0,
            "group_id": 0,
            "response_id": 0,
            "generation_seed": 42,
            "generation_config_hash": "generation-hash",
            "generation_config": {
                "temperature": 1.0,
                "top_p": 1.0,
                "responses_per_prompt": 4,
                "top_k": 2,
            },
            "student_revision": "student-revision",
            "post_teacher_revision": "post-revision",
            "pre_teacher_revision": "pre-revision",
            "tokenizer_hash": "tokenizer-hash",
            "offline_direct_opd_stage": "fully_scored",
        }
    )
    metadata["generation_config_hash"] = canonical_hash(metadata["generation_config"])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shard = data_dir / "part-00000.parquet"
    table = pa.Table.from_pylist([{"prompt": "prompt", "label": "answer", "metadata": metadata}])
    pq.write_table(cast_offline_direct_opd_storage(table), shard)
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source dataset")
    asset_lock = tmp_path / "assets.json"
    asset_lock.write_text(
        json.dumps(
            {
                "schema_version": "offline_direct_opd_assets_v1",
                "tokenizer_hash": "tokenizer-hash",
                "token_id_compatibility": {"model_vocab_size": 10},
                "models": {
                    "student": {"revision": "student-revision", "model_vocab_size": 10},
                    "post_teacher": {"revision": "post-revision"},
                    "pre_teacher": {"revision": "pre-revision"},
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = data_dir / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_direct_opd_manifest.py",
            "--input",
            str(data_dir),
            "--source-dataset",
            str(source),
            "--asset-lock",
            str(asset_lock),
            "--manifest-out",
            str(manifest),
        ],
    )

    main()
    sealed = validate_sealed_manifest(manifest, expected_top_k=2)
    assert sealed["total_rows"] == 1
    assert sealed["generation_config"]["responses_per_prompt"] == 4
    assert sealed["loss_mask_storage_dtype"] == "bool"
    assert sealed["score_storage_dtype"] == "float32"
    assert sealed["token_storage_dtype"] == "int32"


def test_streaming_scorer_aligns_prefixes_across_chunks():
    from data_curation.precompute_direct_opd_scores import score_sequence

    class TinyCausalModel:
        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            assert use_cache is True
            offset = int(past_key_values or 0)
            positions = torch.arange(offset, offset + input_ids.shape[1], dtype=torch.float32)
            token_axis = torch.arange(20, dtype=torch.float32)
            logits = positions[:, None] * token_axis[None, :] / 10.0
            return SimpleNamespace(logits=logits.unsqueeze(0), past_key_values=offset + input_ids.shape[1])

    metadata = {
        "prompt_tokens": [10, 11],
        "response_tokens": [12, 13],
        "candidate_ids": [[1, 2], [3, 4]],
    }
    sampled = score_sequence(
        TinyCausalModel(),
        metadata,
        "student_ref_sampled_log_probs",
        device="cpu",
        chunk_size=2,
    )
    candidates = score_sequence(
        TinyCausalModel(),
        metadata,
        "post_teacher_log_probs",
        device="cpu",
        chunk_size=2,
    )
    support = score_sequence(
        TinyCausalModel(),
        metadata,
        "student_support_topk",
        device="cpu",
        chunk_size=2,
    )

    token_axis = torch.arange(20, dtype=torch.float32)
    expected_logits = torch.stack([token_axis / 10.0, token_axis * 2 / 10.0])
    expected_sampled = expected_logits.log_softmax(-1).gather(-1, torch.tensor([[12], [13]])).squeeze(-1)
    expected_candidates = expected_logits.log_softmax(-1).gather(-1, torch.tensor([[1, 2], [3, 4]]))
    torch.testing.assert_close(torch.tensor(sampled), expected_sampled)
    torch.testing.assert_close(torch.tensor(candidates), expected_candidates)
    expected_support_scores, expected_support_ids = expected_logits.log_softmax(-1).topk(2, dim=-1)
    assert support["candidate_ids"] == expected_support_ids.tolist()
    torch.testing.assert_close(torch.tensor(support["behavior_topk_log_probs"]), expected_support_scores)
    torch.testing.assert_close(torch.tensor(support["student_ref_sampled_log_probs"]), expected_sampled)


def test_teacher_scorer_renormalizes_over_common_student_vocab():
    from data_curation.precompute_direct_opd_scores import score_sequence_batch

    class PaddedVocabModel:
        def __call__(self, input_ids, attention_mask=None, use_cache=False):
            logits = torch.tensor([0.0, 1.0, 2.0, 100.0]).reshape(1, 1, 4)
            return SimpleNamespace(logits=logits.expand(input_ids.shape[0], input_ids.shape[1], -1))

    metadata = [
        {
            "prompt_tokens": [0],
            "response_tokens": [1],
            "candidate_ids": [[0, 2]],
        }
    ]
    scores = score_sequence_batch(
        PaddedVocabModel(),
        metadata,
        "post_teacher_log_probs",
        device="cpu",
        normalization_vocab_size=3,
    )
    expected = torch.tensor([0.0, 1.0, 2.0]).log_softmax(-1)[torch.tensor([0, 2])]
    torch.testing.assert_close(torch.tensor(scores[0][0]), expected)


def test_cross_tokenizer_student_reference_normalizes_in_student_vocab():
    from data_curation.precompute_direct_opd_scores import TOKEN_PROJECTION_MODE, resolve_normalization_vocab_size

    compatibility = {
        "mode": TOKEN_PROJECTION_MODE,
        "normalization_vocab_sizes": {
            "student": 100_278,
            "pre_teacher": 151_643,
            "post_teacher": 151_643,
        },
    }
    assert (
        resolve_normalization_vocab_size(
            compatibility,
            score_role="student",
            actual_vocab_size=100_278,
        )
        == 100_278
    )
    assert (
        resolve_normalization_vocab_size(
            compatibility,
            score_role="post_teacher",
            actual_vocab_size=151_936,
        )
        == 151_643
    )


def test_model_vocab_size_supports_nested_multimodal_text_config():
    from data_curation.prepare_direct_opd_assets import resolve_model_vocab_size, text_config_value

    assert resolve_model_vocab_size(SimpleNamespace(vocab_size=100_278)) == 100_278
    assert (
        resolve_model_vocab_size(
            SimpleNamespace(
                vocab_size=None,
                text_config=SimpleNamespace(vocab_size=248_320),
            )
        )
        == 248_320
    )
    assert text_config_value(SimpleNamespace(max_position_embeddings=40_960), "max_position_embeddings") == 40_960
    assert (
        text_config_value(
            SimpleNamespace(
                max_position_embeddings=None,
                text_config=SimpleNamespace(max_position_embeddings=65_536),
            ),
            "max_position_embeddings",
        )
        == 65_536
    )
    assert text_config_value(SimpleNamespace(), "max_position_embeddings") is None


def test_cross_tokenizer_projection_expands_context_and_masks_non_atomic_candidates():
    from data_curation.precompute_direct_opd_scores import ExactTokenStringProjection

    class StudentTokenizer:
        def get_vocab(self):
            return {"a": 0, "b": 1, "ab": 2, "x": 3}

    class TeacherTokenizer:
        class Model:
            def tokenize(self, symbol):
                ids = {"a": 4, "b": 5, "x": 6}
                return [
                    SimpleNamespace(value=value, id=ids[value])
                    for value in (["a", "b"] if symbol == "ab" else [symbol])
                ]

        backend_tokenizer = SimpleNamespace(model=Model())

        def get_vocab(self):
            return {"a": 4, "b": 5, "x": 6}

    projection = ExactTokenStringProjection(
        StudentTokenizer(),
        TeacherTokenizer(),
    )
    context, targets, indices, valid_mask = projection.prepare_row(
        {
            "prompt_tokens": [2],
            "response_tokens": [0, 2],
            "candidate_ids": [[0, 1], [2, 3]],
        }
    )

    assert context == [4, 5, 4, 4, 5]
    assert targets == [[4, 5], [0, 6]]
    assert indices == [1, 2]
    assert valid_mask == [True, False]


def test_cross_tokenizer_projection_preserves_incomplete_utf8_byte_symbols():
    from data_curation.precompute_direct_opd_scores import ExactTokenStringProjection

    fragment = "İĺìĿ´"

    class StudentTokenizer:
        def get_vocab(self):
            return {fragment: 0}

    class TeacherTokenizer:
        class Model:
            def tokenize(self, symbol):
                assert symbol == fragment
                return [
                    SimpleNamespace(value="İĺ", id=10),
                    SimpleNamespace(value="ìĿ´", id=11),
                ]

        backend_tokenizer = SimpleNamespace(model=Model())

        def get_vocab(self):
            return {"İĺ": 10, "ìĿ´": 11}

    projection = ExactTokenStringProjection(
        StudentTokenizer(),
        TeacherTokenizer(),
    )
    assert projection.context_ids(0) == [10, 11]


def test_cross_tokenizer_projection_reencodes_unshared_added_tokens_for_context():
    from data_curation.precompute_direct_opd_scores import ExactTokenStringProjection

    control_token = "<｜User｜>"

    class StudentTokenizer:
        def get_vocab(self):
            return {"a": 0, control_token: 1}

        def get_added_vocab(self):
            return {control_token: 1}

    class TeacherTokenizer:
        class Model:
            def tokenize(self, symbol):
                assert symbol == "a"
                return [SimpleNamespace(value="a", id=4)]

        backend_tokenizer = SimpleNamespace(model=Model())

        def get_vocab(self):
            return {"a": 4, "<": 5, "User": 6, ">": 7}

        def encode(self, text, add_special_tokens):
            assert text == control_token
            assert add_special_tokens is False
            return [5, 6, 7]

        def decode(
            self,
            token_ids,
            skip_special_tokens,
            clean_up_tokenization_spaces,
        ):
            assert token_ids == [5, 6, 7]
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return control_token

    projection = ExactTokenStringProjection(
        StudentTokenizer(),
        TeacherTokenizer(),
    )

    assert projection.context_ids(1) == [5, 6, 7]
    context, targets, indices, valid_mask = projection.prepare_row(
        {
            "prompt_tokens": [1],
            "response_tokens": [0],
            "candidate_ids": [[0, 1]],
        }
    )
    assert context == [5, 6, 7, 4]
    assert targets == [[4, 0]]
    assert indices == [2]
    assert valid_mask == [False]


def test_cross_tokenizer_streaming_scorer_handles_sparse_boundaries():
    from data_curation.precompute_direct_opd_scores import ExactTokenStringProjection, score_sequence

    class StudentTokenizer:
        def get_vocab(self):
            return {"a": 0, "b": 1, "ab": 2, "x": 3}

    class TeacherTokenizer:
        class Model:
            def tokenize(self, symbol):
                ids = {"a": 4, "b": 5, "x": 6}
                return [
                    SimpleNamespace(value=value, id=ids[value])
                    for value in (["a", "b"] if symbol == "ab" else [symbol])
                ]

        backend_tokenizer = SimpleNamespace(model=Model())

        def get_vocab(self):
            return {"a": 4, "b": 5, "x": 6}

    class TinyCausalModel:
        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            assert use_cache is True
            offset = int(past_key_values or 0)
            positions = torch.arange(
                offset,
                offset + input_ids.shape[1],
                dtype=torch.float32,
            )
            token_axis = torch.arange(20, dtype=torch.float32)
            logits = positions[:, None] * token_axis[None, :] / 10.0
            return SimpleNamespace(
                logits=logits.unsqueeze(0),
                past_key_values=offset + input_ids.shape[1],
            )

    projection = ExactTokenStringProjection(
        StudentTokenizer(),
        TeacherTokenizer(),
    )
    metadata = {
        "prompt_tokens": [2],
        "response_tokens": [0, 2],
        "candidate_ids": [[0, 1], [2, 3]],
    }
    scores = score_sequence(
        TinyCausalModel(),
        metadata,
        "post_teacher_log_probs",
        device="cpu",
        chunk_size=2,
        token_projection=projection,
    )

    token_axis = torch.arange(20, dtype=torch.float32)
    expected_logits = torch.stack([token_axis / 10.0, token_axis * 2 / 10.0])
    expected = expected_logits.log_softmax(-1).gather(
        -1,
        torch.tensor([[4, 5], [0, 6]]),
    )
    torch.testing.assert_close(torch.tensor(scores), expected)
    assert metadata["token_projection_valid_mask"] == [True, False]


def test_batched_scorer_right_padding_matches_expected_positions():
    from data_curation.precompute_direct_opd_scores import score_sequence_batch

    class TinyBatchCausalModel:
        def __call__(self, input_ids, attention_mask=None, use_cache=False):
            assert use_cache is False
            assert attention_mask is not None
            positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
            token_axis = torch.arange(20, dtype=torch.float32)
            logits = positions[None, :, None] * token_axis[None, None, :] / 10.0
            return SimpleNamespace(logits=logits.expand(input_ids.shape[0], -1, -1).clone())

    metadata_rows = [
        {
            "prompt_tokens": [10, 11],
            "response_tokens": [12, 13],
            "candidate_ids": [[1, 2], [3, 4]],
        },
        {
            "prompt_tokens": [5, 6, 7],
            "response_tokens": [8],
            "candidate_ids": [[9, 10]],
        },
    ]
    sampled = score_sequence_batch(
        TinyBatchCausalModel(),
        metadata_rows,
        "student_ref_sampled_log_probs",
        device="cpu",
    )
    candidates = score_sequence_batch(
        TinyBatchCausalModel(),
        metadata_rows,
        "post_teacher_log_probs",
        device="cpu",
    )
    support = score_sequence_batch(
        TinyBatchCausalModel(),
        metadata_rows,
        "student_support_topk",
        device="cpu",
    )

    token_axis = torch.arange(20, dtype=torch.float32)
    expected_sampled_0 = (token_axis[None, :] * torch.tensor([[1.0], [2.0]]) / 10.0).log_softmax(-1)
    expected_sampled_0 = expected_sampled_0.gather(-1, torch.tensor([[12], [13]])).squeeze(-1)
    expected_sampled_1 = (token_axis * 2 / 10.0).log_softmax(-1)[8]
    torch.testing.assert_close(torch.tensor(sampled[0]), expected_sampled_0)
    torch.testing.assert_close(torch.tensor(sampled[1]), expected_sampled_1.reshape(1))
    expected_candidates_0 = (token_axis[None, :] * torch.tensor([[1.0], [2.0]]) / 10.0).log_softmax(-1)
    expected_candidates_0 = expected_candidates_0.gather(-1, torch.tensor([[1, 2], [3, 4]]))
    expected_candidates_1 = (token_axis * 2 / 10.0).log_softmax(-1)[torch.tensor([9, 10])]
    torch.testing.assert_close(torch.tensor(candidates[0]), expected_candidates_0)
    torch.testing.assert_close(torch.tensor(candidates[1]), expected_candidates_1.reshape(1, 2))
    expected_support_0, expected_support_ids_0 = (
        (token_axis[None, :] * torch.tensor([[1.0], [2.0]]) / 10.0).log_softmax(-1).topk(2, dim=-1)
    )
    expected_support_1, expected_support_ids_1 = (token_axis * 2 / 10.0).log_softmax(-1).topk(2, dim=-1)
    assert support[0]["candidate_ids"] == expected_support_ids_0.tolist()
    assert support[1]["candidate_ids"] == expected_support_ids_1.reshape(1, 2).tolist()
    torch.testing.assert_close(torch.tensor(support[0]["behavior_topk_log_probs"]), expected_support_0)
    torch.testing.assert_close(
        torch.tensor(support[1]["behavior_topk_log_probs"]),
        expected_support_1.reshape(1, 2),
    )
    torch.testing.assert_close(torch.tensor(support[0]["student_ref_sampled_log_probs"]), expected_sampled_0)
    torch.testing.assert_close(
        torch.tensor(support[1]["student_ref_sampled_log_probs"]),
        expected_sampled_1.reshape(1),
    )


def test_scored_shard_writer_buffers_inference_batches_for_storage(tmp_path):
    from data_curation.precompute_direct_opd_scores import write_scored_shard

    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "value": row_index,
                    "metadata": {"score": None, "post_teacher_log_probs": None},
                }
                for row_index in range(5)
            ]
        ),
        source,
    )

    def score_rows(rows, first_row_index):
        for row_index, row in enumerate(rows, start=first_row_index):
            assert row["value"] == row_index
            row["metadata"]["score"] = [float(row_index)]
            row["metadata"]["post_teacher_log_probs"] = [[float(row_index), -1.0]]

    written = write_scored_shard(
        source,
        destination,
        score_rows=score_rows,
        row_batch_size=2,
    )

    metadata = pq.read_metadata(destination)
    assert written == 5
    assert metadata.num_rows == 5
    assert metadata.num_row_groups == 1
    post_leaf = next(
        metadata.schema.column(index)
        for index in range(len(metadata.schema))
        if metadata.schema.column(index).path == "metadata.post_teacher_log_probs.list.element.list.element"
    )
    assert post_leaf.physical_type == "FLOAT"
    assert [row["metadata"]["score"] for row in pq.read_table(destination).to_pylist()] == [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]


def test_skywork_converter_matches_official_dapo_prompt():
    from data_curation.prepare_direct_opd_skywork_math import PROMPT_PREFIX, PROMPT_SUFFIX, convert_row

    converted = convert_row(
        {
            "data_source": "open_math_instruct",
            "prompt": [{"role": "user", "content": "  What is 1+1?  "}],
            "ability": "math",
            "reward_model": {"ground_truth": '["2"]', "style": "rule"},
            "extra_info": {
                "index": 7,
                "model_difficulty": {
                    "DeepSeek-R1-Distill-Qwen-1.5B": 1,
                    "DeepSeek-R1-Distill-Qwen-32B": 1,
                    "DeepSeek-R1-Distill-Qwen-7B": 1,
                },
            },
        },
    )

    assert converted["prompt"] == [{"content": f"{PROMPT_PREFIX}What is 1+1?{PROMPT_SUFFIX}", "role": "user"}]
    assert converted["reward_model"]["ground_truth"] == "2"
    assert converted["extra_info"]["prompt_style"] == "dapo_original"
