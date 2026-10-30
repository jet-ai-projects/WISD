from __future__ import annotations

import numpy as np
import torch

from slime.backends.megatron_utils import cp_utils
from slime.rollout.offline_direct_opd import direct_opd_cpu_tensor


def _response_offsets(total_length: int, response_length: int) -> tuple[tuple[int, int], tuple[int, int]]:
    prompt_length = total_length - response_length
    _, _, logits_offsets, _ = cp_utils.get_logits_and_tokens_offset_with_cp(
        total_length,
        response_length,
    )
    return tuple(
        (
            start - (prompt_length - 1),
            end - (prompt_length - 1),
        )
        for start, end in logits_offsets
    )


def test_direct_opd_object_arrays_tensorize_without_python_expansion():
    candidate_rows = np.empty(3, dtype=object)
    candidate_rows[:] = [
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([3, 4], dtype=np.int32),
        np.asarray([5, 6], dtype=np.int32),
    ]
    score_rows = np.empty(3, dtype=object)
    score_rows[:] = [
        np.asarray([-0.1, -1.1], dtype=np.float32),
        np.asarray([-0.2, -1.2], dtype=np.float32),
        np.asarray([-0.3, -1.3], dtype=np.float32),
    ]
    sampled_scores = np.empty(3, dtype=object)
    sampled_scores[:] = [
        np.float32(-0.1),
        np.float32(-0.2),
        np.float32(-0.3),
    ]

    candidates = direct_opd_cpu_tensor(candidate_rows, dtype=torch.long)
    scores = direct_opd_cpu_tensor(score_rows, dtype=torch.float32)
    sampled = direct_opd_cpu_tensor(sampled_scores, dtype=torch.float32)

    torch.testing.assert_close(
        candidates,
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
    )
    torch.testing.assert_close(
        scores,
        torch.tensor([[-0.1, -1.1], [-0.2, -1.2], [-0.3, -1.3]]),
    )
    torch.testing.assert_close(sampled, torch.tensor([-0.1, -0.2, -0.3]))


def test_direct_opd_response_fields_and_token_reduction_support_context_parallelism(monkeypatch):
    """Every Direct-OPD response row is owned once and reduced once under CP."""

    samples = (
        (19, 13),
        (32, 7),
        (41, 31),
    )
    for cp_size in (2, 4):
        rank = 0
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: cp_size)
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: rank)

        one_dimensional = [
            torch.arange(response_length, dtype=torch.float32)
            for _, response_length in samples
        ]
        candidate_rows = [
            torch.stack((value, value + 1000), dim=-1)
            for value in one_dimensional
        ]
        loss_masks = [
            (value.remainder(3) != 0).to(torch.int32)
            for value in one_dimensional
        ]

        ownership = [
            torch.zeros(response_length, dtype=torch.int32)
            for _, response_length in samples
        ]
        reduced_across_cp = torch.zeros(())
        for cp_rank in range(cp_size):
            rank = cp_rank
            sliced_values = []
            for sample_index, (total_length, response_length) in enumerate(samples):
                sliced = cp_utils.slice_log_prob_with_cp(
                    one_dimensional[sample_index],
                    total_length,
                    response_length,
                )
                sliced_candidates = cp_utils.slice_log_prob_with_cp(
                    candidate_rows[sample_index],
                    total_length,
                    response_length,
                )
                assert sliced_candidates.shape == (sliced.shape[0], 2)
                torch.testing.assert_close(sliced_candidates[:, 0], sliced)

                offsets = _response_offsets(total_length, response_length)
                expected = torch.cat(
                    [
                        one_dimensional[sample_index][start:end]
                        for start, end in offsets
                    ]
                )
                torch.testing.assert_close(sliced, expected)
                for start, end in offsets:
                    ownership[sample_index][start:end] += 1
                sliced_values.append(sliced)

            reduce_local_tokens = cp_utils.get_sum_of_sample_mean(
                [total_length for total_length, _ in samples],
                [response_length for _, response_length in samples],
                loss_masks,
                calculate_per_token_loss=True,
            )
            reduced_across_cp += reduce_local_tokens(torch.cat(sliced_values))

        for owner_count in ownership:
            torch.testing.assert_close(owner_count, torch.ones_like(owner_count))
        expected_sum = sum(
            (value * mask).sum()
            for value, mask in zip(one_dimensional, loss_masks, strict=True)
        )
        torch.testing.assert_close(reduced_across_cp, expected_sum)
