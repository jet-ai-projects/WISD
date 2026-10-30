"""Contracts that keep original/weak scores comparable on fixed trajectories."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data_curation.precompute_direct_opd_scores import _score_logits
from self_distillation.scoring import (
    admit_output, precision_identity, validate_original_identity,
    validate_pair_precision, validate_request, validate_row,
)


def setup(tmp_path, **overrides):
    args = SimpleNamespace(model=str(tmp_path / 'original'), model_revision='revision',
                           input=tmp_path / 'post', output_dir=tmp_path / 'pre',
                           dtype='bfloat16', overwrite=False, attn_implementation='sdpa',
                           weak_anchor='rtn3_g128', weak_anchor_module_scope='all',
                           score_field='pre_teacher_log_probs', rank=0, world_size=1,
                           row_batch_size=8, chunk_size=4096)
    for name, value in overrides.items():
        setattr(args, name, value)
    assets = {'tokenizer_hash': 'vocabulary', 'models': {
        role: {'path': args.model, 'revision': args.model_revision, 'tokenizer_hash': 'vocabulary'}
        for role in ('student', 'post_teacher', 'pre_teacher')}}
    return args, assets


def test_weaken_only_pre_with_original_checkpoint(tmp_path):
    args, assets = setup(tmp_path)
    validate_request(args, assets)
    args.score_field = 'post_teacher_log_probs'
    with pytest.raises(ValueError, match='only when scoring pre'):
        validate_request(args, assets)
    args.score_field = 'pre_teacher_log_probs'
    assets['models']['pre_teacher']['path'] = str(tmp_path / 'different-model')
    with pytest.raises(ValueError, match='original checkpoint'):
        validate_request(args, assets)


def test_clean_and_weak_precision_must_match(tmp_path):
    args, _ = setup(tmp_path)
    precision = precision_identity(args, 151936)
    original_precision = dict(precision, weak_anchor_variant=None, weak_anchor_module_scope=None)
    row = {'post_teacher_log_probs': [[-1.] * 16],
           'post_score_precision_json': json.dumps(original_precision)}
    validate_pair_precision(row, precision)
    for field, replacement in [('normalization_vocab_size', 151669),
                               ('forward_parameter_dtype', 'float32'),
                               ('attention_backend', 'flash_attention_2')]:
        changed = dict(original_precision, **{field: replacement})
        row['post_score_precision_json'] = json.dumps(changed)
        with pytest.raises(ValueError, match=field):
            validate_pair_precision(row, precision)


def test_fixed_support_and_clean_scores_required_before_weak(tmp_path):
    _, assets = setup(tmp_path)
    row = {'tokenizer_hash': 'vocabulary', 'student_revision': 'revision',
           'student_model_path': str(tmp_path / 'original'),
           'post_teacher_revision': 'revision', 'prompt_tokens': [1], 'response_tokens': [2],
           'candidate_ids': [list(range(16))], 'post_teacher_log_probs': [[-4.] * 16],
           'behavior_topk_log_probs': [[-4.] * 16], 'behavior_sampled_log_probs': [-4.]}
    before = deepcopy(row)
    validate_row(row, assets, weak=True)
    assert row == before
    row['pre_teacher_log_probs'] = [[-5.] * 16]
    row['pre_teacher_revision'] = 'revision'
    with pytest.raises(ValueError, match='already contains weak'):
        validate_row(row, assets, weak=True)
    row.pop('pre_teacher_log_probs')
    row['candidate_ids'] = [[0] * 16]
    with pytest.raises(ValueError, match='16 distinct'):
        validate_row(row, assets, weak=True)


def test_same_tokenizer_cache_from_another_original_is_rejected(tmp_path):
    _, assets = setup(tmp_path)
    row = {'tokenizer_hash': 'vocabulary', 'student_revision': 'another-checkpoint'}
    with pytest.raises(ValueError, match='student_revision'):
        validate_row(row, assets)
    assert row['student_revision'] == 'another-checkpoint'


def test_cache_path_is_checked_even_when_revision_labels_match(tmp_path):
    _, assets = setup(tmp_path)
    row = {'tokenizer_hash': 'vocabulary', 'student_revision': 'revision',
           'student_model_path': str(tmp_path / 'different-original')}
    with pytest.raises(ValueError, match='student_model_path'):
        validate_row(row, assets)


def test_loaded_tokenizer_must_match_asset_lock(tmp_path):
    args, assets = setup(tmp_path)
    with pytest.raises(ValueError, match='Loaded tokenizer'):
        validate_original_identity(args.model, args.model_revision, assets, tokenizer_hash='different-vocab')


def test_support_reselection_is_not_a_public_scoring_mode(tmp_path):
    args, assets = setup(tmp_path, score_field='student_support_topk', weak_anchor=None)
    with pytest.raises(ValueError, match='fixed rollout support'):
        validate_request(args, assets)


def test_output_identity_prevents_mixing_configurations(tmp_path):
    args, assets = setup(tmp_path)
    precision = precision_identity(args, 151936)
    first = admit_output(args, assets, precision)
    assert admit_output(args, assets, precision) == first
    args.weak_anchor = 'group_prune_g128_tau1of6_rownorm'
    with pytest.raises(ValueError, match='different scoring configuration'):
        admit_output(args, assets, precision_identity(args, 151936))


def test_full_vocabulary_normalization_keeps_other_mass():
    logits = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 8
    candidates = [list(range(16))]
    actual = torch.tensor(_score_logits(logits, candidates, [2], 'pre_teacher_log_probs', 0))
    expected = logits.log_softmax(-1)[:, :16]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.exp().sum() < 1
    assert not torch.allclose(actual, logits[:, :16].log_softmax(-1))
