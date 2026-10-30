"""Numerical and parameter-preservation tests for the released WISD operators."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from self_distillation.anchors._qwen3 import PROJECTIONS, transform_row_blocks
from self_distillation.anchors.pruning import apply_pruning, transform_weight
from self_distillation.anchors.rtn import apply_quantization, quantize_weight


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_rtn_groups_ties_and_zero_groups():
    weight = torch.zeros(2, 256)
    weight[0, :8] = torch.tensor([3., .5, 1.5, 2.5, -.5, -1.5, -2.5, -3.])
    weight[0, 128:132] = torch.tensor([6., 1., 3., -3.])
    result = quantize_weight(weight, bits=3, group_size=128)
    torch.testing.assert_close(result[0, :8], torch.tensor([3., 0., 2., 2., 0., -2., -2., -3.]))
    torch.testing.assert_close(result[0, 128:132], torch.tensor([6., 0., 4., -4.]))
    assert torch.equal(result[1], weight[1])


def test_pruning_threshold_equality_and_whole_row_norm():
    weight = torch.zeros(2, 256)
    weight[0, :4] = torch.tensor([6., 1., -1., 2.])
    weight[0, 128:132] = torch.tensor([12., 2., -2., 3.])
    plain = transform_weight(weight, 'group_prune_g128_tau1of6')
    assert plain[0, 1] == plain[0, 2] == plain[0, 129] == plain[0, 130] == 0
    normalized = transform_weight(weight, 'group_prune_g128_tau1of6_rownorm')
    assert torch.equal(plain != 0, normalized != 0)
    torch.testing.assert_close(normalized.norm(dim=1), weight.norm(dim=1), rtol=2e-6, atol=0)
    # One multiplier for the entire row, not a separate norm per group.
    factors = normalized[plain != 0] / plain[plain != 0]
    torch.testing.assert_close(factors, factors[0].expand_as(factors))
    assert torch.count_nonzero(normalized[1]) == 0


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_row_block_writeback_preserves_half_precision(dtype):
    generator = torch.Generator().manual_seed(12)
    weight = nn.Parameter(torch.randn(35, 256, generator=generator).to(dtype))
    original = weight.detach().clone()
    expected = quantize_weight(original.float(), bits=3, group_size=128).to(dtype)
    identity = id(weight)
    transform_row_blocks(weight, lambda rows: quantize_weight(rows, bits=3, group_size=128))
    assert id(weight) == identity and weight.dtype == dtype
    torch.testing.assert_close(weight, expected, rtol=0, atol=0)


class TinyQwen3(nn.Module):
    """The same 36-layer projection inventory, at CPU-test dimensions."""

    def __init__(self):
        super().__init__()
        values = dict(num_hidden_layers=36, hidden_size=128, intermediate_size=128,
                      num_attention_heads=1, num_key_value_heads=1, head_dim=128,
                      vocab_size=128, tie_word_embeddings=True)
        self.config = SimpleNamespace(**values, to_dict=lambda: deepcopy(values))
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 128)
        self.model.layers = nn.ModuleList()
        for _ in range(36):
            layer = nn.Module()
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
            for path in PROJECTIONS:
                prefix, name = path.split('.')
                setattr(getattr(layer, prefix), name, nn.Linear(128, 128, bias=True))
            layer.norm = nn.LayerNorm(128)
            self.model.layers.append(layer)
        self.lm_head = nn.Linear(128, 128, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.to(torch.bfloat16)


@pytest.mark.parametrize('method,variant', [
    (apply_quantization, 'rtn3_g128'),
    (apply_pruning, 'group_prune_g128_tau1of6_rownorm'),
])
def test_model_adapter_preserves_excluded_weights_aliases_and_dtype(method, variant):
    model = TinyQwen3()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    aliases = {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
    report = method(model, variant, require_original=False)
    targets = set(report['target_names'])
    assert len(targets) == 252
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name in targets)
    for name, value in model.named_parameters():
        assert value.dtype == torch.bfloat16
        if name not in targets:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert aliases == {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with pytest.raises(ValueError, match='exactly once'):
        method(model, variant, require_original=False)


def test_adapter_rejects_an_unrecognized_architecture_before_transforming():
    model = TinyQwen3()
    original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    with pytest.raises(ValueError, match='Wrong original Qwen3-4B'):
        apply_quantization(model, 'rtn3_g128')
    torch.testing.assert_close(model.model.layers[0].self_attn.q_proj.weight, original, rtol=0, atol=0)
