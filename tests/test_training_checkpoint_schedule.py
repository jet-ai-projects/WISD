"""Exercise the production training loop with CPU actor/rollout stand-ins."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_training(monkeypatch, *, batches, interval, epoch_batches=None, start=0, end=None):
    trained, saved, saved_rollouts = [], [], []

    class Actor:
        def async_train(self, rollout_id, data):
            assert data == rollout_id
            trained.append(rollout_id)

        def save_model(self, rollout_id):
            # A checkpoint must include the completed update for its rollout.
            assert trained[-1] == rollout_id
            saved.append(rollout_id)

        def update_weights(self):
            pass

        def clear_memory(self):
            pass

    manager = SimpleNamespace(
        generate=SimpleNamespace(remote=lambda rollout_id: rollout_id),
        save=SimpleNamespace(remote=saved_rollouts.append),
        dispose=SimpleNamespace(remote=lambda: None),
    )
    actor = Actor()
    modules = {
        "ray": SimpleNamespace(get=lambda value: value),
        "sglang.srt.constants": SimpleNamespace(
            GPU_MEMORY_TYPE_KV_CACHE="kv", GPU_MEMORY_TYPE_WEIGHTS="weights",
            GPU_MEMORY_TYPE_CUDA_GRAPH="graph",
        ),
        "slime.ray.placement_group": SimpleNamespace(
            create_placement_groups=lambda args: {"rollout": None},
            create_rollout_manager=lambda args, placement: (manager, epoch_batches),
            create_training_models=lambda args, placements, rollouts: (actor, None),
        ),
        "slime.utils.arguments": SimpleNamespace(parse_args=lambda: None),
        "slime.utils.logging_utils": SimpleNamespace(configure_logger=lambda: None),
        "slime.utils.tracking_utils": SimpleNamespace(init_tracking=lambda args: None),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("wisd_checkpoint_test_train", ROOT / "train.py")
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    args = SimpleNamespace(
        num_rollout=batches, start_rollout_id=start, rollout_end_step=end,
        save_interval=interval, eval_interval=None, rollout_global_dataset=True,
        offload_rollout=False, offload_train=False, enable_weights_backuper=False,
        use_critic=False, num_critic_only_steps=0, update_weights_interval=1,
    )
    entrypoint.train(args)
    assert trained == list(range(start, batches if end is None else end))
    assert saved_rollouts == saved
    return saved


@pytest.mark.parametrize(
    "batches,interval,epoch_batches,expected",
    [
        (25, 50, 50, [24]),  # 100 updates finish before the first scheduled save.
        (60, 50, None, [49, 59]),  # Save the final partial interval as well.
        (50, 50, 50, [49]),  # The default 200-update run saves only once.
        (60, 50, 20, [19, 39, 49, 59]),  # Existing epoch saves are retained.
        (25, None, 50, []),  # Explicitly disabled checkpointing stays disabled.
    ],
)
def test_final_checkpoint_matches_custom_training_budget(
    monkeypatch, batches, interval, epoch_batches, expected
):
    assert run_training(
        monkeypatch, batches=batches, interval=interval, epoch_batches=epoch_batches
    ) == expected


def test_resumed_run_saves_its_final_checkpoint(monkeypatch):
    assert run_training(monkeypatch, batches=60, interval=50, start=50) == [59]


def test_segment_boundary_does_not_change_the_global_training_horizon(monkeypatch):
    assert run_training(monkeypatch, batches=50, interval=50, end=25) == []
