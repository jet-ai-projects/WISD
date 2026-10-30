from argparse import Namespace

from slime.utils.scheduler_override import reapply_runtime_lr_bounds


class FakeOptimizer:
    def __init__(self, param_groups, *, is_stub_optimizer=False):
        self.param_groups = param_groups
        self.is_stub_optimizer = is_stub_optimizer


class FakeScheduler:
    def __init__(self, optimizer, *, num_steps):
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.step_calls = []

    def get_lr(self, param_group):
        return param_group["max_lr"]

    def step(self, increment):
        self.step_calls.append(increment)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(param_group)


def runtime_args(**overrides):
    values = {
        "override_opt_param_scheduler": True,
        "lr": 2.5e-7,
        "min_lr": 0.0,
        "decoupled_lr": None,
        "decoupled_min_lr": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_reapplies_runtime_lr_after_checkpoint_param_groups_overwrite_it():
    optimizer = FakeOptimizer(
        [
            {"max_lr": 1e-6, "min_lr": 0.0, "lr": 1e-6},
            {"max_lr": 1e-6, "min_lr": 0.0, "lr": 1e-6},
        ]
    )
    scheduler = FakeScheduler(optimizer, num_steps=12_800)

    applied = reapply_runtime_lr_bounds(
        runtime_args(),
        optimizer,
        scheduler,
        expected_num_steps=12_800,
    )

    assert applied
    assert scheduler.step_calls == [0]
    assert scheduler.num_steps == 12_800
    assert [group["max_lr"] for group in optimizer.param_groups] == [2.5e-7] * 2
    assert [group["lr"] for group in optimizer.param_groups] == [2.5e-7] * 2


def test_realigns_scheduler_and_preserves_decoupled_runtime_bounds():
    optimizer = FakeOptimizer(
        [
            {"max_lr": 1e-6, "min_lr": 0.0, "lr": 1e-6},
            {
                "is_decoupled_lr": True,
                "max_lr": 1e-6,
                "min_lr": 0.0,
                "lr": 1e-6,
            },
        ]
    )
    scheduler = FakeScheduler(optimizer, num_steps=0)

    reapply_runtime_lr_bounds(
        runtime_args(decoupled_lr=7.5e-7, decoupled_min_lr=1e-8),
        optimizer,
        scheduler,
        expected_num_steps=12_800,
    )

    assert scheduler.num_steps == 12_800
    assert optimizer.param_groups[0]["max_lr"] == 2.5e-7
    assert optimizer.param_groups[0]["min_lr"] == 0.0
    assert optimizer.param_groups[1]["max_lr"] == 7.5e-7
    assert optimizer.param_groups[1]["min_lr"] == 1e-8


def test_noop_when_override_is_disabled_or_optimizer_is_stub():
    optimizer = FakeOptimizer([{"max_lr": 1e-6, "lr": 1e-6}])
    scheduler = FakeScheduler(optimizer, num_steps=12_800)

    assert not reapply_runtime_lr_bounds(
        runtime_args(override_opt_param_scheduler=False),
        optimizer,
        scheduler,
        expected_num_steps=12_800,
    )
    assert not reapply_runtime_lr_bounds(
        runtime_args(),
        FakeOptimizer([], is_stub_optimizer=True),
        scheduler,
        expected_num_steps=12_800,
    )
    assert scheduler.step_calls == []
    assert optimizer.param_groups[0]["lr"] == 1e-6
