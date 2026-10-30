"""Checkpoint-resume compatibility for runtime optimizer-scheduler overrides."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

logger = logging.getLogger(__name__)


def reapply_runtime_lr_bounds(
    args: Namespace,
    optimizer: Any,
    opt_param_scheduler: Any,
    *,
    expected_num_steps: int,
) -> bool:
    """Reapply runtime LR bounds after a full optimizer checkpoint load.

    Older Megatron-LM checkpoint loaders restore ``max_lr``/``min_lr`` inside
    each optimizer parameter group even when
    ``--override-opt-param-scheduler`` is set.  Since
    :meth:`OptimizerParamScheduler.get_lr` prefers those per-group values,
    the restored checkpoint LR silently wins over the requested runtime LR.

    This mirrors the upstream checkpoint-load fix: restore runtime LR bounds,
    align scheduler progress with the resumed data position, then perform a
    zero-increment scheduler step so the optimizer's live ``lr`` fields are
    refreshed immediately.

    Returns:
        ``True`` when an override was applied, otherwise ``False``.
    """

    if not getattr(args, "override_opt_param_scheduler", False):
        return False
    if (
        optimizer is None
        or getattr(optimizer, "is_stub_optimizer", False)
        or opt_param_scheduler is None
    ):
        logger.warning(
            "Skipping runtime LR override because optimizer or scheduler is unavailable"
        )
        return False

    runtime_lr = getattr(args, "lr")
    runtime_min_lr = getattr(args, "min_lr")
    decoupled_lr = getattr(args, "decoupled_lr", None)
    decoupled_min_lr = getattr(args, "decoupled_min_lr", None)
    for param_group in optimizer.param_groups:
        if param_group.get("is_decoupled_lr", False) and decoupled_lr is not None:
            param_group["max_lr"] = decoupled_lr
            param_group["min_lr"] = (
                decoupled_min_lr
                if decoupled_min_lr is not None
                else runtime_min_lr
            )
        else:
            param_group["max_lr"] = runtime_lr
            param_group["min_lr"] = runtime_min_lr

    if opt_param_scheduler.num_steps != expected_num_steps:
        logger.warning(
            "Aligning scheduler num_steps from %s to resumed value %s",
            opt_param_scheduler.num_steps,
            expected_num_steps,
        )
        opt_param_scheduler.num_steps = expected_num_steps
    opt_param_scheduler.step(increment=0)

    effective_lrs = [
        opt_param_scheduler.get_lr(param_group)
        for param_group in optimizer.param_groups
    ]
    logger.info(
        "Reapplied runtime optimizer LR bounds after checkpoint load: "
        "max_lr=%s min_lr=%s effective_lrs=%s num_steps=%s",
        runtime_lr,
        runtime_min_lr,
        effective_lrs,
        opt_param_scheduler.num_steps,
    )
    return True
