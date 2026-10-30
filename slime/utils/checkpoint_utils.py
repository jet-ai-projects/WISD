"""Lightweight helpers for distinguishing initialization and resume checkpoints."""

from pathlib import Path


def is_initialization_checkpoint(path: str | Path, *, is_megatron_checkpoint: bool) -> bool:
    """Return true for HF paths and converted Megatron ``release`` checkpoints."""

    if not is_megatron_checkpoint:
        return True
    tracker = Path(path) / "latest_checkpointed_iteration.txt"
    try:
        return tracker.read_text(encoding="utf-8").strip() == "release"
    except OSError:
        # A direct iter_XXXXXXX path is a real resume even though its parent
        # tracker is not part of that directory.
        return False
