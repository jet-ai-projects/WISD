"""Shared hashing, shard discovery, and atomic output for the data pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

OFFLINE_STORAGE_TYPES = {
    "behavior_sampled_log_probs": pa.list_(pa.float32()),
    "behavior_topk_log_probs": pa.list_(pa.list_(pa.float32())),
    "candidate_ids": pa.list_(pa.list_(pa.int32())),
    "loss_mask": pa.list_(pa.bool_()),
    "post_teacher_log_probs": pa.list_(pa.list_(pa.float32())),
    "pre_teacher_log_probs": pa.list_(pa.list_(pa.float32())),
    "prompt_tokens": pa.list_(pa.int32()),
    "response_tokens": pa.list_(pa.int32()),
    "student_ref_sampled_log_probs": pa.list_(pa.float32()),
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Publish a completed file; leave the previous output intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(value: Any, path: str | Path) -> None:
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


@contextmanager
def staged_directory(path: str | Path) -> Iterator[Path]:
    """Build a new dataset privately and publish it only after completion."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing existing output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        yield temporary
        if path.exists():
            raise FileExistsError(f"refusing existing output directory: {path}")
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_parquet(table: pa.Table, path: str | Path, **kwargs: Any) -> None:
    with atomic_output(path) as temporary:
        pq.write_table(table, temporary, **kwargs)


def parquet_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    return [path] if path.is_file() else sorted(path.rglob("*.parquet"))
