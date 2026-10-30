#!/usr/bin/env python3
"""Download and lock a LiveCodeBench release without executing dataset code."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(release_version):
    if release_version not in {"release_v5", "release_v6"}:
        raise ValueError("Supported releases: release_v5 and release_v6")
    count = int(release_version.removeprefix("release_v"))
    return ["test.jsonl"] + [f"test{index}.jsonl" for index in range(2, count + 1)]


def load_locked_rows(lock_path):
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text())
    start = datetime.fromisoformat(lock["start_date"])
    rows = []
    for source in lock["source_files"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = lock_path.parent / path
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if datetime.fromisoformat(row["contest_date"]) >= start:
                    rows.append(row)
    rows.sort(key=lambda row: row["question_id"])
    if [row["question_id"] for row in rows] != lock["question_ids"]:
        raise ValueError("Dataset questions do not match the lock")
    return lock, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-version", choices=("release_v5", "release_v6"), default="release_v6")
    parser.add_argument("--start-date")
    parser.add_argument("--expected-problems", type=int)
    parser.add_argument("--revision", default="main", help="HF dataset commit or revision")
    parser.add_argument("--raw-dir", type=Path, help="Directory containing downloaded test*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    defaults = {"release_v5": ("2024-08-01", 279), "release_v6": ("2025-02-01", 131)}
    start_date = args.start_date or defaults[args.release_version][0]
    expected = args.expected_problems or defaults[args.release_version][1]
    filenames = release_files(args.release_version)
    if args.raw_dir is None:
        from huggingface_hub import snapshot_download

        raw_dir = Path(snapshot_download(
            "livecodebench/code_generation_lite",
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=filenames,
        ))
    else:
        raw_dir = args.raw_dir.resolve()
    start = datetime.fromisoformat(start_date)
    rows, sources = [], []
    for filename in filenames:
        path = raw_dir / filename
        sources.append({
            "name": filename, "path": str(path.absolute()),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        })
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if datetime.fromisoformat(row["contest_date"]) >= start:
                    rows.append(row)
    rows.sort(key=lambda row: row["question_id"])
    ids = [row["question_id"] for row in rows]
    if len(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError(f"Expected {expected} unique questions; found {len(ids)}")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_bytes(row) + b"\n")
    lock = {
        # This schema also feeds prepare_direct_opd_klear_code.py.
        "schema_version": "lightning_dopd_lcb_v6_dataset_lock_v1",
        "dataset": "livecodebench/code_generation_lite",
        "release_version": args.release_version,
        "revision": args.revision,
        "start_date": start_date,
        "problem_count": len(rows),
        "question_ids": ids,
        "question_ids_sha256": hashlib.sha256(canonical_bytes(ids)).hexdigest(),
        "filtered_task_content_sha256": digest.hexdigest(),
        "source_files": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Locked {len(rows)} questions in {args.output}")


if __name__ == "__main__":
    main()
