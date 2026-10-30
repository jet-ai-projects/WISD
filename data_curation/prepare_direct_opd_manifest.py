#!/usr/bin/env python3
"""Write the cache index consumed by Offline Direct-OPD training."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import file_sha256, parquet_paths, write_json

SCHEMA_VERSION = "offline_direct_opd_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--asset-lock", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_manifest(
    metadata: Mapping[str, Any],
    *,
    asset_lock_path: Path,
    source_dataset: Path,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    asset_lock = json.loads(asset_lock_path.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "sealed": True,
        "top_k": metadata["generation_config"]["top_k"],
        "student_model": asset_lock["models"]["student"],
        "post_teacher_model": asset_lock["models"]["post_teacher"],
        "pre_teacher_model": asset_lock["models"]["pre_teacher"],
        "asset_lock_sha256": file_sha256(asset_lock_path),
        "tokenizer_hash": metadata["tokenizer_hash"],
        "model_vocab_size": asset_lock["models"]["student"]["model_vocab_size"],
        "token_id_compatibility": asset_lock["token_id_compatibility"],
        "generation_config": metadata["generation_config"],
        "generation_config_hash": metadata["generation_config_hash"],
        "loss_mask_storage_dtype": "bool",
        "score_storage_dtype": "float32",
        "token_storage_dtype": "int32",
        "source_dataset_sha256": file_sha256(source_dataset),
        "total_rows": sum(shard["rows"] for shard in shards),
        "shards": shards,
    }


def shard_record(path: Path, root: Path) -> dict[str, Any]:
    rows = pq.read_metadata(path).num_rows
    return {"path": str(path.relative_to(root)), "rows": rows, "sha256": file_sha256(path)}


def main() -> None:
    args = parse_args()
    if args.manifest_out.exists() and not args.overwrite:
        raise FileExistsError(args.manifest_out)
    paths = parquet_paths(args.input)
    records = [shard_record(path, args.manifest_out.parent) for path in paths]
    total_rows = sum(record["rows"] for record in records)
    first_batch = next(pq.ParquetFile(paths[0]).iter_batches(batch_size=1, columns=["metadata"]))
    metadata = first_batch.to_pylist()[0]["metadata"]
    manifest = make_manifest(
        metadata,
        asset_lock_path=args.asset_lock,
        source_dataset=args.source_dataset,
        shards=records,
    )
    write_json(manifest, args.manifest_out)
    print(f"Prepared training manifest: {args.manifest_out} ({total_rows:,} rows)")


if __name__ == "__main__":
    main()
