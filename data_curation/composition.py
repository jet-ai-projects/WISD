"""Arrow operations for cached-target transformations."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def flatten_metadata(table: pa.Table) -> pa.StructArray:
    return table["metadata"].combine_chunks()


def replace_metadata_fields(table: pa.Table, replacements: dict[str, pa.Array]) -> pa.Table:
    metadata = flatten_metadata(table)
    arrays = [replacements.get(field.name, metadata.field(field.name)) for field in metadata.type]
    fields = [
        pa.field(field.name, array.type, nullable=field.nullable, metadata=field.metadata)
        for field, array in zip(metadata.type, arrays)
    ]
    rebuilt = pa.StructArray.from_arrays(arrays, fields=fields, mask=metadata.is_null())
    index = table.schema.get_field_index("metadata")
    return table.set_column(index, table.schema.field(index).with_type(rebuilt.type), rebuilt)


def table_chunks(
    manifest_path: Path, manifest: Mapping[str, Any], *, rows: int, chunk_rows: int
) -> Iterator[pa.Table]:
    """Stream a row prefix into fixed-size output chunks across source shards."""
    pieces, buffered = [], 0
    for shard in manifest["shards"]:
        for batch in pq.ParquetFile(manifest_path.parent / shard["path"]).iter_batches(batch_size=chunk_rows):
            table = pa.Table.from_batches([batch])
            offset = 0
            while offset < table.num_rows and rows:
                take = min(chunk_rows - buffered, table.num_rows - offset, rows)
                pieces.append(table.slice(offset, take))
                buffered += take
                offset += take
                rows -= take
                if buffered == chunk_rows or not rows:
                    yield pa.concat_tables(pieces)
                    pieces, buffered = [], 0
            if not rows:
                return
    if pieces:
        yield pa.concat_tables(pieces)


def trainable_tokens(table: pa.Table) -> int:
    mask = table["loss_mask"] if "loss_mask" in table.column_names else flatten_metadata(table).field("loss_mask")
    return int(pc.sum(pc.cast(pc.list_flatten(mask), pa.int64())).as_py() or 0)
