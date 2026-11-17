import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient

if TYPE_CHECKING:
    import pyarrow as pa

SCROLL_BATCH = 1000
# Hugging Face's push_to_hub cuts Parquet shards at about this size by default.
MAX_SHARD_BYTES = 500 * 1024 * 1024
COLUMN_TYPES = {"string": "string", "bool": "bool_"}


def export_collection(
    client: QdrantClient,
    collection: str,
    field_types: tuple[tuple[str, str], ...],
    out_dir: Path,
) -> list[tuple[Path, int]]:
    """Write every point of the collection as Parquet shards in the Hugging Face layout.

    Each row holds the point ID, the payload `fields` in order, and the vector. Returns each shard with its row count.
    """
    fields = dict(field_types)
    # Imported here because pyarrow is the optional `export` extra.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [("id", pa.string())]
        + [(name, getattr(pa, COLUMN_TYPES[kind])()) for name, kind in fields.items()]
        + [("vector", pa.list_(pa.float32()))]
    )
    total = client.count(collection, exact=True).count
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("*.parquet"):
        stale.unlink()

    shards: list[tuple[Path, int]] = []
    writer: pq.ParquetWriter | None = None
    rows_per_shard = shard_count = 0
    for batch in record_batches(client, collection, fields, schema):
        if not rows_per_shard:
            rows_per_shard = max(
                1, MAX_SHARD_BYTES * batch.num_rows // max(batch.nbytes, 1)
            )
            shard_count = math.ceil(total / rows_per_shard)
        while batch.num_rows:
            if writer is None or shards[-1][1] == rows_per_shard:
                if writer is not None:
                    writer.close()
                path = (
                    data_dir / f"train-{len(shards):05d}-of-{shard_count:05d}.parquet"
                )
                writer = pq.ParquetWriter(path, schema)
                shards.append((path, 0))
            take = min(batch.num_rows, rows_per_shard - shards[-1][1])
            writer.write_batch(batch.slice(0, take))
            shards[-1] = (shards[-1][0], shards[-1][1] + take)
            batch = batch.slice(take)
    if writer is not None:
        writer.close()

    written = sum(rows for _, rows in shards)
    if written != total:
        raise RuntimeError(
            f"exported {written} rows but the collection has {total} points"
        )
    return shards


def record_batches(
    client: QdrantClient, collection: str, fields: dict[str, str], schema: "pa.Schema"
) -> Iterator["pa.RecordBatch"]:
    import pyarrow as pa

    offset: Any = None
    while True:
        points, next_offset = client.scroll(
            collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if points:
            columns: dict[str, list[Any]] = {"id": [str(p.id) for p in points]}
            for name in fields:
                columns[name] = [(p.payload or {}).get(name) for p in points]
            columns["vector"] = [single_vector(p.vector) for p in points]
            yield pa.RecordBatch.from_pydict(columns, schema=schema)
        if next_offset is None:
            return
        offset = next_offset


def single_vector(vector: Any) -> list[float]:
    """Return the point's only vector, whether it is unnamed or stored under a single name."""
    if isinstance(vector, dict):
        if len(vector) != 1:
            raise ValueError(f"expected one vector per point, found {sorted(vector)}")
        return next(iter(vector.values()))
    return vector
