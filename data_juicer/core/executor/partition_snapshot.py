"""Durable identity of a checkpointed run: ordered input content and operator config."""

import hashlib
import json

import pyarrow as pa
import ray

from data_juicer.core.executor.partition_protocol import (
    COMMIT_DIR,
    KEY_COLUMNS,
    PROTOCOL_VERSION,
    RUN_KEY,
    ProtocolError,
)

_HASH_ALGORITHM = "sha256-sequence-v1"
_HASH_MODULUS = 1 << 256
_HASH_BASE = int.from_bytes(hashlib.sha256(b"data-juicer-partition-sequence-v1").digest(), "big") | 1
_OFFSET_COLUMN = "__dj_partition_offset__"


def _canonical_row_bytes(row) -> bytes:
    """Encode an input row for a diagnostic fingerprint, not a dataset version.

    Non-JSON values use their display representation, which need not include
    every tensor element or the contents of externally referenced files.
    """
    try:
        serialized = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        serialized = str(row)
    return serialized.encode("utf-8")


def _hash_rows(table: pa.Table) -> int:
    """Order-sensitive rolling hash of one batch."""
    sequence_hash = 0
    for row in table.to_pylist():
        digest = int.from_bytes(hashlib.sha256(_canonical_row_bytes(row)).digest(), "big")
        sequence_hash = (sequence_hash * _HASH_BASE + digest) % _HASH_MODULUS
    return sequence_hash


def _combine(partials) -> str:
    """Combine ordered batch hashes independently of batch boundaries."""
    row_count, sequence_hash = 0, 0
    for partial in partials:
        rows = int(partial["row_count"])
        sequence_hash = (
            sequence_hash * pow(_HASH_BASE, rows, _HASH_MODULUS) + int(partial["sequence_hash"], 16)
        ) % _HASH_MODULUS
        row_count += rows
    return hashlib.sha256(f"{_HASH_ALGORITHM}:{row_count}:{sequence_hash:064x}".encode("ascii")).hexdigest()


def _scan_batch(batch):
    offsets = batch[_OFFSET_COLUMN]
    table = batch.drop_columns([_OFFSET_COLUMN])
    rows = table.num_rows
    first = offsets[0].as_py() if rows else 0
    if rows and offsets[rows - 1].as_py() - first != rows - 1:
        raise ProtocolError("Input block row offsets are not contiguous")
    schema_hash = hashlib.sha256(table.schema.serialize().to_pybytes()).digest()
    return pa.table(
        {
            "start": [first],
            "row_count": [rows],
            "sequence_hash": [f"{_hash_rows(table):064x}"],
            "schema_hash": [schema_hash],
        }
    )


def _scan(dataset):
    """Materialize and fingerprint the ordered input on Ray workers."""
    data = dataset.materialize()
    total_rows = data.count()
    schema = data.schema()
    schema = getattr(schema, "base_schema", schema)
    if not isinstance(schema, pa.Schema):
        raise ProtocolError("Input needs a concrete Arrow schema")
    reserved = {_OFFSET_COLUMN, *KEY_COLUMNS} & set(schema.names)
    if reserved:
        raise ProtocolError(f"Input contains reserved checkpoint columns {sorted(reserved)}")
    schema_bytes = schema.serialize().to_pybytes()
    partials = []
    if total_rows:
        offsets = ray.data.range(total_rows).rename_columns({"id": _OFFSET_COLUMN})
        partials = data.zip(offsets).map_batches(_scan_batch, batch_format="pyarrow", batch_size=None).take_all()
        partials = sorted((item for item in partials if item["row_count"]), key=lambda item: item["start"])
        expected_start = 0
        for partial in partials:
            if partial["start"] != expected_start:
                raise ProtocolError("Input blocks do not cover the input rows exactly once")
            expected_start += partial["row_count"]
        if expected_start != total_rows:
            raise ProtocolError("Input blocks do not cover the input rows exactly once")
        if {partial["schema_hash"] for partial in partials} != {hashlib.sha256(schema_bytes).digest()}:
            raise ProtocolError("Input schema changed between batches")
    fingerprint = hashlib.sha256(schema_bytes + _combine(partials).encode("ascii")).hexdigest()
    return fingerprint, total_rows, schema_bytes, data


def open_run(store, dataset, *, config_fingerprint: str, stage_count: int, resuming: bool):
    """Record or verify the run identity; return ``(source_rows, materialized_input)``.

    The caller must preserve logical row identity and read order, because
    commits name source rows by offset. The input fingerprint catches common
    accidental changes; it does not prove that arbitrary payloads are unchanged.
    The checkpointed operator configuration must also remain compatible.
    """
    recorded = store.get(RUN_KEY)
    if recorded is not None and not resuming:
        raise ProtocolError("A checkpoint already exists for this job; resume it or use a new job ID")
    if recorded is None and store.list(COMMIT_DIR):
        raise ProtocolError("Checkpoint commits exist without their run identity")
    fingerprint, rows, schema, data = _scan(dataset)
    identity = {
        "version": PROTOCOL_VERSION,
        "input_fingerprint": fingerprint,
        "input_rows": rows,
        "schema_hex": schema.hex(),
        "config_fingerprint": config_fingerprint,
        "stage_count": stage_count,
    }
    if recorded is None:
        if not store.put_if_absent(RUN_KEY, json.dumps(identity, sort_keys=True).encode()):
            raise ProtocolError("Another driver created this checkpoint concurrently")
    elif json.loads(recorded) != identity:
        raise ProtocolError("Input content, read order, or operator config changed since the checkpoint")
    return rows, data
