"""Ordered, addressable source snapshot for local recovery units."""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_juicer.core.executor.unit_protocol import ProtocolError, _fsync_dir, _sha256_file


def _row_fingerprint_update(digest, table):
    for row in table.to_pylist():
        encoded = json.dumps(row, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def fingerprint_dataset(dataset):
    """Hash content and read order independently of Ray block boundaries."""
    digest = hashlib.sha256()
    rows = 0
    schema = None
    for table in dataset.iter_batches(batch_format="pyarrow", batch_size=512):
        if schema is None:
            schema = table.schema
            digest.update(schema.serialize().to_pybytes())
        elif not schema.equals(table.schema, check_metadata=True):
            raise ProtocolError("Input schema changed between batches")
        _row_fingerprint_update(digest, table)
        rows += table.num_rows
    if schema is None:
        schema = dataset.schema()
        if not isinstance(schema, pa.Schema):
            raise ProtocolError("Empty input needs a concrete Arrow schema")
        digest.update(schema.serialize().to_pybytes())
    return digest.hexdigest(), rows


@dataclass(frozen=True)
class ParquetUnitSource:
    directory: str
    unit_size: int

    def __call__(self, start, stop):
        if start % self.unit_size or not 0 < stop - start <= self.unit_size:
            raise ProtocolError("Input unit does not match its source snapshot")
        ordinal = start // self.unit_size
        path = Path(self.directory) / f"unit-{ordinal:012d}.parquet"
        if not path.is_file():
            raise ProtocolError("Input unit snapshot file is missing")
        table = pq.read_table(path)
        if table.num_rows != stop - start:
            raise ProtocolError("Input unit snapshot row count changed")
        yield table


def open_or_create_snapshot(dataset, directory: str, unit_size: int, *, resuming: bool):
    """Return a durable snapshot manifest, validating current input on resume."""
    if unit_size < 1:
        raise ValueError("unit_size must be positive")
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if resuming:
        if not manifest_path.is_file():
            raise ProtocolError("Unit snapshot is missing on resume")
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if manifest.get("version") != 1 or manifest.get("unit_size") != unit_size:
            raise ProtocolError("Unit snapshot plan changed")
        if len(manifest.get("files", [])) != (manifest["rows"] + unit_size - 1) // unit_size:
            raise ProtocolError("Unit snapshot file count changed")
        for index, item in enumerate(manifest["files"]):
            path = root / f"unit-{index:012d}.parquet"
            if (
                item["name"] != path.name
                or not path.is_file()
                or _sha256_file(path) != item["sha256"]
                or pq.read_metadata(path).num_rows != item["rows"]
            ):
                raise ProtocolError("Unit snapshot file is missing or modified")
        current_hash, current_rows = fingerprint_dataset(dataset)
        if (current_hash, current_rows) != (manifest["fingerprint"], manifest["rows"]):
            raise ProtocolError("Input content or read order changed since the unit snapshot")
        return manifest, ParquetUnitSource(str(root), unit_size)

    if root.exists():
        raise ProtocolError("Unit snapshot already exists; resume or use a new job ID")
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = root.parent / f".{root.name}.{uuid.uuid4().hex}.tmp"
    tmp.mkdir()
    digest = hashlib.sha256()
    schema = None
    chunks = []
    chunk_rows = 0
    total_rows = 0
    files = []

    def flush():
        nonlocal chunks, chunk_rows
        if not chunks:
            return
        table = pa.concat_tables(chunks)
        path = tmp / f"unit-{len(files):012d}.parquet"
        pq.write_table(table, path)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        files.append({"name": path.name, "rows": table.num_rows, "sha256": _sha256_file(path)})
        chunks = []
        chunk_rows = 0

    try:
        for table in dataset.iter_batches(batch_format="pyarrow", batch_size=unit_size):
            if schema is None:
                schema = table.schema
                digest.update(schema.serialize().to_pybytes())
            elif not schema.equals(table.schema, check_metadata=True):
                raise ProtocolError("Input schema changed between batches")
            _row_fingerprint_update(digest, table)
            total_rows += table.num_rows
            offset = 0
            while offset < table.num_rows:
                length = min(unit_size - chunk_rows, table.num_rows - offset)
                chunks.append(table.slice(offset, length))
                chunk_rows += length
                offset += length
                if chunk_rows == unit_size:
                    flush()
        flush()
        if schema is None:
            schema = dataset.schema()
            if not isinstance(schema, pa.Schema):
                raise ProtocolError("Empty input needs a concrete Arrow schema")
            digest.update(schema.serialize().to_pybytes())
        manifest = {
            "version": 1,
            "fingerprint": digest.hexdigest(),
            "rows": total_rows,
            "unit_size": unit_size,
            "schema_hex": None if schema is None else schema.serialize().to_pybytes().hex(),
            "files": files,
        }
        with (tmp / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(tmp)
        os.replace(tmp, root)
        _fsync_dir(root.parent)
        return manifest, ParquetUnitSource(str(root), unit_size)
    except BaseException:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        raise
