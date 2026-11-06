"""Durable commit log for dynamically formed checkpoint shards.

Rows stream between operators through Ray Data. Each checkpointed operator
batch records which upstream rows it consumed and which immutable files hold
its complete output. A commit is a manifest created only if absent, so the log
needs neither a database nor a fixed partition plan.

Every row carries a lineage key ``(seq, idx)``: ``seq`` names the commit that
produced it (``SOURCE_SEQ`` for input rows) and ``idx`` its position in that
commit's output. Consumed keys are kept as half-open index ranges.
"""

import hashlib
import json
import os
import time
import uuid
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROTOCOL_VERSION = 1
SOURCE_SEQ = -1
SEQ_COLUMN = "__dj_ckpt_seq__"
IDX_COLUMN = "__dj_ckpt_idx__"
KEY_COLUMNS = (SEQ_COLUMN, IDX_COLUMN)
RUN_KEY = "run.json"
COMMIT_DIR = "commits"
DATA_DIR = "data"
EXPORT_DIR = "exports"

Ranges = Tuple[Tuple[int, int, int], ...]


class ProtocolError(RuntimeError):
    """The requested transition would violate the checkpoint protocol."""


def _fsync_dir(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LocalStore:
    """Checkpoint storage on a local or shared POSIX filesystem.

    Conditional creation relies on ``link(2)`` failing for an existing name,
    so the filesystem must support hard links.
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def _write(self, key: str, data: bytes, exclusive: bool) -> bool:
        path = Path(self.root, key)
        created = [parent for parent in path.parents if not parent.exists()]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if exclusive:
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    return False
            else:
                os.replace(tmp, path)
            _fsync_dir(path.parent)
            for directory in created:
                _fsync_dir(directory.parent)
            return True
        finally:
            tmp.unlink(missing_ok=True)

    def put(self, key: str, data: bytes) -> None:
        self._write(key, data, exclusive=False)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        return self._write(key, data, exclusive=True)

    def get(self, key: str) -> Optional[bytes]:
        try:
            with open(Path(self.root, key), "rb") as stream:
                return stream.read()
        except FileNotFoundError:
            return None

    def delete(self, key: str) -> None:
        Path(self.root, key).unlink(missing_ok=True)

    def failed_write_is_unpublished(self, key: str) -> bool:
        """A completed synchronous POSIX call cannot publish its file later."""
        return self.get(key) is None

    def list(self, directory: str) -> List[str]:
        try:
            names = os.listdir(Path(self.root, directory))
        except FileNotFoundError:
            return []
        return [f"{directory}/{name}" for name in names if not name.startswith(".")]

    def __repr__(self):
        return f"LocalStore({self.root!r})"


class S3Store:
    """Checkpoint storage on S3-compatible object storage.

    Conditional creation uses ``PutObject`` with ``If-None-Match: *``, which
    AWS S3 and MinIO support. Credentials and ``AWS_ENDPOINT_URL`` come from
    the standard boto3 configuration, which every Ray worker must share.
    """

    def __init__(self, uri: str):
        parsed = urlparse(uri)
        if not parsed.netloc:
            raise ValueError(f"S3 checkpoint location needs a bucket: {uri}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self._client = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_client"] = None
        return state

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def failed_write_is_unpublished(self, key: str) -> bool:
        """An unsuccessful remote response does not prove nonpublication."""
        return False

    def put_if_absent(self, key: str, data: bytes) -> bool:
        from botocore.exceptions import ClientError

        for attempt in range(6):
            try:
                self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, IfNoneMatch="*")
                return True
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code")
                status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code == "PreconditionFailed" or status == 412:
                    return False
                if code != "ConditionalRequestConflict" and status != 409:
                    raise
                time.sleep(0.05 * 2**attempt)
        raise ProtocolError(f"Conditional write of {key} kept conflicting")

    def get(self, key: str) -> Optional[bytes]:
        from botocore.exceptions import ClientError

        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def list(self, directory: str) -> List[str]:
        prefix = self._key(directory) + "/"
        strip = len(self.prefix) + 1 if self.prefix else 0
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"][strip:] for item in page.get("Contents", ()))
        return keys

    def __repr__(self):
        return f"S3Store('s3://{self.bucket}/{self.prefix}')"


def is_supported_location(location: str) -> bool:
    return urlparse(location).scheme.lower() in ("", "file", "s3")


def open_store(location: str):
    """Open checkpoint storage for a POSIX path or an ``s3://`` URI."""
    parsed = urlparse(location)
    scheme = parsed.scheme.lower()
    if scheme == "s3":
        return S3Store(location)
    if scheme == "file":
        return LocalStore(parsed.path)
    if not scheme:
        return LocalStore(location)
    raise ValueError(
        f"Checkpoint storage scheme '{scheme}://' is not supported; "
        "use a shared POSIX path or an s3:// URI (AWS S3 or MinIO)"
    )


def key_ranges(seqs, idxs) -> Ranges:
    """Compress lineage keys into sorted ``(seq, start, stop)`` runs."""
    seqs = np.asarray(seqs, dtype=np.int64)
    idxs = np.asarray(idxs, dtype=np.int64)
    if not len(seqs):
        return ()
    order = np.lexsort((idxs, seqs))
    seqs, idxs = seqs[order], idxs[order]
    same_seq = np.diff(seqs) == 0
    if np.any(same_seq & (np.diff(idxs) == 0)):
        raise ProtocolError("A batch contains the same upstream row twice")
    breaks = np.flatnonzero(~same_seq | (np.diff(idxs) != 1)) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(seqs)]))
    return tuple((int(seqs[a]), int(idxs[a]), int(idxs[b - 1]) + 1) for a, b in zip(starts, stops))


class IntervalSet:
    """Disjoint half-open integer ranges."""

    def __init__(self):
        self.starts: List[int] = []
        self.stops: List[int] = []

    def overlaps(self, start: int, stop: int) -> bool:
        index = bisect_right(self.starts, start) - 1
        if index >= 0 and self.stops[index] > start:
            return True
        return index + 1 < len(self.starts) and self.starts[index + 1] < stop

    def add(self, start: int, stop: int) -> None:
        index = bisect_right(self.starts, start)
        left = index > 0 and self.stops[index - 1] == start
        right = index < len(self.starts) and self.starts[index] == stop
        if left and right:
            self.stops[index - 1] = self.stops.pop(index)
            self.starts.pop(index)
        elif left:
            self.stops[index - 1] = stop
        elif right:
            self.starts[index] = start
        else:
            self.starts.insert(index, start)
            self.stops.insert(index, stop)

    def gaps(self, total: int) -> List[Tuple[int, int]]:
        result, position = [], 0
        for start, stop in zip(self.starts, self.stops):
            if start > position:
                result.append((position, start))
            position = max(position, stop)
        if position < total:
            result.append((position, total))
        return result


@dataclass(frozen=True)
class OutputFile:
    key: str
    rows: int
    sha256: str


@dataclass(frozen=True)
class Commit:
    stage: int
    seq: int
    consumed: Ranges
    files: Tuple[OutputFile, ...]
    stats_files: Tuple[OutputFile, ...] = ()

    @property
    def rows(self) -> int:
        return sum(item.rows for item in self.files)

    def to_bytes(self) -> bytes:
        payload = {
            "version": PROTOCOL_VERSION,
            "stage": self.stage,
            "seq": self.seq,
            "consumed": [list(item) for item in self.consumed],
            "files": [item.__dict__ for item in self.files],
            "stats_files": [item.__dict__ for item in self.stats_files],
        }
        return json.dumps(payload, sort_keys=True).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Commit":
        payload = json.loads(data)
        if payload.get("version") != PROTOCOL_VERSION:
            raise ProtocolError(f"Checkpoint commit uses an incompatible protocol version; expected {PROTOCOL_VERSION}")
        return cls(
            payload["stage"],
            payload["seq"],
            tuple(tuple(item) for item in payload["consumed"]),
            tuple(OutputFile(**item) for item in payload["files"]),
            tuple(OutputFile(**item) for item in payload["stats_files"]),
        )


@dataclass(frozen=True)
class PendingCommit:
    commit: Commit
    ranges: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class RecoveryPlan:
    source: Tuple[Tuple[int, int], ...]
    stages: Dict[int, Tuple[PendingCommit, ...]]
    final: Tuple[Commit, ...]


def commit_key(stage: int, seq: int) -> str:
    return f"{COMMIT_DIR}/{stage:04d}-{seq:016d}.json"


def open_export_attempt(store) -> Tuple[str, Tuple[str, ...]]:
    """Register a new export filename prefix; return it with earlier prefixes, oldest first."""
    earlier = tuple(json.loads(store.get(key))["prefix"] for key in sorted(store.list(EXPORT_DIR)))
    prefix = f"dj{len(earlier):04d}-{uuid.uuid4().hex[:12]}"
    if not store.put_if_absent(f"{EXPORT_DIR}/{len(earlier):08d}.json", json.dumps({"prefix": prefix}).encode()):
        raise ProtocolError("Another driver started an export of this job")
    return prefix, earlier


def write_tables(store, tables: Iterable[pa.Table], kind: str = "part") -> Tuple[OutputFile, ...]:
    """Write consecutive same-schema tables into new immutable files."""
    directory = f"{DATA_DIR}/{uuid.uuid4().hex}"
    nonempty = (table for table in tables if table.num_rows)
    files = []
    owned = []
    try:
        for index, (_, group) in enumerate(groupby(nonempty, key=lambda table: table.schema.serialize().to_pybytes())):
            table = pa.concat_tables(list(group))
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            data = sink.getvalue().to_pybytes()
            key = f"{directory}/{kind}-{index:05d}.parquet"
            owned.append(key)
            store.put(key, data)
            files.append(OutputFile(key, table.num_rows, hashlib.sha256(data).hexdigest()))
    except BaseException:
        # No manifest can reference these files: they have not left this call.
        discard_files(store, owned)
        raise
    return tuple(files)


def discard_files(store, keys) -> None:
    """Best-effort removal of explicitly owned, proven-unpublished files only."""
    from loguru import logger

    for key in keys:
        try:
            store.delete(key)
        except Exception as error:
            logger.warning(f"Could not remove uncommitted checkpoint file {key}: {error}")


def discard_failed_commit(store, commit: Commit) -> None:
    """Reclaim owned files only when the backend can prove nonpublication."""
    from loguru import logger

    try:
        if store.failed_write_is_unpublished(commit_key(commit.stage, commit.seq)):
            discard_files(store, (item.key for item in (*commit.files, *commit.stats_files)))
    except Exception as error:
        logger.warning(f"Could not check failed checkpoint publication: {error}")


def read_tables(store, files: Sequence[OutputFile]) -> Iterable[pa.Table]:
    for item in files:
        data = store.get(item.key)
        if data is None or hashlib.sha256(data).hexdigest() != item.sha256:
            raise ProtocolError(f"Committed checkpoint file {item.key} is missing or modified")
        table = pq.read_table(pa.BufferReader(data))
        if table.num_rows != item.rows:
            raise ProtocolError(f"Committed checkpoint file {item.key} has an unexpected row count")
        yield table


def concat_outputs(tables: Sequence[pa.Table]) -> pa.Table:
    if len({table.schema for table in tables}) == 1:
        return pa.concat_tables(tables)
    return pa.concat_tables(tables, promote_options="permissive")


def with_keys(table: pa.Table, seq: int) -> pa.Table:
    rows = table.num_rows
    return table.append_column(SEQ_COLUMN, pa.array(np.full(rows, seq, dtype=np.int64))).append_column(
        IDX_COLUMN, pa.array(np.arange(rows, dtype=np.int64))
    )


def select_ranges(table: pa.Table, ranges: Sequence[Tuple[int, int]]) -> pa.Table:
    """Keep rows whose ``IDX_COLUMN`` lies in the given ranges."""
    idx = table[IDX_COLUMN].to_numpy()
    mask = np.zeros(len(idx), dtype=bool)
    for start, stop in ranges:
        mask |= (idx >= start) & (idx < stop)
    return table if mask.all() else table.filter(pa.array(mask))


def _canonical(consumed) -> Ranges:
    return tuple(tuple(int(value) for value in item) for item in consumed)


class CommitLog:
    """In-memory view of the durable commit log; not thread-safe.

    Level ``k < stage_count`` is the input of operator stage ``k``; level
    ``stage_count`` tracks rows delivered downstream in the current run.
    """

    def __init__(self, store, stage_count: int, source_rows: int):
        if stage_count < 1 or source_rows < 0:
            raise ValueError("stage_count must be positive and source_rows non-negative")
        self.stage_count = stage_count
        self.source_rows = source_rows
        self.commits: Dict[int, Commit] = {}
        self.consumed: Dict[Tuple[int, int], IntervalSet] = {}
        self.digests: Dict[str, Optional[int]] = {}
        self.next_seq = 0
        # Listing initializes the S3 client before concurrent reads start.
        keys = sorted(store.list(COMMIT_DIR))
        with ThreadPoolExecutor(max_workers=min(32, len(keys)) or 1) as pool:
            for key, data in zip(keys, pool.map(store.get, keys)):
                commit = Commit.from_bytes(data)
                if key != commit_key(commit.stage, commit.seq) or not 0 <= commit.stage < stage_count:
                    raise ProtocolError(f"Checkpoint commit {key} does not belong to this run")
                digest, duplicate = self._claim(commit.stage, commit.consumed)
                if duplicate:
                    raise ProtocolError(f"Checkpoint commit {key} duplicates an earlier commit")
                self._record(digest, commit)

    def _upstream_rows(self, level: int, seq: int) -> int:
        if level == 0:
            if seq != SOURCE_SEQ:
                raise ProtocolError("The first stage can only consume source rows")
            return self.source_rows
        upstream = self.commits.get(seq)
        if upstream is None or upstream.stage != level - 1:
            raise ProtocolError(f"Stage {level} consumed rows of an unknown upstream commit {seq}")
        return upstream.rows

    @staticmethod
    def _digest(level: int, consumed: Ranges) -> str:
        return hashlib.sha256(json.dumps([level, consumed]).encode()).hexdigest()

    def find(self, stage: int, consumed) -> Optional[Commit]:
        seq = self.digests.get(self._digest(stage, _canonical(consumed)))
        return self.commits[seq] if seq is not None else None

    def summary(self) -> Dict[int, Tuple[int, int, int]]:
        totals = {}
        for commit in self.commits.values():
            count, consumed, produced = totals.get(commit.stage, (0, 0, 0))
            totals[commit.stage] = (
                count + 1,
                consumed + sum(stop - start for _, start, stop in commit.consumed),
                produced + commit.rows,
            )
        return totals

    def _claim(self, level: int, consumed: Ranges) -> Tuple[str, bool]:
        """Reserve consumed keys; return ``(digest, is_exact_duplicate)``."""
        if not consumed:
            raise ProtocolError("A commit must consume at least one row")
        digest = self._digest(level, consumed)
        if digest in self.digests:
            return digest, True
        previous = None
        for item in consumed:
            seq, start, stop = item
            if not 0 <= start < stop <= self._upstream_rows(level, seq) or (previous and item < previous):
                raise ProtocolError("A commit consumed malformed or unknown upstream rows")
            if previous and previous[0] == seq and start < previous[2]:
                raise ProtocolError("A commit consumed the same upstream row twice")
            existing = self.consumed.get((level, seq))
            if existing is not None and existing.overlaps(start, stop):
                raise ProtocolError(
                    f"Stage {level} consumed upstream rows {seq}:[{start}, {stop}) that another commit already "
                    "consumed; Ray re-batched a retried input"
                )
            previous = item
        for seq, start, stop in consumed:
            self.consumed.setdefault((level, seq), IntervalSet()).add(start, stop)
        return digest, False

    def _record(self, digest: str, commit: Commit) -> None:
        self.digests[digest] = commit.seq
        self.commits[commit.seq] = commit
        self.next_seq = max(self.next_seq, commit.seq + 1)

    def prepare(self, stage: int, consumed, files, stats_files=()) -> Tuple[Commit, bool]:
        """Accept a commit in memory; return ``(commit, is_new)``.

        An exact duplicate returns the earlier commit so a retried batch
        replays the output its downstream consumers may already have seen.
        """
        if not 0 <= stage < self.stage_count:
            raise ProtocolError(f"Invalid checkpoint stage {stage}")
        consumed = _canonical(consumed)
        digest, duplicate = self._claim(stage, consumed)
        if duplicate:
            return self.commits[self.digests[digest]], False
        commit = Commit(stage, self.next_seq, consumed, tuple(files), tuple(stats_files))
        self._record(digest, commit)
        return commit, True

    def deliver(self, consumed) -> None:
        digest, duplicate = self._claim(self.stage_count, _canonical(consumed))
        if not duplicate:
            self.digests[digest] = None

    def _upstreams(self, level: int) -> List[Tuple[int, int]]:
        if level == 0:
            return [(SOURCE_SEQ, self.source_rows)]
        return [(seq, commit.rows) for seq, commit in sorted(self.commits.items()) if commit.stage == level - 1]

    def _gaps(self, level: int, seq: int, rows: int) -> List[Tuple[int, int]]:
        consumed = self.consumed.get((level, seq))
        return consumed.gaps(rows) if consumed is not None else ([(0, rows)] if rows else [])

    def plan(self) -> RecoveryPlan:
        """Rows each stage still has to consume, plus the committed final output."""
        stages = {}
        for level in range(1, self.stage_count):
            pending = []
            for seq, rows in self._upstreams(level):
                gaps = self._gaps(level, seq, rows)
                if gaps:
                    pending.append(PendingCommit(self.commits[seq], tuple(gaps)))
            stages[level] = tuple(pending)
        final = tuple(self.commits[seq] for seq, rows in self._upstreams(self.stage_count) if rows)
        return RecoveryPlan(tuple(self._gaps(0, SOURCE_SEQ, self.source_rows)), stages, final)

    def verify(self) -> Dict[int, Tuple[OutputFile, ...]]:
        """Check that every row reached the final delivery; return stats files by stage."""
        for level in range(self.stage_count + 1):
            missing = sum(
                stop - start for seq, rows in self._upstreams(level) for start, stop in self._gaps(level, seq, rows)
            )
            if missing:
                what = "delivered downstream" if level == self.stage_count else f"consumed by stage {level}"
                raise ProtocolError(f"{missing} committed rows were never {what}")
        stats = {}
        for seq, commit in sorted(self.commits.items()):
            stats.setdefault(commit.stage, []).extend(commit.stats_files)
        return {stage: tuple(files) for stage, files in stats.items()}
