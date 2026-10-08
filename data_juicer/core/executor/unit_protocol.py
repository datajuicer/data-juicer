"""Durable, local commit protocol for recovery units.

This module does not execute operators. A caller gives each unit a stable input
range, runs an attempt, writes all of that attempt's output, and commits it.
An empty output list is a valid completed result. The SQLite database and
output files must live on one local POSIX filesystem; a distributed backend
needs the same conditional-commit semantics, not a shared SQLite file.
"""

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 2


class ProtocolError(RuntimeError):
    """The requested transition would violate the recovery protocol."""


@dataclass(frozen=True)
class Unit:
    unit_id: str
    ordinal: int
    start: int
    stop: int


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    unit_id: str
    stage: int
    generation: int
    input_commit_id: Optional[str]


@dataclass(frozen=True)
class OutputFile:
    path: str
    rows: int
    sha256: str


@dataclass(frozen=True)
class PreparedOutput:
    files: Tuple[OutputFile, ...]
    schema: pa.Schema


@dataclass(frozen=True)
class Commit:
    attempt: Attempt
    files: Tuple[OutputFile, ...]
    rows: int
    schema: pa.Schema


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalUnitStore:
    """Single-host durable planner and conditional commit store.

    The caller must provide a fingerprint of the *content and order* of the
    immutable input snapshot and a fingerprint of the complete operator config.
    Those values are compared on every open; they are not inferred from paths.
    """

    def __init__(self, directory: str, run_id: str):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "units.sqlite3"
        self.run_id = run_id
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                    input_fingerprint TEXT NOT NULL, config_fingerprint TEXT NOT NULL,
                    input_rows INTEGER NOT NULL, unit_size INTEGER NOT NULL,
                    stage_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS units (
                    run_id TEXT NOT NULL, unit_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, start_row INTEGER NOT NULL,
                    stop_row INTEGER NOT NULL,
                    PRIMARY KEY (run_id, unit_id), UNIQUE (run_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS stage_state (
                    run_id TEXT NOT NULL, unit_id TEXT NOT NULL,
                    stage INTEGER NOT NULL, generation INTEGER NOT NULL,
                    committed_attempt_id TEXT,
                    PRIMARY KEY (run_id, unit_id, stage)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL, stage INTEGER NOT NULL,
                    generation INTEGER NOT NULL, input_commit_id TEXT,
                    files_json TEXT, rows INTEGER, schema_hex TEXT
                );
                """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(attempts)")}
            if "schema_hex" not in columns:
                raise ProtocolError("Checkpoint database uses an incompatible protocol version")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            else:
                db.execute("COMMIT")

    def plan(
        self, *, input_fingerprint: str, config_fingerprint: str, input_rows: int, unit_size: int, stage_count: int
    ) -> Tuple[Unit, ...]:
        if not input_fingerprint or not config_fingerprint:
            raise ValueError("Input and config fingerprints are required")
        if input_rows < 0 or unit_size < 1 or stage_count < 1:
            raise ValueError("input_rows must be non-negative; sizes must be positive")
        signature = (SCHEMA_VERSION, input_fingerprint, config_fingerprint, input_rows, unit_size, stage_count)
        with self._transaction() as db:
            prior = db.execute("SELECT * FROM runs WHERE run_id=?", (self.run_id,)).fetchone()
            if prior is None:
                db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)", (self.run_id, *signature))
                for ordinal, start in enumerate(range(0, input_rows, unit_size)):
                    stop = min(start + unit_size, input_rows)
                    unit_id = f"{ordinal:012d}"
                    db.execute("INSERT INTO units VALUES (?, ?, ?, ?, ?)", (self.run_id, unit_id, ordinal, start, stop))
            elif (
                tuple(
                    prior[key]
                    for key in (
                        "schema_version",
                        "input_fingerprint",
                        "config_fingerprint",
                        "input_rows",
                        "unit_size",
                        "stage_count",
                    )
                )
                != signature
            ):
                raise ProtocolError("Recovery plan differs from the recorded run")
            rows = db.execute(
                "SELECT unit_id, ordinal, start_row, stop_row FROM units WHERE run_id=? ORDER BY ordinal",
                (self.run_id,),
            ).fetchall()
            return tuple(Unit(r["unit_id"], r["ordinal"], r["start_row"], r["stop_row"]) for r in rows)

    def _stage_count(self, db) -> int:
        row = db.execute("SELECT stage_count FROM runs WHERE run_id=?", (self.run_id,)).fetchone()
        if row is None:
            raise ProtocolError("Plan the run before starting attempts")
        return row[0]

    def begin(self, unit_id: str, stage: int) -> Attempt:
        with self._transaction() as db:
            if not 0 <= stage < self._stage_count(db):
                raise ProtocolError("Invalid stage")
            if (
                db.execute("SELECT 1 FROM units WHERE run_id=? AND unit_id=?", (self.run_id, unit_id)).fetchone()
                is None
            ):
                raise ProtocolError("Unknown unit")
            input_commit_id = None
            if stage:
                upstream = db.execute(
                    "SELECT committed_attempt_id FROM stage_state WHERE run_id=? AND unit_id=? AND stage=?",
                    (self.run_id, unit_id, stage - 1),
                ).fetchone()
                if upstream is None or upstream[0] is None:
                    raise ProtocolError("Upstream stage has not committed")
                input_commit_id = upstream[0]
            state = db.execute(
                "SELECT generation, committed_attempt_id FROM stage_state WHERE run_id=? AND unit_id=? AND stage=?",
                (self.run_id, unit_id, stage),
            ).fetchone()
            if state is not None and state["committed_attempt_id"] is not None:
                raise ProtocolError("Stage already committed")
            generation = 1 if state is None else state["generation"] + 1
            attempt_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO stage_state VALUES (?, ?, ?, ?, NULL) "
                "ON CONFLICT(run_id, unit_id, stage) DO UPDATE SET generation=excluded.generation",
                (self.run_id, unit_id, stage, generation),
            )
            db.execute(
                "INSERT INTO attempts (attempt_id, run_id, unit_id, stage, generation, input_commit_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (attempt_id, self.run_id, unit_id, stage, generation, input_commit_id),
            )
            return Attempt(attempt_id, unit_id, stage, generation, input_commit_id)

    def write_output(
        self, attempt: Attempt, batches: Iterable[pa.Table], *, empty_schema: Optional[pa.Schema] = None
    ) -> PreparedOutput:
        """Write complete blocks and record the schema even for zero rows."""
        directory = self.root / "attempts" / attempt.attempt_id
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        schema = empty_schema
        for index, batch in enumerate(batches):
            if not isinstance(batch, pa.Table):
                raise TypeError("Attempt output must contain PyArrow tables")
            if schema is None:
                schema = batch.schema
            elif not schema.equals(batch.schema, check_metadata=True):
                raise ProtocolError("Attempt output schema differs between blocks")
            if batch.num_rows == 0:
                continue
            name = f"part-{index:08d}.parquet"
            final = directory / name
            tmp = directory / f".{name}.{uuid.uuid4().hex}.tmp"
            try:
                pq.write_table(batch, tmp)
                with tmp.open("rb") as f:
                    os.fsync(f.fileno())
                os.link(tmp, final)  # Fails if this attempt already wrote the part.
            finally:
                tmp.unlink(missing_ok=True)
            digest = _sha256_file(final)
            files.append(OutputFile(str(final.relative_to(self.root)), batch.num_rows, digest))
        _fsync_dir(directory)
        _fsync_dir(directory.parent)
        if schema is None:
            raise ProtocolError("Empty output requires an explicit schema")
        return PreparedOutput(tuple(files), schema)

    def commit(self, attempt: Attempt, output: PreparedOutput) -> Commit:
        """Atomically accept a complete attempt, or reject a stale one."""
        if not isinstance(output.schema, pa.Schema):
            raise ProtocolError("Attempt output requires a PyArrow schema")
        files = output.files
        rows = 0
        seen = set()
        expected_dir = (self.root / "attempts" / attempt.attempt_id).resolve()
        for item in files:
            path = (self.root / item.path).resolve()
            if path.parent != expected_dir or item.path in seen or item.rows < 1:
                raise ProtocolError("Invalid attempt output file")
            seen.add(item.path)
            if not path.is_file() or _sha256_file(path) != item.sha256:
                raise ProtocolError("Missing or modified attempt output file")
            metadata = pq.read_metadata(path)
            if metadata.num_rows != item.rows:
                raise ProtocolError("Attempt output row count mismatch")
            if not pq.read_schema(path).equals(output.schema, check_metadata=True):
                raise ProtocolError("Attempt output schema mismatch")
            rows += item.rows
        with self._transaction() as db:
            recorded = db.execute(
                "SELECT * FROM attempts WHERE attempt_id=? AND run_id=?", (attempt.attempt_id, self.run_id)
            ).fetchone()
            if recorded is None or tuple(
                recorded[key] for key in ("unit_id", "stage", "generation", "input_commit_id")
            ) != (attempt.unit_id, attempt.stage, attempt.generation, attempt.input_commit_id):
                raise ProtocolError("Unknown or modified attempt")
            state = db.execute(
                "SELECT generation, committed_attempt_id FROM stage_state WHERE run_id=? AND unit_id=? AND stage=?",
                (self.run_id, attempt.unit_id, attempt.stage),
            ).fetchone()
            if state["generation"] != attempt.generation or state["committed_attempt_id"] is not None:
                raise ProtocolError("Attempt is stale or stage already committed")
            if attempt.stage:
                upstream = db.execute(
                    "SELECT committed_attempt_id FROM stage_state WHERE run_id=? AND unit_id=? AND stage=?",
                    (self.run_id, attempt.unit_id, attempt.stage - 1),
                ).fetchone()
                if upstream is None or upstream[0] != attempt.input_commit_id:
                    raise ProtocolError("Upstream commit changed")
            db.execute(
                "UPDATE attempts SET files_json=?, rows=?, schema_hex=? WHERE attempt_id=?",
                (
                    json.dumps([item.__dict__ for item in files]),
                    rows,
                    output.schema.serialize().to_pybytes().hex(),
                    attempt.attempt_id,
                ),
            )
            db.execute(
                "UPDATE stage_state SET committed_attempt_id=? " "WHERE run_id=? AND unit_id=? AND stage=?",
                (attempt.attempt_id, self.run_id, attempt.unit_id, attempt.stage),
            )
        return Commit(attempt, files, rows, output.schema)

    def get_commit(self, unit_id: str, stage: int) -> Optional[Commit]:
        """Read a committed result and fail if its durable output is damaged."""
        with self._connect() as db:
            row = db.execute(
                "SELECT a.* FROM stage_state s JOIN attempts a ON a.attempt_id=s.committed_attempt_id "
                "WHERE s.run_id=? AND s.unit_id=? AND s.stage=?",
                (self.run_id, unit_id, stage),
            ).fetchone()
            if row is None:
                return None
            attempt = Attempt(
                row["attempt_id"], row["unit_id"], row["stage"], row["generation"], row["input_commit_id"]
            )
            files = tuple(OutputFile(**item) for item in json.loads(row["files_json"]))
            if not row["schema_hex"]:
                raise ProtocolError("Committed output schema is missing")
            schema = pa.ipc.read_schema(pa.BufferReader(bytes.fromhex(row["schema_hex"])))
            if sum(item.rows for item in files) != row["rows"]:
                raise ProtocolError("Committed output row count mismatch")
            expected_dir = (self.root / "attempts" / attempt.attempt_id).resolve()
            for item in files:
                path = (self.root / item.path).resolve()
                if (
                    path.parent != expected_dir
                    or not path.is_file()
                    or _sha256_file(path) != item.sha256
                    or pq.read_metadata(path).num_rows != item.rows
                    or not pq.read_schema(path).equals(schema, check_metadata=True)
                ):
                    raise ProtocolError("Committed output file is missing or modified")
            return Commit(attempt, files, row["rows"], schema)

    def pending(self, stage: int) -> Tuple[Unit, ...]:
        """Units ready for this stage but not yet durably completed."""
        with self._connect() as db:
            if not 0 <= stage < self._stage_count(db):
                raise ProtocolError("Invalid stage")
            sql = (
                "SELECT u.unit_id, u.ordinal, u.start_row, u.stop_row FROM units u "
                "LEFT JOIN stage_state current ON current.run_id=u.run_id "
                "AND current.unit_id=u.unit_id AND current.stage=? "
            )
            params = [stage]
            if stage:
                sql += (
                    "JOIN stage_state previous ON previous.run_id=u.run_id "
                    "AND previous.unit_id=u.unit_id AND previous.stage=? "
                    "AND previous.committed_attempt_id IS NOT NULL "
                )
                params.append(stage - 1)
            sql += "WHERE u.run_id=? AND current.committed_attempt_id IS NULL ORDER BY u.ordinal"
            params.append(self.run_id)
            rows = db.execute(sql, params).fetchall()
            return tuple(Unit(r["unit_id"], r["ordinal"], r["start_row"], r["stop_row"]) for r in rows)
