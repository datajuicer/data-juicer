"""Crash and output-contract tests for the local recovery-unit protocol."""

import sqlite3

import pyarrow as pa
import pytest

from data_juicer.core.executor.unit_protocol import LocalUnitStore, ProtocolError
from data_juicer.core.executor.unit_runner import UnitStage, run_local_units


def test_old_checkpoint_database_is_rejected_before_running(tmp_path):
    with sqlite3.connect(tmp_path / "units.sqlite3") as db:
        db.execute("CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, files_json TEXT, rows INTEGER)")
    with pytest.raises(ProtocolError, match="incompatible protocol version"):
        LocalUnitStore(str(tmp_path), "job-1")


def _store(tmp_path, *, rows=5, size=2, stages=2):
    store = LocalUnitStore(str(tmp_path), "job-1")
    units = store.plan(
        input_fingerprint="ordered-input-hash",
        config_fingerprint="operator-chain-hash",
        input_rows=rows,
        unit_size=size,
        stage_count=stages,
    )
    return store, units


def test_plan_is_stable_and_changed_provenance_fails_closed(tmp_path):
    store, units = _store(tmp_path)
    assert [(u.unit_id, u.start, u.stop) for u in units] == [
        ("000000000000", 0, 2),
        ("000000000001", 2, 4),
        ("000000000002", 4, 5),
    ]
    assert _store(tmp_path)[1] == units
    with pytest.raises(ProtocolError, match="differs"):
        store.plan(
            input_fingerprint="changed",
            config_fingerprint="operator-chain-hash",
            input_rows=5,
            unit_size=2,
            stage_count=2,
        )
    with pytest.raises(ProtocolError, match="differs"):
        store.plan(
            input_fingerprint="ordered-input-hash",
            config_fingerprint="changed",
            input_rows=5,
            unit_size=2,
            stage_count=2,
        )


def test_zero_output_and_many_outputs_are_complete_commits(tmp_path):
    store, (unit,) = _store(tmp_path, rows=2, size=2)
    first = store.begin(unit.unit_id, 0)
    first_output = store.write_output(first, [], empty_schema=pa.schema([("copy", pa.int64())]))
    assert first_output.files == ()
    assert store.commit(first, first_output).rows == 0
    assert store.pending(0) == ()

    second = store.begin(unit.unit_id, 1)
    assert second.input_commit_id == first.attempt_id
    # The unit emits more rows than it received, across two physical files.
    files = store.write_output(
        second,
        [
            pa.table({"copy": [0, 1]}),
            pa.table({"copy": [2, 3, 4]}),
        ],
    )
    assert store.commit(second, files).rows == 5
    reopened, _ = _store(tmp_path, rows=2, size=2)
    restored = reopened.get_commit(unit.unit_id, 1)
    assert restored is not None and restored.rows == 5
    assert len(restored.files) == 2
    assert reopened.pending(1) == ()


def test_uncommitted_output_is_reprocessed_and_late_attempt_is_fenced(tmp_path):
    store, (unit,) = _store(tmp_path, rows=2, size=2, stages=1)
    old = store.begin(unit.unit_id, 0)
    old_files = store.write_output(old, [pa.table({"value": ["old"]})])
    # Simulate a crash after durable output, before the metadata commit.
    restarted, _ = _store(tmp_path, rows=2, size=2, stages=1)
    assert restarted.get_commit(unit.unit_id, 0) is None
    assert restarted.pending(0) == (unit,)
    current = restarted.begin(unit.unit_id, 0)
    current_files = restarted.write_output(current, [pa.table({"value": ["new"]})])
    with pytest.raises(ProtocolError, match="stale"):
        restarted.commit(old, old_files)
    accepted = restarted.commit(current, current_files)
    assert restarted.get_commit(unit.unit_id, 0) == accepted
    with pytest.raises(ProtocolError, match="already committed"):
        restarted.begin(unit.unit_id, 0)
    with pytest.raises(ProtocolError, match="already committed"):
        restarted.commit(old, old_files)


def test_downstream_requires_upstream_and_can_commit_empty_result(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1)
    assert store.pending(1) == ()
    with pytest.raises(ProtocolError, match="Upstream"):
        store.begin(unit.unit_id, 1)
    first = store.begin(unit.unit_id, 0)
    store.commit(first, store.write_output(first, [pa.table({"value": [1, 2, 3]})]))
    assert store.pending(1) == (unit,)
    second = store.begin(unit.unit_id, 1)
    # A Filter can drop every output of the previous stage.
    store.commit(second, store.write_output(second, [], empty_schema=pa.schema([("value", pa.int64())])))
    assert store.get_commit(unit.unit_id, 1).rows == 0
    assert store.pending(1) == ()


def test_missing_or_modified_output_cannot_be_committed(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    files = store.write_output(attempt, [pa.table({"value": [1]})])
    (tmp_path / files.files[0].path).unlink()
    with pytest.raises(ProtocolError, match="Missing or modified"):
        store.commit(attempt, files)
    assert store.get_commit(unit.unit_id, 0) is None


def test_forged_attempt_cannot_commit_another_units_output(tmp_path):
    store, units = _store(tmp_path, rows=3, size=2, stages=1)
    first = store.begin(units[0].unit_id, 0)
    second = store.begin(units[1].unit_id, 0)
    files = store.write_output(first, [pa.table({"value": [1]})])
    with pytest.raises(ProtocolError, match="Invalid attempt output file"):
        store.commit(second, files)


def test_committed_file_damage_fails_closed_on_recovery(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    files = store.write_output(attempt, [pa.table({"value": [1]})])
    store.commit(attempt, files)
    (tmp_path / files.files[0].path).unlink()
    reopened, _ = _store(tmp_path, rows=1, size=1, stages=1)
    with pytest.raises(ProtocolError, match="Committed output file"):
        reopened.get_commit(unit.unit_id, 0)


def test_runner_rejects_short_snapshot_read_before_commit(tmp_path):
    store, units = _store(tmp_path, rows=2, size=2, stages=1)

    def short_read(start, stop):
        yield pa.table({"value": [start]})

    def identity(table):
        yield table

    with pytest.raises(ProtocolError, match="input row count"):
        run_local_units(store, units, short_read, [identity])
    assert store.get_commit(units[0].unit_id, 0) is None


def test_runner_restarts_from_committed_units_with_expansion_and_drop_all(tmp_path):
    store, units = _store(tmp_path, rows=4, size=2)
    reads = []
    expands = []
    filtered = []
    fail_once = {"value": True}

    def source(start, stop):
        reads.append((start, stop))
        yield pa.table({"value": list(range(start, stop))})

    def expand(table):
        expands.append(table.column("value")[0].as_py())
        yield table
        yield table

    def keep_first_unit(table):
        first = table.column("value")[0].as_py()
        filtered.append(first)
        if first == 2 and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("worker died")
        kept = table.filter(pa.array([value < 2 for value in table["value"].to_pylist()]))
        if kept.num_rows:
            yield kept

    with pytest.raises(RuntimeError, match="worker died"):
        run_local_units(store, units, source, [expand, UnitStage(keep_first_unit, lambda schema: schema)])
    assert len(reads) == len(expands) == 2
    assert store.get_commit(units[0].unit_id, 1).rows == 4
    assert store.get_commit(units[1].unit_id, 1) is None

    reopened, same_units = _store(tmp_path, rows=4, size=2)
    blocks = run_local_units(reopened, same_units, source, [expand, UnitStage(keep_first_unit, lambda schema: schema)])
    assert len(reads) == len(expands) == 2  # Committed Mapper units were not repeated.
    assert filtered == [0, 0, 2, 2, 2]  # Only the unfinished Filter unit repeats.
    assert [value for block in blocks for value in block["value"].to_pylist()] == [0, 1, 0, 1]
    assert reopened.get_commit(units[1].unit_id, 1).rows == 0


def test_runner_restores_schema_when_every_final_unit_is_empty(tmp_path):
    store, units = _store(tmp_path, rows=2, size=1, stages=1)

    def source(start, stop):
        yield pa.table({"value": list(range(start, stop))})

    def drop_all(table):
        return ()

    blocks = run_local_units(store, units, source, [UnitStage(drop_all, lambda schema: schema)])
    assert len(blocks) == 1
    assert blocks[0].num_rows == 0
    assert blocks[0].schema == pa.schema([("value", pa.int64())])
    reopened, same_units = _store(tmp_path, rows=2, size=1, stages=1)
    assert run_local_units(reopened, same_units, source, [UnitStage(drop_all, lambda schema: schema)]) == blocks
