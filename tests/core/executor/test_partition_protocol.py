"""Ray-free checks of checkpoint storage and the lineage commit log."""

import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa

from data_juicer.core.executor.partition_protocol import (
    COMMIT_DIR,
    SEQ_COLUMN,
    SOURCE_SEQ,
    Commit,
    CommitLog,
    IntervalSet,
    LocalStore,
    ProtocolError,
    S3Store,
    commit_key,
    key_ranges,
    open_store,
    read_tables,
    select_ranges,
    with_keys,
    write_tables,
)


def _persist(store, commit):
    assert store.put_if_absent(commit_key(commit.stage, commit.seq), commit.to_bytes())


def _prepare(store, log, stage, consumed, rows):
    files = write_tables(store, [pa.table({"v": list(range(rows))})]) if rows else ()
    commit, new = log.prepare(stage, consumed, files)
    if new:
        _persist(store, commit)
    return commit, new


class PartitionProtocolTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="dj-recovery-test-")
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_key_ranges_compress_unsorted_keys(self):
        self.assertEqual(key_ranges([3, 3, -1, 3, -1], [2, 0, 5, 1, 4]), ((-1, 4, 6), (3, 0, 3)))
        self.assertEqual(key_ranges([], []), ())
        with self.assertRaisesRegex(ProtocolError, "same upstream row twice"):
            key_ranges([1, 1], [0, 0])

    def test_interval_set_merges_and_reports_gaps(self):
        ranges = IntervalSet()
        ranges.add(4, 6)
        ranges.add(0, 2)
        ranges.add(2, 4)
        self.assertEqual((ranges.starts, ranges.stops), ([0], [6]))
        ranges.add(8, 9)
        self.assertTrue(ranges.overlaps(5, 7) and ranges.overlaps(7, 9) and (not ranges.overlaps(6, 8)))
        self.assertEqual(ranges.gaps(10), [(6, 8), (9, 10)])

    def test_local_store_creates_only_once_and_hides_temporaries(self):
        store = LocalStore(str(self.tmp_path / "ckpt"))
        self.assertTrue(store.put_if_absent("commits/a.json", b"first"))
        self.assertFalse(store.put_if_absent("commits/a.json", b"second"))
        self.assertEqual(store.get("commits/a.json"), b"first")
        (self.tmp_path / "ckpt" / "commits" / ".a.json.x.tmp").write_bytes(b"partial")
        self.assertEqual(store.list("commits"), ["commits/a.json"])
        self.assertEqual(store.list("missing"), [])
        self.assertIs(store.get("missing.json"), None)

    def test_open_store_accepts_posix_and_s3_only(self):
        self.assertIsInstance(open_store(str(self.tmp_path)), LocalStore)
        self.assertEqual(open_store(f"file://{self.tmp_path}").root, str(self.tmp_path))
        s3 = open_store("s3://bucket/prefix/run")
        self.assertTrue(isinstance(s3, S3Store) and (s3.bucket, s3.prefix) == ("bucket", "prefix/run"))
        for location in ("gs://bucket/x", "hdfs://cluster/x", "oss://bucket/x"):
            with self.assertRaisesRegex(ValueError, "not supported"):
                open_store(location)

    def test_files_round_trip_and_detect_tampering(self):
        store = LocalStore(str(self.tmp_path))
        tables = [pa.table({"a": [1, 2]}), pa.table({"a": [3]}), pa.table({"b": ["x"]}), pa.table({"a": []})]
        files = write_tables(store, tables)
        self.assertEqual([item.rows for item in files], [3, 1])
        self.assertEqual(
            [table.to_pylist() for table in read_tables(store, files)], [[{"a": 1}, {"a": 2}, {"a": 3}], [{"b": "x"}]]
        )
        with open(os.path.join(store.root, files[0].key), "ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ProtocolError, "missing or modified"):
            list(read_tables(store, files))

    def test_keys_select_requested_rows(self):
        table = with_keys(pa.table({"v": list("abcde")}), 7)
        self.assertEqual(table[SEQ_COLUMN].to_pylist(), [7] * 5)
        self.assertEqual(select_ranges(table, [(0, 1), (3, 5)])["v"].to_pylist(), ["a", "d", "e"])

    def test_failed_table_write_removes_only_this_calls_files(self):
        store = LocalStore(str(self.tmp_path))
        retained = write_tables(store, [pa.table({"committed": [1]})])
        put = store.put
        calls = []

        def fail_second_write(key, data):
            put(key, data)
            calls.append(key)
            if len(calls) == 2:
                raise OSError("injected data write failure")

        with patch.object(store, "put", side_effect=fail_second_write):
            with self.assertRaisesRegex(OSError, "injected data write failure"):
                write_tables(store, [pa.table({"a": [1]}), pa.table({"b": [2]})])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(store.get(key) is None for key in calls))
        self.assertEqual(list(read_tables(store, retained))[0].to_pylist(), [{"committed": 1}])

    def test_exact_duplicate_returns_winner_and_partial_overlap_fails(self):
        store = LocalStore(str(self.tmp_path))
        log = CommitLog(store, stage_count=2, source_rows=10)
        first, new = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 4)], 3)
        self.assertTrue(new and first.seq == 0)
        retry, new = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 4)], 2)
        self.assertTrue(not new and retry == first)
        with self.assertRaisesRegex(ProtocolError, "already consumed"):
            log.prepare(0, [(SOURCE_SEQ, 3, 6)], ())
        with self.assertRaisesRegex(ProtocolError, "unknown upstream"):
            log.prepare(1, [(99, 0, 1)], ())
        with self.assertRaisesRegex(ProtocolError, "malformed or unknown"):
            log.prepare(1, [(first.seq, 0, 4)], ())
        with self.assertRaisesRegex(ProtocolError, "first stage"):
            log.prepare(0, [(first.seq, 0, 1)], ())

    def test_reloaded_log_plans_pending_rows_and_verifies_delivery(self):
        store = LocalStore(str(self.tmp_path))
        log = CommitLog(store, stage_count=2, source_rows=10)
        first, _ = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 4)], 3)
        _prepare(store, log, 0, [(SOURCE_SEQ, 4, 7)], 0)
        downstream, _ = _prepare(store, log, 1, [(first.seq, 0, 2)], 5)
        # An attempt that never wrote its manifest leaves no trace in the reloaded log.
        log.prepare(0, [(SOURCE_SEQ, 7, 10)], ())

        reloaded = CommitLog(store, stage_count=2, source_rows=10)
        plan = reloaded.plan()
        self.assertEqual(plan.source, ((7, 10),))
        self.assertEqual([(item.commit.seq, item.ranges) for item in plan.stages[1]], [(first.seq, ((2, 3),))])
        self.assertEqual(plan.final, (downstream,))
        self.assertEqual(reloaded.next_seq, 3)
        with self.assertRaisesRegex(ProtocolError, "never consumed by stage 0"):
            reloaded.verify()

        rest, _ = _prepare(store, reloaded, 0, [(SOURCE_SEQ, 7, 10)], 1)
        tail, _ = _prepare(store, reloaded, 1, [(first.seq, 2, 3), (rest.seq, 0, 1)], 1)
        with self.assertRaisesRegex(ProtocolError, "never delivered downstream"):
            reloaded.verify()
        reloaded.deliver([(downstream.seq, 0, 5)])
        reloaded.deliver([(downstream.seq, 0, 5)])
        with self.assertRaisesRegex(ProtocolError, "already consumed"):
            reloaded.deliver([(downstream.seq, 4, 5), (tail.seq, 0, 1)])
        reloaded.deliver([(tail.seq, 0, 1)])
        self.assertEqual(reloaded.verify(), {0: (), 1: ()})

    def test_log_rejects_foreign_or_duplicate_manifests(self):
        store = LocalStore(str(self.tmp_path))
        commit = Commit(0, 0, ((SOURCE_SEQ, 0, 2),), ())
        store.put(f"{COMMIT_DIR}/0000-0000000000000005.json", commit.to_bytes())
        with self.assertRaisesRegex(ProtocolError, "does not belong"):
            CommitLog(store, stage_count=1, source_rows=2)

        other = LocalStore(str(self.tmp_path / "other"))
        _persist(other, commit)
        _persist(other, Commit(0, 1, ((SOURCE_SEQ, 0, 2),), ()))
        with self.assertRaisesRegex(ProtocolError, "duplicates"):
            CommitLog(other, stage_count=1, source_rows=2)

    def test_stats_files_are_returned_by_stage(self):
        store = LocalStore(str(self.tmp_path))
        log = CommitLog(store, stage_count=1, source_rows=2)
        stats = write_tables(store, [pa.table({"s": [1, 2]})], kind="stats")
        commit, _ = log.prepare(0, [(SOURCE_SEQ, 0, 2)], (), stats)
        _persist(store, commit)
        self.assertEqual(CommitLog(store, stage_count=1, source_rows=2).verify(), {0: stats})

    def test_find_and_summary_include_empty_output_commits(self):
        store = LocalStore(str(self.tmp_path))
        log = CommitLog(store, stage_count=2, source_rows=5)
        first, _ = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 3)], 2)
        _prepare(store, log, 0, [(SOURCE_SEQ, 3, 5)], 0)
        last, _ = _prepare(store, log, 1, [(first.seq, 0, 2)], 4)
        self.assertEqual(log.find(0, [[SOURCE_SEQ, 0, 3]]), first)
        self.assertIs(log.find(0, [(SOURCE_SEQ, 0, 2)]), None)
        self.assertEqual(log.find(1, [(first.seq, 0, 2)]), last)
        self.assertEqual(log.summary(), {0: (2, 5, 2), 1: (1, 2, 4)})
        restored = CommitLog(store, stage_count=2, source_rows=5)
        self.assertEqual(restored.find(0, first.consumed), first)
        self.assertEqual(restored.summary(), log.summary())

    def test_manifest_reads_overlap_and_record_in_dependency_order(self):
        store = LocalStore(str(self.tmp_path))
        log = CommitLog(store, stage_count=2, source_rows=2)
        first, _ = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 2)], 2)
        _prepare(store, log, 1, [(first.seq, 0, 2)], 1)
        barrier = threading.Barrier(2)

        class ConcurrentStore:
            def list(self, prefix):
                return list(reversed(store.list(prefix)))

            def get(self, key):
                barrier.wait(timeout=5)
                return store.get(key)

        restored = CommitLog(ConcurrentStore(), stage_count=2, source_rows=2)
        self.assertEqual(restored.summary(), {0: (1, 2, 2), 1: (1, 2, 1)})


if __name__ == "__main__":
    unittest.main()
