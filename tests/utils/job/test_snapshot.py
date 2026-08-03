import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from data_juicer.utils.job.snapshot import (
    JobSnapshot,
    OperationStatus,
    PartitionStatus,
    ProcessingSnapshotAnalyzer,
    ProcessingStatus,
    create_snapshot,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class PartitionStatusTest(DataJuicerTestCaseBase):

    def test_post_init_defaults(self):
        ps = PartitionStatus(partition_id=0, status=ProcessingStatus.NOT_STARTED)
        self.assertEqual(ps.completed_operations, [])
        self.assertEqual(ps.failed_operations, [])
        self.assertEqual(ps.checkpointed_operations, [])

    def test_post_init_preserves_provided(self):
        ps = PartitionStatus(
            partition_id=1,
            status=ProcessingStatus.COMPLETED,
            completed_operations=['op1'],
        )
        self.assertEqual(ps.completed_operations, ['op1'])

    def test_mutable_defaults_independent(self):
        ps1 = PartitionStatus(partition_id=0, status=ProcessingStatus.NOT_STARTED)
        ps2 = PartitionStatus(partition_id=1, status=ProcessingStatus.NOT_STARTED)
        ps1.completed_operations.append('op_a')
        self.assertEqual(ps2.completed_operations, [])


class OperationStatusTest(DataJuicerTestCaseBase):

    def test_basic_creation(self):
        op = OperationStatus(
            operation_name='filter_op',
            operation_idx=0,
            status=ProcessingStatus.COMPLETED,
            start_time=100.0,
            end_time=110.0,
            duration=10.0,
            input_rows=1000,
            output_rows=900,
        )
        self.assertEqual(op.operation_name, 'filter_op')
        self.assertEqual(op.duration, 10.0)
        self.assertEqual(op.output_rows, 900)


class JobSnapshotTest(DataJuicerTestCaseBase):

    def test_defaults(self):
        snap = JobSnapshot(job_id='test_job')
        self.assertEqual(snap.total_partitions, 0)
        self.assertEqual(snap.overall_status, ProcessingStatus.NOT_STARTED)
        self.assertFalse(snap.resumable)


class AnalyzeEventsTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_empty_events(self):
        partitions, operations = self.analyzer.analyze_events([])
        self.assertEqual(partitions, {})
        self.assertEqual(operations, {})

    def test_partition_lifecycle(self):
        events = [
            {'event_type': 'partition_creation_start', 'partition_id': 0, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {'sample_count': 100}},
            {'event_type': 'partition_start', 'partition_id': 0, 'timestamp': 3.0},
            {'event_type': 'partition_complete', 'partition_id': 0, 'timestamp': 10.0},
        ]
        partitions, _ = self.analyzer.analyze_events(events)
        p = partitions[0]
        self.assertEqual(p.status, ProcessingStatus.COMPLETED)
        self.assertEqual(p.sample_count, 100)
        self.assertAlmostEqual(p.creation_start_time, 1.0)
        self.assertAlmostEqual(p.processing_end_time, 10.0)

    def test_partition_failed(self):
        events = [
            {'event_type': 'partition_creation_start', 'partition_id': 0, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'partition_start', 'partition_id': 0, 'timestamp': 3.0},
            {'event_type': 'partition_failed', 'partition_id': 0,
             'timestamp': 5.0, 'error_message': 'OOM'},
        ]
        partitions, _ = self.analyzer.analyze_events(events)
        self.assertEqual(partitions[0].status, ProcessingStatus.FAILED)
        self.assertEqual(partitions[0].error_message, 'OOM')

    def test_op_lifecycle(self):
        events = [
            {'event_type': 'partition_creation_start', 'partition_id': 0, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'op_start', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'filter_a', 'timestamp': 3.0},
            {'event_type': 'op_complete', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'filter_a', 'timestamp': 5.0,
             'metadata': {'input_rows': 100, 'output_rows': 80}},
        ]
        partitions, operations = self.analyzer.analyze_events(events)
        key = 'p0_op0_filter_a'
        self.assertIn(key, operations)
        op = operations[key]
        self.assertEqual(op.status, ProcessingStatus.COMPLETED)
        self.assertAlmostEqual(op.duration, 2.0)
        self.assertEqual(op.input_rows, 100)
        self.assertEqual(op.output_rows, 80)
        self.assertIn('filter_a', partitions[0].completed_operations)

    def test_op_failed(self):
        events = [
            {'event_type': 'partition_creation_start', 'partition_id': 0, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'op_start', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'bad_op', 'timestamp': 3.0},
            {'event_type': 'op_failed', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'bad_op', 'timestamp': 4.0,
             'error_message': 'crash'},
        ]
        _, operations = self.analyzer.analyze_events(events)
        self.assertEqual(operations['p0_op0_bad_op'].status, ProcessingStatus.FAILED)
        self.assertEqual(operations['p0_op0_bad_op'].error_message, 'crash')

    def test_checkpoint_save(self):
        events = [
            {'event_type': 'partition_creation_start', 'partition_id': 0, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'op_start', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'op_a', 'timestamp': 3.0},
            {'event_type': 'checkpoint_save', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'op_a', 'timestamp': 4.0},
        ]
        partitions, operations = self.analyzer.analyze_events(events)
        self.assertEqual(operations['p0_op0_op_a'].status,
                         ProcessingStatus.CHECKPOINTED)
        self.assertIn('op_a', partitions[0].checkpointed_operations)

    def test_multiple_partitions(self):
        events = [
            {'event_type': 'partition_creation_start', 'partition_id': 0, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'partition_creation_start', 'partition_id': 1, 'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 1,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'partition_start', 'partition_id': 0, 'timestamp': 3.0},
            {'event_type': 'partition_complete', 'partition_id': 0, 'timestamp': 5.0},
            {'event_type': 'partition_start', 'partition_id': 1, 'timestamp': 3.0},
            {'event_type': 'partition_failed', 'partition_id': 1,
             'timestamp': 4.0, 'error_message': 'err'},
        ]
        partitions, _ = self.analyzer.analyze_events(events)
        self.assertEqual(len(partitions), 2)
        self.assertEqual(partitions[0].status, ProcessingStatus.COMPLETED)
        self.assertEqual(partitions[1].status, ProcessingStatus.FAILED)


class DetermineOverallStatusTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_empty(self):
        self.assertEqual(
            self.analyzer.determine_overall_status({}, {}),
            ProcessingStatus.NOT_STARTED)

    def test_all_completed(self):
        parts = {
            0: PartitionStatus(0, ProcessingStatus.COMPLETED),
            1: PartitionStatus(1, ProcessingStatus.COMPLETED),
        }
        self.assertEqual(
            self.analyzer.determine_overall_status(parts, {}),
            ProcessingStatus.COMPLETED)

    def test_all_failed(self):
        parts = {0: PartitionStatus(0, ProcessingStatus.FAILED)}
        self.assertEqual(
            self.analyzer.determine_overall_status(parts, {}),
            ProcessingStatus.FAILED)

    def test_mixed_in_progress(self):
        parts = {
            0: PartitionStatus(0, ProcessingStatus.COMPLETED),
            1: PartitionStatus(1, ProcessingStatus.IN_PROGRESS),
        }
        self.assertEqual(
            self.analyzer.determine_overall_status(parts, {}),
            ProcessingStatus.IN_PROGRESS)

    def test_some_failed_some_completed(self):
        parts = {
            0: PartitionStatus(0, ProcessingStatus.COMPLETED),
            1: PartitionStatus(1, ProcessingStatus.FAILED),
        }
        self.assertEqual(
            self.analyzer.determine_overall_status(parts, {}),
            ProcessingStatus.IN_PROGRESS)

    def test_all_not_started(self):
        parts = {0: PartitionStatus(0, ProcessingStatus.NOT_STARTED)}
        self.assertEqual(
            self.analyzer.determine_overall_status(parts, {}),
            ProcessingStatus.NOT_STARTED)


class CalculateStatisticsTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_counts(self):
        parts = {
            0: PartitionStatus(0, ProcessingStatus.COMPLETED),
            1: PartitionStatus(1, ProcessingStatus.FAILED),
            2: PartitionStatus(2, ProcessingStatus.IN_PROGRESS),
        }
        ops = {
            'k1': OperationStatus('a', 0, ProcessingStatus.COMPLETED),
            'k2': OperationStatus('b', 1, ProcessingStatus.FAILED),
            'k3': OperationStatus('c', 2, ProcessingStatus.CHECKPOINTED),
            'k4': OperationStatus('d', 3, ProcessingStatus.IN_PROGRESS),
        }
        stats = self.analyzer.calculate_statistics(parts, ops)
        self.assertEqual(stats['total_partitions'], 3)
        self.assertEqual(stats['completed_partitions'], 1)
        self.assertEqual(stats['failed_partitions'], 1)
        self.assertEqual(stats['in_progress_partitions'], 1)
        self.assertEqual(stats['total_operations'], 4)
        self.assertEqual(stats['completed_operations'], 1)
        self.assertEqual(stats['failed_operations'], 1)
        self.assertEqual(stats['checkpointed_operations'], 1)


class FormatDurationTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_none(self):
        self.assertIsNone(self.analyzer._format_duration(None))

    def test_seconds_only(self):
        self.assertEqual(self.analyzer._format_duration(45.0), '45s')

    def test_minutes_seconds(self):
        self.assertEqual(self.analyzer._format_duration(65.0), '1m 5s')

    def test_hours_minutes_seconds(self):
        self.assertEqual(self.analyzer._format_duration(3661.0), '1h 1m 1s')

    def test_zero(self):
        self.assertEqual(self.analyzer._format_duration(0.0), '0s')


class PartitionProgressTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_completed(self):
        p = PartitionStatus(0, ProcessingStatus.COMPLETED)
        self.assertEqual(self.analyzer._calculate_partition_progress(p), 100.0)

    def test_failed(self):
        p = PartitionStatus(0, ProcessingStatus.FAILED)
        self.assertEqual(self.analyzer._calculate_partition_progress(p), 0.0)

    def test_not_started(self):
        p = PartitionStatus(0, ProcessingStatus.NOT_STARTED)
        self.assertEqual(self.analyzer._calculate_partition_progress(p), 0.0)

    def test_in_progress_no_ops(self):
        p = PartitionStatus(0, ProcessingStatus.IN_PROGRESS)
        self.assertEqual(self.analyzer._calculate_partition_progress(p), 10.0)

    def test_in_progress_with_ops(self):
        p = PartitionStatus(0, ProcessingStatus.IN_PROGRESS,
                            completed_operations=['a', 'b'])
        progress = self.analyzer._calculate_partition_progress(p)
        self.assertGreater(progress, 10.0)
        self.assertLessEqual(progress, 90.0)


class OperationProgressTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_completed(self):
        op = OperationStatus('a', 0, ProcessingStatus.COMPLETED)
        self.assertEqual(self.analyzer._calculate_operation_progress(op), 100.0)

    def test_failed(self):
        op = OperationStatus('a', 0, ProcessingStatus.FAILED)
        self.assertEqual(self.analyzer._calculate_operation_progress(op), 0.0)

    def test_checkpointed(self):
        op = OperationStatus('a', 0, ProcessingStatus.CHECKPOINTED)
        self.assertEqual(self.analyzer._calculate_operation_progress(op), 100.0)

    def test_not_started(self):
        op = OperationStatus('a', 0, ProcessingStatus.NOT_STARTED)
        self.assertEqual(self.analyzer._calculate_operation_progress(op), 0.0)


class OverallProgressTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_full_progress(self):
        snap = JobSnapshot(
            job_id='test', total_partitions=2, completed_partitions=2,
            total_operations=4, completed_operations=4,
            partition_statuses={}, operation_statuses={})
        progress = self.analyzer._calculate_overall_progress(snap)
        self.assertEqual(progress['partition_percentage'], 100.0)
        self.assertEqual(progress['operation_percentage'], 100.0)
        self.assertEqual(progress['overall_percentage'], 100.0)

    def test_half_progress(self):
        snap = JobSnapshot(
            job_id='test', total_partitions=4, completed_partitions=2,
            total_operations=8, completed_operations=4,
            partition_statuses={}, operation_statuses={})
        progress = self.analyzer._calculate_overall_progress(snap)
        self.assertEqual(progress['partition_percentage'], 50.0)
        self.assertEqual(progress['operation_percentage'], 50.0)
        self.assertEqual(progress['overall_percentage'], 50.0)

    def test_zero_total(self):
        snap = JobSnapshot(
            job_id='test', total_partitions=0, completed_partitions=0,
            total_operations=0, completed_operations=0,
            partition_statuses={}, operation_statuses={})
        pp = self.analyzer._calculate_partition_progress_percentage(snap)
        op = self.analyzer._calculate_operation_progress_percentage(snap)
        self.assertEqual(pp, 100.0)
        self.assertEqual(op, 100.0)


class CheckpointProgressTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_no_operations(self):
        snap = JobSnapshot(
            job_id='test', total_operations=0,
            checkpointed_operations=0,
            partition_statuses={}, operation_statuses={})
        result = self.analyzer._calculate_checkpoint_progress(snap)
        self.assertEqual(result['percentage'], 0.0)

    def test_with_checkpointed(self):
        snap = JobSnapshot(
            job_id='test', total_operations=4,
            checkpointed_operations=2,
            partition_statuses={},
            operation_statuses={
                'k1': OperationStatus('a', 0, ProcessingStatus.CHECKPOINTED,
                                      checkpoint_time=100.0),
                'k2': OperationStatus('b', 1, ProcessingStatus.CHECKPOINTED,
                                      checkpoint_time=200.0),
                'k3': OperationStatus('c', 2, ProcessingStatus.COMPLETED),
                'k4': OperationStatus('d', 3, ProcessingStatus.COMPLETED),
            })
        result = self.analyzer._calculate_checkpoint_progress(snap)
        self.assertEqual(result['percentage'], 50.0)
        self.assertEqual(len(result['checkpointed_operations']), 2)
        self.assertAlmostEqual(result['checkpoint_coverage'], 0.5)


class GenerateSnapshotFromFilesTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _write_events(self, events):
        path = os.path.join(self.tmpdir, 'events.jsonl')
        with open(path, 'w') as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

    def _write_job_summary(self, summary):
        path = os.path.join(self.tmpdir, 'job_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f)

    def test_generate_from_files(self):
        self._write_events([
            {'event_type': 'job_start', 'timestamp': 1.0,
             'metadata': {'checkpoint_strategy': 'per_op'}},
            {'event_type': 'partition_creation_start', 'partition_id': 0,
             'timestamp': 2.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 3.0, 'metadata': {'sample_count': 50}},
            {'event_type': 'partition_start', 'partition_id': 0,
             'timestamp': 4.0},
            {'event_type': 'op_start', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'mapper_a', 'timestamp': 4.5},
            {'event_type': 'op_complete', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'mapper_a', 'timestamp': 6.0,
             'metadata': {'input_rows': 50, 'output_rows': 50}},
            {'event_type': 'partition_complete', 'partition_id': 0,
             'timestamp': 7.0},
            {'event_type': 'job_complete', 'timestamp': 8.0},
        ])
        self._write_job_summary({
            'start_time': 1.0,
            'end_time': 8.0,
            'duration': 7.0,
            'status': 'completed',
        })

        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        snapshot = analyzer.generate_snapshot()

        self.assertEqual(snapshot.overall_status, ProcessingStatus.COMPLETED)
        self.assertEqual(snapshot.total_partitions, 1)
        self.assertEqual(snapshot.completed_partitions, 1)
        self.assertEqual(snapshot.total_operations, 1)
        self.assertEqual(snapshot.completed_operations, 1)
        self.assertAlmostEqual(snapshot.job_start_time, 1.0)
        self.assertAlmostEqual(snapshot.total_duration, 7.0)
        self.assertEqual(snapshot.checkpoint_strategy, 'per_op')

    def test_generate_resumable(self):
        self._write_events([
            {'event_type': 'partition_creation_start', 'partition_id': 0,
             'timestamp': 1.0},
            {'event_type': 'partition_creation_complete', 'partition_id': 0,
             'timestamp': 2.0, 'metadata': {}},
            {'event_type': 'op_start', 'partition_id': 0, 'operation_idx': 0,
             'operation_name': 'op_x', 'timestamp': 3.0},
            {'event_type': 'checkpoint_save', 'partition_id': 0,
             'operation_idx': 0, 'operation_name': 'op_x', 'timestamp': 4.0},
        ])

        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        snapshot = analyzer.generate_snapshot()
        self.assertTrue(snapshot.resumable)
        self.assertAlmostEqual(snapshot.last_checkpoint_time, 4.0)

    def test_create_snapshot_convenience(self):
        self._write_events([])
        snapshot = create_snapshot(self.tmpdir)
        self.assertIsInstance(snapshot, JobSnapshot)

    def test_load_events_missing_file(self):
        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        events = analyzer.load_events()
        self.assertEqual(events, [])

    def test_load_dag_missing_file(self):
        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        dag = analyzer.load_dag_plan()
        self.assertEqual(dag, {})

    def test_load_job_summary_missing(self):
        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        summary = analyzer.load_job_summary()
        self.assertEqual(summary, {})

    def test_finds_latest_events_file(self):
        old_path = os.path.join(self.tmpdir, 'events_20250101.jsonl')
        new_path = os.path.join(self.tmpdir, 'events_20250102.jsonl')
        with open(old_path, 'w') as f:
            f.write(json.dumps({'event_type': 'old'}) + '\n')
        time.sleep(0.05)
        with open(new_path, 'w') as f:
            f.write(json.dumps({'event_type': 'new'}) + '\n')

        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        events = analyzer.load_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], 'new')


class ToJsonDictTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def test_basic_structure(self):
        events_path = os.path.join(self.tmpdir, 'events.jsonl')
        with open(events_path, 'w') as f:
            f.write(json.dumps({
                'event_type': 'job_start',
                'timestamp': 0.5,
                'metadata': {'checkpoint_strategy': 'per_op'}
            }) + '\n')
            f.write(json.dumps({
                'event_type': 'partition_creation_start',
                'partition_id': 0, 'timestamp': 1.0
            }) + '\n')
            f.write(json.dumps({
                'event_type': 'partition_creation_complete',
                'partition_id': 0, 'timestamp': 2.0, 'metadata': {}
            }) + '\n')
            f.write(json.dumps({
                'event_type': 'partition_start',
                'partition_id': 0, 'timestamp': 3.0
            }) + '\n')
            f.write(json.dumps({
                'event_type': 'partition_complete',
                'partition_id': 0, 'timestamp': 5.0
            }) + '\n')
            f.write(json.dumps({
                'event_type': 'job_complete',
                'timestamp': 6.0
            }) + '\n')

        analyzer = ProcessingSnapshotAnalyzer(self.tmpdir)
        snapshot = analyzer.generate_snapshot()
        result = analyzer.to_json_dict(snapshot)

        self.assertIn('job_info', result)
        self.assertIn('overall_status', result)
        self.assertIn('progress_summary', result)
        self.assertIn('partition_progress', result)
        self.assertIn('checkpointing', result)
        self.assertEqual(result['overall_status'], 'completed')
        self.assertIn('0', result['partition_progress'])

        summary = result['progress_summary']
        self.assertEqual(summary['total_partitions'], 1)
        self.assertEqual(summary['completed_partitions'], 1)
        self.assertEqual(summary['failed_partitions'], 0)

        p0 = result['partition_progress']['0']
        self.assertEqual(p0['status'], 'completed')
        self.assertIsNotNone(p0['creation_start_time'])

        self.assertIn('timing', result)
        self.assertIsNotNone(result['timing']['start_time'])

        json.dumps(result)


if __name__ == '__main__':
    unittest.main()
