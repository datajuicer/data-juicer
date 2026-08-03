import json
import os
import tempfile
import unittest

from data_juicer.utils.job.monitor import JobProgressMonitor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JobProgressMonitorInitTest(DataJuicerTestCaseBase):

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            monitor = JobProgressMonitor('myjob', base_dir=tmpdir)
            self.assertEqual(monitor.job_id, 'myjob')

    def test_init_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            JobProgressMonitor('nonexistent',
                               base_dir='/tmp/does_not_exist_xyz')


class GetProgressDataTest(DataJuicerTestCaseBase):

    def _setup_job_dir(self, tmpdir):
        job_dir = os.path.join(tmpdir, 'testjob')
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, 'job_summary.json'), 'w') as f:
            json.dump({'status': 'completed', 'start_time': 100.0}, f)
        events = [
            {'event_type': 'partition_start', 'partition_id': 0,
             'timestamp': 1.0},
            {'event_type': 'partition_complete', 'partition_id': 0,
             'timestamp': 5.0},
        ]
        with open(os.path.join(job_dir, 'events.jsonl'), 'w') as f:
            for e in events:
                f.write(json.dumps(e) + '\n')
        return job_dir

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_job_dir(tmpdir)
            monitor = JobProgressMonitor('testjob', base_dir=tmpdir)
            data = monitor.get_progress_data()
            self.assertIn('job_id', data)
            self.assertIn('job_summary', data)
            self.assertIn('overall_progress', data)
            self.assertIn('partition_status', data)
            self.assertEqual(data['job_id'], 'testjob')

    def test_progress_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_job_dir(tmpdir)
            monitor = JobProgressMonitor('testjob', base_dir=tmpdir)
            data = monitor.get_progress_data()
            progress = data['overall_progress']
            self.assertEqual(progress['total_partitions'], 1)
            self.assertEqual(progress['completed_partitions'], 1)
            self.assertEqual(progress['progress_percentage'], 100.0)


class DisplayProgressTest(DataJuicerTestCaseBase):

    def test_display_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'testjob')
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, 'job_summary.json'), 'w') as f:
                json.dump({'status': 'running'}, f)
            with open(os.path.join(job_dir, 'events.jsonl'), 'w') as f:
                f.write(json.dumps({
                    'event_type': 'partition_start',
                    'partition_id': 0, 'timestamp': 1.0
                }) + '\n')

            monitor = JobProgressMonitor('testjob', base_dir=tmpdir)
            import io
            import sys
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                monitor.display_progress(detailed=False)
            finally:
                sys.stdout = old_stdout
            output = captured.getvalue()
            self.assertIn('testjob', output)
            self.assertIn('PARTITION STATUS', output)

    def test_display_detailed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'testjob')
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, 'job_summary.json'), 'w') as f:
                json.dump({'status': 'completed', 'start_time': 100.0,
                           'duration': 5.0}, f)
            events = [
                {'event_type': 'partition_start', 'partition_id': 0,
                 'timestamp': 1.0},
                {'event_type': 'op_start', 'partition_id': 0,
                 'operation_name': 'filter_a', 'operation_idx': 0,
                 'timestamp': 1.5},
                {'event_type': 'op_complete', 'partition_id': 0,
                 'operation_name': 'filter_a', 'operation_idx': 0,
                 'timestamp': 2.0, 'duration': 0.5, 'input_rows': 100,
                 'output_rows': 90, 'performance_metrics': {
                     'throughput': 200, 'reduction_ratio': 0.1}},
                {'event_type': 'checkpoint_save', 'partition_id': 0,
                 'operation_name': 'filter_a', 'operation_idx': 0,
                 'checkpoint_path': '/tmp/ckpt', 'timestamp': 2.5},
                {'event_type': 'partition_complete', 'partition_id': 0,
                 'timestamp': 3.0},
            ]
            with open(os.path.join(job_dir, 'events.jsonl'), 'w') as f:
                for e in events:
                    f.write(json.dumps(e) + '\n')

            monitor = JobProgressMonitor('testjob', base_dir=tmpdir)
            import io
            import sys
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                monitor.display_progress(detailed=True)
            finally:
                sys.stdout = old_stdout
            output = captured.getvalue()
            self.assertIn('OPERATION DETAILS', output)
            self.assertIn('filter_a', output)


if __name__ == '__main__':
    unittest.main()
