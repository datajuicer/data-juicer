import json
import os
import subprocess
import tempfile
import time
import unittest

import psutil

from data_juicer.utils.job.stopper import JobStopper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JobStopperInitTest(DataJuicerTestCaseBase):

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            stopper = JobStopper('myjob', base_dir=tmpdir)
            self.assertEqual(stopper.job_id, 'myjob')

    def test_init_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            JobStopper('nonexistent', base_dir='/tmp/does_not_exist_xyz')


class CleanupJobResourcesTest(DataJuicerTestCaseBase):

    def test_updates_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            original = {'status': 'running', 'start_time': 100.0}
            with open(os.path.join(job_dir, 'job_summary.json'), 'w') as f:
                json.dump(original, f)

            stopper = JobStopper('myjob', base_dir=tmpdir)
            stopper.cleanup_job_resources()

            with open(os.path.join(job_dir, 'job_summary.json')) as f:
                updated = json.load(f)
            self.assertEqual(updated['status'], 'stopped')
            self.assertEqual(updated['stop_reason'], 'manual_stop')
            self.assertIn('stop_time', updated)
            self.assertEqual(updated['start_time'], 100.0)

    def test_no_summary_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            stopper = JobStopper('myjob', base_dir=tmpdir)
            stopper.cleanup_job_resources()


class TerminateProcessGracefullyTest(DataJuicerTestCaseBase):

    def test_terminate_returns_true_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            stopper = JobStopper('myjob', base_dir=tmpdir)

            proc = subprocess.Popen(['sleep', '1000'])
            ps_proc = psutil.Process(proc.pid)
            try:
                result = stopper.terminate_process_gracefully(ps_proc, timeout=1)
            finally:
                try:
                    proc.kill()
                    proc.wait()
                except OSError:
                    pass

            self.assertTrue(result)
            self.assertFalse(psutil.pid_exists(proc.pid))


class StopJobNoProcessesTest(DataJuicerTestCaseBase):

    def test_no_events_no_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, 'job_summary.json'), 'w') as f:
                json.dump({'status': 'running'}, f)
            with open(os.path.join(job_dir, 'events.jsonl'), 'w') as f:
                f.write(json.dumps({'event_type': 'job_start',
                                    'timestamp': 1.0}) + '\n')

            stopper = JobStopper('myjob', base_dir=tmpdir)
            result = stopper.stop_job()
            self.assertEqual(result['processes_found'], 0)
            self.assertEqual(result['threads_found'], 0)
            self.assertFalse(result['success'])

            with open(os.path.join(job_dir, 'job_summary.json')) as f:
                updated = json.load(f)
            self.assertEqual(updated['status'], 'stopped')

    def test_stale_pids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'myjob')
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, 'job_summary.json'), 'w') as f:
                json.dump({'status': 'running'}, f)
            events = [
                {'event_type': 'op_start', 'process_id': 999999999,
                 'timestamp': 1.0},
            ]
            with open(os.path.join(job_dir, 'events.jsonl'), 'w') as f:
                for e in events:
                    f.write(json.dumps(e) + '\n')

            stopper = JobStopper('myjob', base_dir=tmpdir)
            result = stopper.stop_job()
            self.assertEqual(result['processes_found'], 1)
            self.assertEqual(result['processes_terminated'], 0)


if __name__ == '__main__':
    unittest.main()
