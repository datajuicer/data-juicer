import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from unittest.mock import MagicMock, patch

from data_juicer.utils.job.monitor import JobProgressMonitor, main, show_job_progress
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

from . import isolate_job_logging


def setUpModule():
    isolate_job_logging()


class JobProgressMonitorTest(DataJuicerTestCaseBase):

    def _monitor(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, "job"))
        return JobProgressMonitor("job", base_dir=tmpdir)

    @TEST_TAG("standalone")
    def test_init_and_missing_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = self._monitor(tmpdir)
            self.assertEqual(monitor.job_id, "job")
            self.assertEqual(str(monitor.work_dir), os.path.join(tmpdir, "job"))
            with self.assertRaises(FileNotFoundError):
                JobProgressMonitor("missing", base_dir=tmpdir)

    @TEST_TAG("standalone")
    def test_get_progress_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = self._monitor(tmpdir)
            monitor.job_utils.load_job_summary = MagicMock(return_value={"status": "running"})
            monitor.job_utils.load_dataset_mapping = MagicMock(return_value={"partition_size": 10})
            monitor.job_utils.get_partition_status = MagicMock(return_value={})
            monitor.job_utils.calculate_overall_progress = MagicMock(return_value={"job_status": "running"})

            data = monitor.get_progress_data()

        self.assertEqual(data["job_id"], "job")
        self.assertEqual(data["job_summary"]["status"], "running")
        self.assertEqual(data["dataset_mapping"]["partition_size"], 10)
        self.assertEqual(data["partition_status"], {})
        self.assertEqual(data["overall_progress"], {"job_status": "running"})
        monitor.job_utils.calculate_overall_progress.assert_called_once_with()

    def test_progress_from_event_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = self._monitor(tmpdir)
            job_dir = monitor.work_dir
            (job_dir / "job_summary.json").write_text(json.dumps({"status": "completed", "start_time": 100}))
            events = [
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 5},
            ]
            (job_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
            data = monitor.get_progress_data()
            progress = data["overall_progress"]
            self.assertEqual(progress["total_partitions"], 1)
            self.assertEqual(progress["completed_partitions"], 1)
            self.assertEqual(progress["progress_percentage"], 100.0)
            self.assertEqual(data["partition_status"][0]["status"], "completed")
            output = io.StringIO()
            with redirect_stdout(output):
                monitor.display_progress(detailed=True)
            self.assertIn("100.0% (1/1 partitions)", output.getvalue())

    @TEST_TAG("standalone")
    def test_display_progress_detailed_covers_all_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = self._monitor(tmpdir)
            monitor.job_utils.load_job_summary = MagicMock(return_value={"start_time": 1, "duration": 12.5})
            monitor.job_utils.load_dataset_mapping = MagicMock(
                return_value={
                    "original_dataset_path": "input.jsonl",
                    "original_dataset_size": 100,
                    "partition_size": 50,
                }
            )
            monitor.job_utils.calculate_overall_progress = MagicMock(
                return_value={
                    "job_status": "running",
                    "progress_percentage": 50.0,
                    "completed_partitions": 1,
                    "total_partitions": 2,
                    "processing_partitions": 1,
                    "failed_partitions": 0,
                    "processed_samples": 50,
                    "total_samples": 100,
                    "estimated_remaining_seconds": 120,
                }
            )
            monitor.job_utils.get_partition_status = MagicMock(
                return_value={
                    0: {
                        "status": "completed",
                        "sample_count": 50,
                        "current_op": None,
                        "completed_ops": [
                            {
                                "name": "filter",
                                "idx": 0,
                                "duration": 1.5,
                                "throughput": 20,
                                "reduction_ratio": 0.25,
                            }
                        ],
                        "checkpoints": [
                            {
                                "operation_name": "filter",
                                "operation_idx": 0,
                                "timestamp": 0,
                            }
                        ],
                    },
                    1: {
                        "status": "processing",
                        "sample_count": 50,
                        "current_op": {"name": "mapper", "idx": 1},
                        "completed_ops": [],
                        "checkpoints": [],
                    },
                }
            )
            output = io.StringIO()
            with redirect_stdout(output):
                monitor.display_progress(detailed=True)

        text = output.getvalue()
        for expected in (
            "input.jsonl",
            datetime.fromtimestamp(1).strftime("%Y-%m-%d %H:%M:%S"),
            "12.5 seconds",
            "2.0 minutes",
            "filter",
            "mapper",
            "Total Checkpoints: 1",
        ):
            self.assertIn(expected, text)

    @TEST_TAG("standalone")
    def test_display_progress_empty_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = self._monitor(tmpdir)
            monitor.job_utils.load_job_summary = MagicMock(return_value=None)
            monitor.job_utils.load_dataset_mapping = MagicMock(return_value={})
            monitor.job_utils.get_partition_status = MagicMock(return_value={})
            monitor.job_utils.calculate_overall_progress = MagicMock(
                return_value={
                    "job_status": "unknown",
                    "progress_percentage": 0,
                    "completed_partitions": 0,
                    "total_partitions": 0,
                    "processing_partitions": 0,
                    "failed_partitions": 0,
                    "processed_samples": 0,
                    "total_samples": 0,
                    "estimated_remaining_seconds": 0,
                }
            )
            output = io.StringIO()
            with redirect_stdout(output):
                monitor.display_progress()

        self.assertIn("Dataset: Unknown", output.getvalue())
        self.assertIn("Total Checkpoints: 0", output.getvalue())


class MonitorConvenienceAndCliTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_show_job_progress_delegates_and_returns_data(self):
        expected = {"job_id": "abc"}
        with patch("data_juicer.utils.job.monitor.JobProgressMonitor") as monitor_cls:
            monitor_cls.return_value.get_progress_data.return_value = expected
            result = show_job_progress("abc", "/tmp/base", detailed=True)

        monitor_cls.assert_called_once_with("abc", "/tmp/base")
        monitor_cls.return_value.display_progress.assert_called_once_with(True)
        self.assertIs(result, expected)

    @TEST_TAG("standalone")
    def test_main_once(self):
        with (
            patch("sys.argv", ["monitor", "abc", "--base-dir", "/tmp/base", "--detailed"]),
            patch("data_juicer.utils.job.monitor.JobProgressMonitor") as monitor_cls,
        ):
            main()

        monitor_cls.assert_called_once_with("abc", "/tmp/base")
        monitor_cls.return_value.display_progress.assert_called_once_with(True)

    @TEST_TAG("standalone")
    def test_main_watch_stops_on_keyboard_interrupt(self):
        with (
            patch("sys.argv", ["monitor", "abc", "--watch", "--interval", "3"]),
            patch("data_juicer.utils.job.monitor.JobProgressMonitor") as monitor_cls,
            patch("data_juicer.utils.job.monitor.os.system") as clear,
            patch("data_juicer.utils.job.monitor.time.sleep", side_effect=KeyboardInterrupt) as sleep,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                main()

        clear.assert_called_once()
        monitor_cls.return_value.display_progress.assert_called_once_with(False)
        sleep.assert_called_once_with(3)
        self.assertIn("Stopped watching", output.getvalue())

    @TEST_TAG("standalone")
    def test_main_reports_known_and_unexpected_errors(self):
        for error, message in (
            (FileNotFoundError("missing"), "Error: missing"),
            (RuntimeError("boom"), "Unexpected error: boom"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch("sys.argv", ["monitor", "abc"]),
                patch("data_juicer.utils.job.monitor.JobProgressMonitor", side_effect=error),
            ):
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 1)
                self.assertIn(message, output.getvalue())


if __name__ == "__main__":
    unittest.main()
