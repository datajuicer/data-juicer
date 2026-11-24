import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from data_juicer.utils.job.common import (
    JobUtils,
    _find_latest_events_file_in_dir,
    list_running_jobs,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

from . import isolate_job_logging


def setUpModule():
    isolate_job_logging()


class JobUtilsInitTest(DataJuicerTestCaseBase):

    def test_init_preserves_redirected_application_logging(self):
        # Use a child process because the old logging loop can hang the runner.
        script = textwrap.dedent("""
            import sys
            import tempfile
            from pathlib import Path

            from loguru import logger

            from data_juicer.utils.job.common import JobUtils
            from data_juicer.utils.logger_utils import setup_logger

            with tempfile.TemporaryDirectory() as work_dir:
                setup_logger(work_dir, filename="application.log", redirect=True)
                logger.info("before job inspection")
                for _ in range(2):
                    job = JobUtils("test_job", work_dir=work_dir)
                    assert job.get_partition_status() == {}
                    logger.info("after job inspection")
                    sys.stderr.write("redirected output still works\\n")
                logger.complete()
                content = (Path(work_dir) / "application.log").read_text()
                assert "before job inspection" in content
                assert content.count("after job inspection") == 2
                assert content.count("redirected output still works") == 2
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_init_with_work_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils("test_job", work_dir=tmpdir)
            self.assertEqual(ju.job_id, "test_job")
            self.assertEqual(ju.work_dir, Path(tmpdir))

    def test_init_with_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, "myjob")
            os.makedirs(job_dir)
            ju = JobUtils("myjob", base_dir=tmpdir)
            self.assertEqual(ju.work_dir, Path(tmpdir) / "myjob")

    def test_init_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            JobUtils("nonexistent", work_dir="/tmp/does_not_exist_xyz_abc")


class LoadJobSummaryTest(DataJuicerTestCaseBase):

    def test_load_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {"status": "completed", "start_time": 100.0}
            with open(os.path.join(tmpdir, "job_summary.json"), "w") as f:
                json.dump(summary, f)
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_job_summary()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["start_time"], 100.0)

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_job_summary()
            self.assertIsNone(result)

    def test_load_corrupt_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "job_summary.json").write_text("not json{{{")
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_job_summary()
            self.assertIsNone(result)


class LoadDatasetMappingTest(DataJuicerTestCaseBase):

    def test_load_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "metadata")
            os.makedirs(meta_dir)
            mapping = {"partitions": [{"partition_id": 0}]}
            with open(os.path.join(meta_dir, "dataset_mapping.json"), "w") as f:
                json.dump(mapping, f)
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_dataset_mapping()
            self.assertEqual(len(result["partitions"]), 1)

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_dataset_mapping()
            self.assertEqual(result, {})

    def test_load_corrupt_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            (meta_dir / "dataset_mapping.json").write_text("bad json")
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_dataset_mapping()
            self.assertEqual(result, {})


class FindLatestEventsFileTest(DataJuicerTestCaseBase):

    def test_finds_timestamped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.path.join(tmpdir, "events_20250101.jsonl")
            new = os.path.join(tmpdir, "events_20250102.jsonl")
            with open(old, "w") as f:
                f.write("{}")
            with open(new, "w") as f:
                f.write("{}")
            os.utime(old, (100, 100))
            os.utime(new, (200, 200))
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju._find_latest_events_file()
            self.assertEqual(result.name, "events_20250102.jsonl")

    def test_fallback_to_events_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = os.path.join(tmpdir, "events.jsonl")
            with open(fallback, "w") as f:
                f.write("{}")
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju._find_latest_events_file()
            self.assertEqual(result.name, "events.jsonl")

    def test_returns_none_when_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju._find_latest_events_file()
            self.assertIsNone(result)


class FindLatestEventsFileInDirTest(DataJuicerTestCaseBase):

    def test_finds_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.jsonl")
            with open(path, "w") as f:
                f.write("{}")
            result = _find_latest_events_file_in_dir(Path(tmpdir))
            self.assertEqual(result.name, "events.jsonl")

    def test_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_latest_events_file_in_dir(Path(tmpdir))
            self.assertIsNone(result)


class LoadEventLogsTest(DataJuicerTestCaseBase):

    def test_load_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 1.0},
                {"event_type": "job_complete", "timestamp": 2.0},
            ]
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_event_logs()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["event_type"], "job_start")

    def test_load_with_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                f.write(json.dumps({"event_type": "ok"}) + "\n")
                f.write("not valid json\n")
                f.write(json.dumps({"event_type": "also_ok"}) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_event_logs()
            self.assertEqual(len(result), 2)

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils("test", work_dir=tmpdir)
            result = ju.load_event_logs()
            self.assertEqual(result, [])


class ExtractProcessThreadIdsTest(DataJuicerTestCaseBase):

    def test_extract_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "op_start", "process_id": 1234, "thread_id": 5678, "timestamp": 1.0},
                {"event_type": "op_start", "process_id": 1234, "thread_id": 9012, "timestamp": 2.0},
                {"event_type": "op_start", "process_id": 4321, "thread_id": None, "timestamp": 3.0},
            ]
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            ids = ju.extract_process_thread_ids()
            self.assertEqual(ids["process_ids"], {1234, 4321})
            self.assertEqual(ids["thread_ids"], {5678, 9012})

    def test_extract_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                f.write(json.dumps({"event_type": "job_start"}) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            ids = ju.extract_process_thread_ids()
            self.assertEqual(ids["process_ids"], set())
            self.assertEqual(ids["thread_ids"], set())


class GetPartitionStatusTest(DataJuicerTestCaseBase):

    def test_from_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1.0},
                {
                    "event_type": "op_start",
                    "partition_id": 0,
                    "operation_name": "filter_a",
                    "operation_idx": 0,
                    "timestamp": 2.0,
                },
                {
                    "event_type": "op_complete",
                    "partition_id": 0,
                    "operation_name": "filter_a",
                    "operation_idx": 0,
                    "timestamp": 3.0,
                    "metadata": {
                        "duration_seconds": 1.0,
                        "input_rows": 100,
                        "output_rows": 80,
                        "throughput_rows_per_second": 100,
                        "reduction_ratio": 0.2,
                    },
                },
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 4.0},
            ]
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            status = ju.get_partition_status()
            self.assertIn(0, status)
            self.assertEqual(status[0]["status"], "completed")
            self.assertEqual(len(status[0]["completed_ops"]), 1)
            self.assertIsNone(status[0]["current_op"])
            self.assertEqual(
                status[0]["completed_ops"][0],
                {
                    "name": "filter_a",
                    "idx": 0,
                    "duration": 1.0,
                    "input_rows": 100,
                    "output_rows": 80,
                    "throughput": 100,
                    "reduction_ratio": 0.2,
                },
            )

    def test_missing_metrics_default_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for extra in [{}, {"metadata": None}, {"metadata": {}}]:
                with self.subTest(extra=extra):
                    event = {"event_type": "op_complete", "partition_id": 0, "timestamp": 1.0, **extra}
                    (Path(tmpdir) / "events.jsonl").write_text(json.dumps(event) + "\n")
                    status = JobUtils("test", work_dir=tmpdir).get_partition_status()
                    self.assertEqual(
                        status[0]["completed_ops"][0],
                        {
                            "name": "Unknown",
                            "idx": 0,
                            "duration": 0,
                            "input_rows": 0,
                            "output_rows": 0,
                            "throughput": 0,
                            "reduction_ratio": 0,
                        },
                    )

    def test_checkpoint_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {
                    "event_type": "checkpoint_save",
                    "partition_id": 0,
                    "operation_name": "op_a",
                    "operation_idx": 0,
                    "checkpoint_path": "/tmp/ckpt",
                    "timestamp": 5.0,
                },
            ]
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            status = ju.get_partition_status()
            self.assertEqual(len(status[0]["checkpoints"]), 1)
            self.assertEqual(status[0]["checkpoints"][0]["checkpoint_path"], "/tmp/ckpt")

    def test_from_dataset_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            mapping = {"partitions": [{"partition_id": 0, "processing_status": "completed", "sample_count": 50}]}
            (meta_dir / "dataset_mapping.json").write_text(json.dumps(mapping))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.get_partition_status()
            self.assertEqual(result[0]["status"], "completed")
            self.assertEqual(result[0]["sample_count"], 50)


class CalculateOverallProgressTest(DataJuicerTestCaseBase):

    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "job_summary.json"), "w") as f:
                json.dump({"status": "completed"}, f)
            events = [
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1.0},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 2.0},
                {"event_type": "partition_start", "partition_id": 1, "timestamp": 1.0},
                {"event_type": "partition_complete", "partition_id": 1, "timestamp": 3.0},
            ]
            with open(os.path.join(tmpdir, "events.jsonl"), "w") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
            ju = JobUtils("test", work_dir=tmpdir)
            progress = ju.calculate_overall_progress()
            self.assertEqual(progress["total_partitions"], 2)
            self.assertEqual(progress["completed_partitions"], 2)
            self.assertEqual(progress["progress_percentage"], 100.0)
            self.assertEqual(progress["job_status"], "completed")

    def test_all_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 100},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 200},
                {"event_type": "partition_start", "partition_id": 1, "timestamp": 100},
                {"event_type": "partition_complete", "partition_id": 1, "timestamp": 200},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            summary = {"status": "completed", "start_time": 100}
            (Path(tmpdir) / "job_summary.json").write_text(json.dumps(summary))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.calculate_overall_progress()
            self.assertEqual(result["progress_percentage"], 100.0)
            self.assertEqual(result["completed_partitions"], 2)

    def test_empty_partitions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.calculate_overall_progress()
            self.assertEqual(result["progress_percentage"], 0)
            self.assertEqual(result["total_partitions"], 0)


class GetOperationPipelineTest(DataJuicerTestCaseBase):

    def test_parse_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = """
dataset_path: /data/test.jsonl
process:
  - language_id_score_filter:
      lang: en
  - whitespace_normalization_mapper:
      text_key: text
other_section:
  key: value
"""
            with open(os.path.join(tmpdir, "partition-checkpoint-eventlog.yaml"), "w") as f:
                f.write(config)
            ju = JobUtils("test", work_dir=tmpdir)
            ops = ju.get_operation_pipeline()
            self.assertEqual(len(ops), 2)
            self.assertEqual(ops[0]["name"], "language_id_score_filter")
            self.assertEqual(ops[1]["name"], "whitespace_normalization_mapper")

    def test_missing_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils("test", work_dir=tmpdir)
            ops = ju.get_operation_pipeline()
            self.assertEqual(ops, [])


class FindProcessesByIdsTest(DataJuicerTestCaseBase):

    def test_finds_running_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            proc = Mock(pid=1234)
            proc.is_running.return_value = True
            with (
                patch("data_juicer.utils.job.common.os.getpid", return_value=999),
                patch("data_juicer.utils.job.common.psutil.Process", return_value=proc) as process,
            ):
                result = ju.find_processes_by_ids({1234})
            self.assertEqual(result, [proc])
            process.assert_called_once_with(1234)
            proc.is_running.assert_called_once_with()

    def test_skips_processes_that_are_not_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            proc = Mock(pid=1234)
            proc.is_running.return_value = False
            with patch("data_juicer.utils.job.common.psutil.Process", return_value=proc):
                self.assertEqual(ju.find_processes_by_ids({1234}), [])

    def test_handles_process_lookup_errors(self):
        errors = [psutil.NoSuchProcess(1234), psutil.AccessDenied(1234), RuntimeError("lookup failed")]
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            for error in errors:
                with (
                    self.subTest(error=type(error).__name__),
                    patch("data_juicer.utils.job.common.psutil.Process", side_effect=error),
                ):
                    self.assertEqual(ju.find_processes_by_ids({1234}), [])

    def test_skips_current_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.find_processes_by_ids({os.getpid()})
            self.assertEqual(len(result), 0)

    def test_handles_nonexistent_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.find_processes_by_ids({99999999})
            self.assertEqual(len(result), 0)


class FindThreadsByIdsTest(DataJuicerTestCaseBase):

    def test_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.find_threads_by_ids({1, 2, 3})
            self.assertEqual(result, [])


class ListRunningJobsTest(DataJuicerTestCaseBase):

    def test_empty_base_dir(self):
        result = list_running_jobs("/tmp/nonexistent_base_dir_xyz")
        self.assertEqual(result, [])

    def test_lists_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job1"
            job_dir.mkdir()
            summary = {"status": "completed", "start_time": 1000}
            (job_dir / "job_summary.json").write_text(json.dumps(summary))
            events = [{"event_type": "op_start", "process_id": 99999999}]
            (job_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            result = list_running_jobs(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["job_id"], "job1")
            self.assertEqual(result[0]["status"], "completed")

    def test_handles_corrupt_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job_bad"
            job_dir.mkdir()
            (job_dir / "job_summary.json").write_text("not json")
            result = list_running_jobs(tmpdir)
            self.assertEqual(len(result), 0)

    def test_handles_no_events_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job2"
            job_dir.mkdir()
            summary = {"status": "running", "start_time": 2000}
            (job_dir / "job_summary.json").write_text(json.dumps(summary))
            result = list_running_jobs(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["processes"], 0)


if __name__ == "__main__":
    unittest.main()
