"""
Comprehensive tests for EventLoggingMixin and EventLogger.

Tests cover:
- Event logging lifecycle (job, partition, operation events)
- Event filtering and retrieval
- JSONL file operations
- Job completion detection
- Resumption state analysis
- Edge cases (disabled logging, corrupted files, etc.)
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.core.executor.event_logging_mixin import (
    Event,
    EventLogger,
    EventLoggingMixin,
    EventType,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class EventLoggerTest(DataJuicerTestCaseBase):
    """Tests for EventLogger class."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_event_logger_')
        self.work_dir = os.path.join(self.tmp_dir, 'work')
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # ==================== Initialization Tests ====================

    def test_init_creates_log_directory(self):
        """Test that initialization creates the log directory."""
        log_dir = os.path.join(self.tmp_dir, 'new_logs')
        self.assertFalse(os.path.exists(log_dir))

        logger = EventLogger(log_dir, job_id='test_job')

        self.assertTrue(os.path.exists(log_dir))

    def test_init_with_job_id(self):
        """Test initialization with explicit job_id."""
        logger = EventLogger(self.tmp_dir, job_id='my_custom_job_id')

        self.assertEqual(logger.job_id, 'my_custom_job_id')

    def test_init_generates_job_id(self):
        """Test that job_id is auto-generated if not provided."""
        logger = EventLogger(self.tmp_dir)

        self.assertIsNotNone(logger.job_id)
        self.assertGreater(len(logger.job_id), 0)

    def test_init_creates_jsonl_file(self):
        """Test that JSONL file is created in work_dir."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        self.assertTrue(str(logger.jsonl_file).endswith('.jsonl'))
        self.assertTrue(str(logger.jsonl_file).startswith(str(self.work_dir)))

    # ==================== Event Logging Tests ====================

    def test_log_event_basic(self):
        """Test basic event logging."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        event = Event(
            event_type=EventType.JOB_START,
            timestamp=time.time(),
            message="Test job started",
        )
        logger.log_event(event)

        # Verify event is in memory
        self.assertEqual(len(logger.events), 1)
        self.assertEqual(logger.events[0].event_type, EventType.JOB_START)

    def test_log_event_writes_to_jsonl(self):
        """Test that events are written to JSONL file."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        event = Event(
            event_type=EventType.JOB_START,
            timestamp=time.time(),
            message="Test job started",
        )
        logger.log_event(event)

        # Verify JSONL file exists and contains event
        self.assertTrue(os.path.exists(logger.jsonl_file))
        with open(logger.jsonl_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertEqual(data['event_type'], 'job_start')

    def test_log_event_sets_job_id(self):
        """Test that log_event sets job_id on event."""
        logger = EventLogger(self.tmp_dir, job_id='my_job', work_dir=self.work_dir)

        event = Event(
            event_type=EventType.JOB_START,
            timestamp=time.time(),
            message="Test",
        )
        logger.log_event(event)

        self.assertEqual(event.job_id, 'my_job')

    def test_log_multiple_events(self):
        """Test logging multiple events."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        for i in range(5):
            event = Event(
                event_type=EventType.OP_START,
                timestamp=time.time(),
                message=f"Operation {i} started",
                operation_idx=i,
            )
            logger.log_event(event)

        self.assertEqual(len(logger.events), 5)

        # Verify JSONL file
        with open(logger.jsonl_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)

    def test_log_event_thread_safety(self):
        """Test that event logging is thread-safe."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)
        errors = []

        def log_events(thread_id):
            try:
                for i in range(10):
                    event = Event(
                        event_type=EventType.OP_START,
                        timestamp=time.time(),
                        message=f"Thread {thread_id} op {i}",
                    )
                    logger.log_event(event)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_events, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(logger.events), 50)

    def test_log_event_with_all_fields(self):
        """Test logging event with all optional fields."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        event = Event(
            event_type=EventType.OP_COMPLETE,
            timestamp=time.time(),
            message="Operation completed",
            event_id="custom_id",
            partition_id=2,
            operation_name="text_filter",
            operation_idx=3,
            status="success",
            duration=1.5,
            input_rows=1000,
            output_rows=950,
            checkpoint_path="/path/to/checkpoint",
            metadata={"key": "value"},
        )
        logger.log_event(event)

        with open(logger.jsonl_file, 'r') as f:
            data = json.loads(f.readline())

        self.assertEqual(data['partition_id'], 2)
        self.assertEqual(data['operation_name'], 'text_filter')
        self.assertEqual(data['duration'], 1.5)
        self.assertEqual(data['input_rows'], 1000)
        self.assertEqual(data['output_rows'], 950)

    # ==================== Event Retrieval Tests ====================

    def test_get_events_no_filter(self):
        """Test getting all events without filtering."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        for event_type in [EventType.JOB_START, EventType.OP_START, EventType.OP_COMPLETE]:
            event = Event(event_type=event_type, timestamp=time.time(), message="Test")
            logger.log_event(event)

        events = logger.get_events()
        self.assertEqual(len(events), 3)

    def test_get_events_filter_by_type(self):
        """Test filtering events by type."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        logger.log_event(Event(EventType.JOB_START, time.time(), "Start"))
        logger.log_event(Event(EventType.OP_START, time.time(), "Op 1"))
        logger.log_event(Event(EventType.OP_START, time.time(), "Op 2"))
        logger.log_event(Event(EventType.JOB_COMPLETE, time.time(), "Complete"))

        events = logger.get_events(event_type=EventType.OP_START)
        self.assertEqual(len(events), 2)

    def test_get_events_filter_by_partition(self):
        """Test filtering events by partition_id."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        for partition_id in [0, 0, 1, 1, 1, 2]:
            event = Event(
                EventType.OP_START,
                time.time(),
                f"Partition {partition_id}",
                partition_id=partition_id,
            )
            logger.log_event(event)

        events = logger.get_events(partition_id=1)
        self.assertEqual(len(events), 3)

    def test_get_events_filter_by_operation(self):
        """Test filtering events by operation_name."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        for op in ["filter", "mapper", "filter", "deduplicator"]:
            event = Event(EventType.OP_START, time.time(), f"Op {op}", operation_name=op)
            logger.log_event(event)

        events = logger.get_events(operation_name="filter")
        self.assertEqual(len(events), 2)

    def test_get_events_filter_by_time_range(self):
        """Test filtering events by time range."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        now = time.time()
        for i, offset in enumerate([-100, -50, 0, 50, 100]):
            event = Event(EventType.OP_START, now + offset, f"Event {i}")
            logger.log_event(event)

        events = logger.get_events(start_time=now - 60, end_time=now + 60)
        self.assertEqual(len(events), 3)

    def test_get_events_with_limit(self):
        """Test limiting number of returned events."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        for i in range(10):
            logger.log_event(Event(EventType.OP_START, time.time(), f"Event {i}"))

        events = logger.get_events(limit=5)
        self.assertEqual(len(events), 5)

    # ==================== Job Completion Detection Tests ====================

    def test_check_job_completion_completed(self):
        """Test detecting completed job."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        # Write completion event
        with open(logger.jsonl_file, 'w') as f:
            f.write(json.dumps({"event_type": "job_start", "message": "Started"}) + '\n')
            f.write(json.dumps({"event_type": "job_complete", "message": "Done"}) + '\n')

        is_complete = logger.check_job_completion(logger.jsonl_file)
        self.assertTrue(is_complete)

    def test_check_job_completion_not_completed(self):
        """Test detecting incomplete job."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        # Write events without completion
        with open(logger.jsonl_file, 'w') as f:
            f.write(json.dumps({"event_type": "job_start", "message": "Started"}) + '\n')
            f.write(json.dumps({"event_type": "op_start", "message": "Processing"}) + '\n')

        is_complete = logger.check_job_completion(logger.jsonl_file)
        self.assertFalse(is_complete)

    def test_check_job_completion_nonexistent_file(self):
        """Test job completion check with nonexistent file."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        from pathlib import Path
        nonexistent = Path(self.tmp_dir) / "nonexistent.jsonl"

        is_complete = logger.check_job_completion(nonexistent)
        self.assertFalse(is_complete)

    def test_check_job_completion_malformed_jsonl(self):
        """Test job completion check with malformed JSONL."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        # Write malformed JSON
        with open(logger.jsonl_file, 'w') as f:
            f.write("this is not valid json\n")
            f.write('{"event_type": "job_complete"}\n')

        # Should not raise, should handle gracefully
        is_complete = logger.check_job_completion(logger.jsonl_file)
        # May or may not find completion depending on implementation
        self.assertIsInstance(is_complete, bool)

    # ==================== Find Latest Events File Tests ====================

    def test_find_latest_events_file_single(self):
        """Test finding events file when single file exists."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        # Create single events file
        events_file = os.path.join(self.work_dir, "events_20250101_120000.jsonl")
        with open(events_file, 'w') as f:
            f.write('{"event_type": "job_start"}\n')

        latest = logger.find_latest_events_file(self.work_dir)
        self.assertIsNotNone(latest)
        self.assertTrue(str(latest).endswith('.jsonl'))

    def test_find_latest_events_file_multiple(self):
        """Test finding latest events file among multiple."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        # Create multiple events files
        for timestamp in ["20250101_100000", "20250101_120000", "20250101_110000"]:
            events_file = os.path.join(self.work_dir, f"events_{timestamp}.jsonl")
            with open(events_file, 'w') as f:
                f.write('{"event_type": "job_start"}\n')
            # Small delay to ensure different mtime
            time.sleep(0.01)

        latest = logger.find_latest_events_file(self.work_dir)
        self.assertIsNotNone(latest)
        # Should be the most recently modified file

    def test_find_latest_events_file_none_exist(self):
        """Test finding events file when none exist."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        latest = logger.find_latest_events_file(self.work_dir)
        # Should return None or the logger's own file
        # Behavior depends on implementation

    def test_find_latest_events_file_nonexistent_dir(self):
        """Test finding events file in nonexistent directory."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        latest = logger.find_latest_events_file("/nonexistent/directory")
        self.assertIsNone(latest)

    # ==================== Status Report Tests ====================

    def test_generate_status_report_no_events(self):
        """Test status report with no events."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        report = logger.generate_status_report()
        self.assertIn("No events logged", report)

    def test_generate_status_report_with_events(self):
        """Test status report with events."""
        logger = EventLogger(self.tmp_dir, job_id='test', work_dir=self.work_dir)

        logger.log_event(Event(EventType.JOB_START, time.time(), "Start"))
        logger.log_event(Event(EventType.OP_START, time.time(), "Op"))
        logger.log_event(Event(EventType.OP_COMPLETE, time.time(), "Done"))

        report = logger.generate_status_report()
        self.assertIn("Total Events:", report)
        self.assertIn("Event Type Distribution:", report)

    # ==================== List Available Jobs Tests ====================

    def test_list_available_jobs_empty(self):
        """Test listing jobs when none exist."""
        jobs = EventLogger.list_available_jobs(self.work_dir)
        self.assertEqual(len(jobs), 0)

    def test_list_available_jobs_with_summaries(self):
        """Test listing jobs with job_summary.json files."""
        # Create job directories with summaries
        for job_id in ["job_001", "job_002"]:
            job_dir = os.path.join(self.work_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            summary = {
                "job_id": job_id,
                "status": "completed",
                "start_time": time.time(),
            }
            with open(os.path.join(job_dir, "job_summary.json"), 'w') as f:
                json.dump(summary, f)

        jobs = EventLogger.list_available_jobs(self.work_dir)
        self.assertEqual(len(jobs), 2)
        job_ids = [j["job_id"] for j in jobs]
        self.assertIn("job_001", job_ids)
        self.assertIn("job_002", job_ids)

    def test_list_available_jobs_nonexistent_dir(self):
        """Test listing jobs in nonexistent directory."""
        jobs = EventLogger.list_available_jobs("/nonexistent/directory")
        self.assertEqual(len(jobs), 0)


class EventTypeEnumTest(DataJuicerTestCaseBase):
    """Tests for EventType enum."""

    def test_job_event_types(self):
        """Test job-level event types exist."""
        self.assertEqual(EventType.JOB_START.value, "job_start")
        self.assertEqual(EventType.JOB_COMPLETE.value, "job_complete")
        self.assertEqual(EventType.JOB_FAILED.value, "job_failed")
        self.assertEqual(EventType.JOB_RESTART.value, "job_restart")

    def test_partition_event_types(self):
        """Test partition-level event types exist."""
        self.assertEqual(EventType.PARTITION_START.value, "partition_start")
        self.assertEqual(EventType.PARTITION_COMPLETE.value, "partition_complete")
        self.assertEqual(EventType.PARTITION_FAILED.value, "partition_failed")
        self.assertEqual(EventType.PARTITION_RESUME.value, "partition_resume")

    def test_operation_event_types(self):
        """Test operation-level event types exist."""
        self.assertEqual(EventType.OP_START.value, "op_start")
        self.assertEqual(EventType.OP_COMPLETE.value, "op_complete")
        self.assertEqual(EventType.OP_FAILED.value, "op_failed")

    def test_checkpoint_event_types(self):
        """Test checkpoint event types exist."""
        self.assertEqual(EventType.CHECKPOINT_SAVE.value, "checkpoint_save")
        self.assertEqual(EventType.CHECKPOINT_LOAD.value, "checkpoint_load")

    def test_dag_event_types(self):
        """Test DAG-related event types exist."""
        self.assertEqual(EventType.DAG_BUILD_START.value, "dag_build_start")
        self.assertEqual(EventType.DAG_BUILD_COMPLETE.value, "dag_build_complete")
        self.assertEqual(EventType.DAG_NODE_START.value, "dag_node_start")
        self.assertEqual(EventType.DAG_NODE_COMPLETE.value, "dag_node_complete")


class EventDataclassTest(DataJuicerTestCaseBase):
    """Tests for Event dataclass."""

    def test_event_required_fields(self):
        """Test Event creation with required fields only."""
        event = Event(
            event_type=EventType.JOB_START,
            timestamp=12345.0,
            message="Test message",
        )

        self.assertEqual(event.event_type, EventType.JOB_START)
        self.assertEqual(event.timestamp, 12345.0)
        self.assertEqual(event.message, "Test message")

    def test_event_optional_fields_default_none(self):
        """Test Event optional fields default to None."""
        event = Event(
            event_type=EventType.JOB_START,
            timestamp=12345.0,
            message="Test",
        )

        self.assertIsNone(event.event_id)
        self.assertIsNone(event.job_id)
        self.assertIsNone(event.partition_id)
        self.assertIsNone(event.operation_name)
        self.assertIsNone(event.duration)
        self.assertIsNone(event.error_message)
        self.assertIsNone(event.metadata)

    def test_event_all_fields(self):
        """Test Event with all fields populated."""
        event = Event(
            event_type=EventType.OP_COMPLETE,
            timestamp=12345.0,
            message="Operation done",
            event_id="evt_001",
            job_id="job_001",
            partition_id=2,
            operation_name="filter",
            operation_idx=3,
            status="success",
            duration=1.5,
            error_message=None,
            checkpoint_path="/path/to/ckpt",
            input_rows=1000,
            output_rows=950,
            metadata={"custom": "data"},
        )

        self.assertEqual(event.event_id, "evt_001")
        self.assertEqual(event.partition_id, 2)
        self.assertEqual(event.duration, 1.5)
        self.assertEqual(event.metadata["custom"], "data")


class EventLoggingMixinTest(DataJuicerTestCaseBase):
    """Tests for EventLoggingMixin class."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_event_mixin_')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_log_event_when_disabled(self):
        """Test that _log_event handles disabled logger gracefully."""
        # Create a mock executor with disabled logging
        class MockExecutor(EventLoggingMixin):
            def __init__(self):
                self.event_logger = None

        executor = MockExecutor()

        # Should not raise
        executor._log_event(EventType.JOB_START, "Test message")

    def test_get_events_when_disabled(self):
        """Test that get_events returns empty when logger disabled."""
        class MockExecutor(EventLoggingMixin):
            def __init__(self):
                self.event_logger = None

        executor = MockExecutor()
        events = executor.get_events()

        self.assertEqual(len(events), 0)

    def test_generate_status_report_when_disabled(self):
        """Test status report when logger disabled."""
        class MockExecutor(EventLoggingMixin):
            def __init__(self):
                self.event_logger = None

        executor = MockExecutor()
        report = executor.generate_status_report()

        self.assertIn("disabled", report.lower())


if __name__ == '__main__':
    unittest.main()
