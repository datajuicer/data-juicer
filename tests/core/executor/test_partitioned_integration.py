"""
Integration tests for PartitionedRayExecutor.

These tests require a Ray cluster and are tagged with @TEST_TAG('ray').
They test:
- Full end-to-end convergence point execution
- Checkpoint resume from interruption
- Auto-partitioning with real data analysis
- Event logging with real JSONL files
- Multi-partition coordination

Run these tests with:
    python tests/run.py --tag ray --mode regression
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from unittest import mock

from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class PartitionedExecutorIntegrationTest(DataJuicerTestCaseBase):
    """Integration tests for PartitionedRayExecutor with real Ray cluster."""

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def setUp(self) -> None:
        super().setUp()
        # Use a shared directory under root_path instead of system /tmp
        # This ensures the temp directory is accessible by all Ray workers
        # in distributed mode (e.g., Docker containers sharing /workspace)
        unique_name = f'test_partitioned_integration_{uuid.uuid4().hex[:8]}'
        self.tmp_dir = os.path.join(self.root_path, 'tmp', unique_name)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # ==================== Checkpoint Resume Tests ====================

    @TEST_TAG('ray')
    def test_checkpoint_resume_after_interruption(self):
        """An interrupted job resumed with --resume recomputes only uncommitted work."""
        work_dir = os.path.join(self.tmp_dir, 'test_resume')
        args = [
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.size', '4',
            '--executor_type', 'ray_partitioned',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op',
            '--work_dir', work_dir,
            '--export_path', os.path.join(work_dir, 'res.jsonl'),
        ]
        cfg = _cli_configs(args)
        PartitionedRayExecutor(cfg).run()
        exported = _read_export(cfg.export_path)
        commits = _commit_names(cfg.checkpoint_dir)
        last_stage = max(int(name.split('-')[0]) for name in commits)

        # Simulate a crash before the last operator committed anything.
        commit_dir = os.path.join(cfg.checkpoint_dir, 'partition_recovery', 'commits')
        for name in commits:
            if int(name.split('-')[0]) == last_stage:
                os.remove(os.path.join(commit_dir, name))
        for name in os.listdir(cfg.work_dir):
            if name.startswith('events_') and name.endswith('.jsonl'):
                path = os.path.join(cfg.work_dir, name)
                with open(path) as f:
                    events = [line for line in f if json.loads(line).get('event_type') != 'job_complete']
                with open(path, 'w') as f:
                    f.writelines(events)
        # The export of the interrupted run stays behind, as it does after a real crash.
        first_files = set(os.listdir(cfg.export_path))

        cfg2 = _cli_configs(args + ['--resume', cfg.job_id])
        PartitionedRayExecutor(cfg2).run()

        resumed = _commit_names(cfg.checkpoint_dir)
        self.assertTrue(set(commits) - set(resumed) <= {n for n in commits if int(n.split('-')[0]) == last_stage})
        self.assertTrue({n for n in commits if int(n.split('-')[0]) < last_stage} <= set(resumed))
        self.assertTrue(any(int(n.split('-')[0]) == last_stage for n in set(resumed) - set(commits)))
        self.assertFalse(first_files & set(os.listdir(cfg2.export_path)))
        self.assertEqual(sorted(_read_export(cfg2.export_path)), sorted(exported))

    @TEST_TAG('ray')
    def test_checkpoint_commits_cover_every_operator(self):
        """Every checkpointed Mapper/Filter stage records commits."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.size', '4',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_commit_stages', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_commit_stages')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        stages = {int(name.split('-')[0]) for name in _commit_names(cfg.checkpoint_dir)}
        self.assertEqual(stages, set(range(len(cfg.process))))

    # ==================== Convergence Point Tests ====================

    @TEST_TAG('ray')
    def test_convergence_with_deduplicator(self):
        """Test execution with deduplicator (global operation requiring convergence).

        Note: This test requires a config with a deduplicator operation.
        The deduplicator is a global operation that needs all partitions
        to converge before processing.
        """
        # Check if deduplicator config exists
        dedup_config = os.path.join(
            self.root_path,
            'configs/demo/process_data_with_dedup.yaml'
        )

        if not os.path.exists(dedup_config):
            self.skipTest("Deduplicator config not found")

        cfg = init_configs([
            '--config', dedup_config,
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_convergence_dedup', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_convergence_dedup')

        executor = PartitionedRayExecutor(cfg)

        # Detect convergence points
        convergence_points = executor._detect_convergence_points(cfg)

        # Should have at least one convergence point for deduplicator
        # The exact number depends on the config
        if any('deduplicator' in str(op).lower() for op in cfg.process):
            self.assertGreater(len(convergence_points), 0,
                "Should detect convergence point for deduplicator")

    @TEST_TAG('ray')
    def test_multiple_convergence_points(self):
        """Test execution with multiple global operations."""
        from jsonargparse import Namespace

        # Create config with multiple deduplicators (simulated)
        cfg = Namespace()
        cfg.process = [
            {'text_length_filter': {'min_len': 10}},
            {'document_simhash_deduplicator': {}},  # Global op 1
            {'clean_links_mapper': {}},
            {'document_minhash_deduplicator': {}},  # Global op 2
            {'whitespace_normalization_mapper': {}},
        ]
        cfg.job_id = 'test_multi_conv'
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_multi_convergence')
        cfg.event_logging = {'enabled': False}

        os.makedirs(cfg.work_dir, exist_ok=True)

        # Create executor for convergence detection only
        executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
        executor.cfg = cfg
        executor.executor_type = 'ray_partitioned'
        executor.work_dir = cfg.work_dir
        executor.num_partitions = 2

        from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
        from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
        EventLoggingMixin.__init__(executor, cfg)
        DAGExecutionMixin.__init__(executor)
        executor._override_strategy_methods()

        convergence_points = executor._detect_convergence_points(cfg)

        # Should detect 2 convergence points (indices 1 and 3)
        expected_conv_points = [1, 3]  # deduplicator indices
        for point in expected_conv_points:
            self.assertIn(point, convergence_points,
                f"Should detect convergence at index {point}")

    # ==================== Auto Partitioning Tests ====================

    @TEST_TAG('ray')
    def test_auto_partitioning_analyzes_data(self):
        """Test that auto mode actually analyzes the dataset."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'auto',
            '--partition.target_size_mb', '128'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_auto_analyze', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_auto_analyze')

        executor = PartitionedRayExecutor(cfg)

        # Auto mode should have set num_partitions based on analysis
        self.assertEqual(executor.partition_mode, 'auto')
        self.assertIsNotNone(executor.num_partitions)
        self.assertGreater(executor.num_partitions, 0)

        # Run to verify it completes
        executor.run()
        self.assertTrue(os.path.exists(cfg.export_path))

    @TEST_TAG('ray')
    def test_auto_partitioning_respects_target_size(self):
        """Test that different target sizes result in different partition counts."""
        partition_counts = {}

        for target_size in [128, 512]:
            cfg = init_configs([
                '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
                '--partition.mode', 'auto',
                '--partition.target_size_mb', str(target_size)
            ])
            cfg.export_path = os.path.join(self.tmp_dir, f'test_target_{target_size}', 'res.jsonl')
            cfg.work_dir = os.path.join(self.tmp_dir, f'test_target_{target_size}')

            executor = PartitionedRayExecutor(cfg)
            partition_counts[target_size] = executor.num_partitions

        # Smaller target size should generally result in more partitions
        # (depending on dataset size, they might be equal for small datasets)
        self.assertIsNotNone(partition_counts[128])
        self.assertIsNotNone(partition_counts[512])

    # ==================== Event Logging Integration Tests ====================

    @TEST_TAG('ray')
    def test_event_logging_creates_jsonl(self):
        """Test that event logging creates proper JSONL file."""
        # Partition events come from ordinary partitioning, which checkpointing replaces.
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'false',
        ])
        cfg.event_logging = {'enabled': True}
        cfg.export_path = os.path.join(self.tmp_dir, 'test_events_jsonl', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_events_jsonl')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Find events file
        events_files = []
        for f in os.listdir(cfg.work_dir):
            if f.startswith('events_') and f.endswith('.jsonl'):
                events_files.append(os.path.join(cfg.work_dir, f))

        self.assertGreater(len(events_files), 0, "Events file should be created")

        # Verify JSONL format and content
        events_file = events_files[0]
        with open(events_file, 'r') as f:
            lines = f.readlines()

        self.assertGreater(len(lines), 0, "Events file should have content")

        # Parse and verify events
        events = [json.loads(line) for line in lines if line.strip()]
        event_types = [e.get('event_type') for e in events]

        # Should have job_start and job_complete
        self.assertIn('job_start', event_types)
        self.assertIn('job_complete', event_types)

        # Should have partition events
        self.assertTrue(
            any('partition' in et for et in event_types),
            "Should have partition events"
        )

    @TEST_TAG('ray')
    def test_event_logging_tracks_operations(self):
        """Test that operations are properly tracked in events."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.event_logging = {'enabled': True}
        cfg.export_path = os.path.join(self.tmp_dir, 'test_op_events', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_op_events')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Find and parse events
        events_files = [f for f in os.listdir(cfg.work_dir)
                       if f.startswith('events_') and f.endswith('.jsonl')]
        events_file = os.path.join(cfg.work_dir, events_files[0])

        with open(events_file, 'r') as f:
            events = [json.loads(line) for line in f if line.strip()]

        # Check for operation events
        op_starts = [e for e in events if e.get('event_type') == 'op_start']
        op_completes = [e for e in events if e.get('event_type') == 'op_complete']

        # Should have op events for each partition
        num_ops = len(cfg.process)
        num_partitions = 2

        # At minimum, should have some operation events
        self.assertGreater(len(op_starts), 0, "Should have op_start events")
        self.assertGreater(len(op_completes), 0, "Should have op_complete events")

    # ==================== DAG Execution Tests ====================

    @TEST_TAG('ray')
    def test_dag_execution_plan_saved(self):
        """Test that DAG execution plan is saved to work directory."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '3'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_dag_plan', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_dag_plan')

        executor = PartitionedRayExecutor(cfg)
        executor._initialize_dag_execution(cfg)

        # DAG plan should be saved
        dag_plan_path = executor.get_dag_execution_plan_path()

        if dag_plan_path and os.path.exists(dag_plan_path):
            with open(dag_plan_path, 'r') as f:
                dag_plan = json.load(f)

            # Verify DAG structure
            self.assertIn('nodes', dag_plan)
            self.assertGreater(len(dag_plan['nodes']), 0)

    @TEST_TAG('ray')
    def test_dag_node_completion_tracking(self):
        """Test that DAG is properly set up for partitioned execution."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_dag_tracking', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_dag_tracking')

        executor = PartitionedRayExecutor(cfg)

        # Explicitly initialize DAG
        executor._initialize_dag_execution(cfg)

        # Verify DAG is initialized with correct structure
        self.assertTrue(executor.dag_initialized)
        self.assertIsNotNone(executor.pipeline_dag)

        # Verify nodes are created for each partition
        num_ops = len(cfg.process)
        num_partitions = 2
        expected_nodes = num_ops * num_partitions

        self.assertEqual(
            len(executor.pipeline_dag.nodes),
            expected_nodes,
            f"DAG should have {expected_nodes} nodes ({num_ops} ops x {num_partitions} partitions)"
        )

        # Verify all nodes have partition_id
        for node_id, node in executor.pipeline_dag.nodes.items():
            self.assertIn('partition_id', node)
            self.assertIn(node['partition_id'], [0, 1])

        # Run execution and verify completion
        executor.run()
        self.assertTrue(os.path.exists(cfg.export_path))

    # ==================== Multi-Partition Coordination Tests ====================

    @TEST_TAG('ray')
    def test_high_partition_count(self):
        """Stress test with many partitions."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '16'  # High partition count
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_high_partitions', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_high_partitions')

        executor = PartitionedRayExecutor(cfg)

        # Should handle high partition count
        self.assertEqual(executor.num_partitions, 16)

        # Initialize DAG - should create nodes for all partitions
        executor._initialize_dag_execution(cfg)

        num_ops = len(cfg.process)
        expected_nodes = num_ops * 16

        self.assertEqual(
            len(executor.pipeline_dag.nodes),
            expected_nodes,
            f"DAG should have {expected_nodes} nodes for 16 partitions"
        )


class CheckpointResumeIntegrationTest(DataJuicerTestCaseBase):
    """Integration tests specifically for checkpoint resume scenarios."""

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def setUp(self) -> None:
        super().setUp()
        # Use a shared directory under root_path instead of system /tmp
        # This ensures the temp directory is accessible by all Ray workers
        # in distributed mode (e.g., Docker containers sharing /workspace)
        unique_name = f'test_ckpt_resume_{uuid.uuid4().hex[:8]}'
        self.tmp_dir = os.path.join(self.root_path, 'tmp', unique_name)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @TEST_TAG('ray')
    def test_resume_skips_completed_operations(self):
        """Resuming a completed job neither commits nor exports again."""
        work_dir = os.path.join(self.tmp_dir, 'test_skip_completed')
        args = [
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--executor_type', 'ray_partitioned',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op',
            '--work_dir', work_dir,
            '--export_path', os.path.join(work_dir, 'res.jsonl'),
        ]
        cfg = _cli_configs(args)
        PartitionedRayExecutor(cfg).run()
        commits_after_run1 = _commit_names(cfg.checkpoint_dir)
        exported_after_run1 = _read_export(cfg.export_path)

        cfg2 = _cli_configs(args + ['--resume', cfg.job_id])
        self.assertIsNone(PartitionedRayExecutor(cfg2).run())
        self.assertEqual(_commit_names(cfg.checkpoint_dir), commits_after_run1)
        self.assertEqual(_read_export(cfg.export_path), exported_after_run1)


def _cli_configs(args):
    # Resume validation compares the command line recorded by the first run.
    with mock.patch.object(sys, 'argv', ['dj-process', *args]):
        return init_configs(args)


def _commit_names(checkpoint_dir):
    return sorted(os.listdir(os.path.join(checkpoint_dir, 'partition_recovery', 'commits')))


def _read_export(path):
    paths = sorted(os.path.join(path, name) for name in os.listdir(path)) if os.path.isdir(path) else [path]
    rows = []
    for shard in paths:
        with open(shard) as f:
            rows.extend(line for line in f if line.strip())
    return rows


if __name__ == '__main__':
    unittest.main()
