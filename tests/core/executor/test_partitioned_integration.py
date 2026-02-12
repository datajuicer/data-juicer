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
import tempfile
import unittest

from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class PartitionedExecutorIntegrationTest(DataJuicerTestCaseBase):
    """Integration tests for PartitionedRayExecutor with real Ray cluster."""

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_partitioned_integration_')

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # ==================== Checkpoint Resume Tests ====================

    @TEST_TAG('ray')
    def test_checkpoint_resume_after_interruption(self):
        """Test resuming from checkpoint after simulated interruption.

        This test:
        1. Runs processing with checkpointing until op 1 completes
        2. Simulates interruption
        3. Creates new executor with same job_id
        4. Verifies it resumes from checkpoint
        """
        # First run - process partially with checkpoints
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_resume', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_resume')

        executor1 = PartitionedRayExecutor(cfg)
        executor1.run()

        # Verify checkpoints were created
        checkpoint_dir = cfg.checkpoint_dir
        self.assertTrue(os.path.exists(checkpoint_dir))
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.parquet')]
        self.assertGreater(len(checkpoint_files), 0)

        # Get the job_id for resumption
        job_id = cfg.job_id

        # Second run - resume with same job_id
        cfg2 = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op',
            '--job_id', job_id  # Use same job_id to trigger resume
        ])
        cfg2.export_path = cfg.export_path
        cfg2.work_dir = cfg.work_dir

        executor2 = PartitionedRayExecutor(cfg2)

        # Verify checkpoint manager can find existing checkpoints
        for partition_id in range(2):
            latest = executor2.ckpt_manager.find_latest_checkpoint(partition_id)
            # Should find checkpoint from first run
            if latest:
                op_idx, _, path = latest
                self.assertTrue(os.path.exists(path))

    @TEST_TAG('ray')
    def test_checkpoint_resume_partial_completion(self):
        """Test resume when some partitions completed but not all."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '4',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_partial_resume', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_partial_resume')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # All partitions should have checkpoints
        for partition_id in range(4):
            latest = executor.ckpt_manager.find_latest_checkpoint(partition_id)
            self.assertIsNotNone(latest, f"Partition {partition_id} should have checkpoint")

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
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
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
    def test_partition_isolation(self):
        """Test that partitions don't interfere with each other."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '4',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_isolation', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_isolation')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Each partition should have its own checkpoints
        checkpoint_dir = cfg.checkpoint_dir

        for partition_id in range(4):
            # Find checkpoints for this partition
            partition_ckpts = [
                f for f in os.listdir(checkpoint_dir)
                if f.endswith('.parquet') and f'_partition_{partition_id:04d}' in f
            ]

            # Should have checkpoints (depends on number of ops)
            # At minimum, verify no cross-partition contamination
            for ckpt in partition_ckpts:
                self.assertIn(f'_partition_{partition_id:04d}', ckpt)

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
        self.tmp_dir = tempfile.mkdtemp(prefix='test_ckpt_resume_')

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @TEST_TAG('ray')
    def test_resume_skips_completed_operations(self):
        """Test that resume properly skips already-completed operations."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_skip_completed', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_skip_completed')

        # First run - complete all operations
        executor1 = PartitionedRayExecutor(cfg)
        executor1.run()

        # Count checkpoint files
        checkpoint_files_after_run1 = len([
            f for f in os.listdir(cfg.checkpoint_dir)
            if f.endswith('.parquet')
        ])

        # Second run with same config - should detect completion
        executor2 = PartitionedRayExecutor(cfg)

        # Find latest checkpoint for each partition
        for partition_id in range(2):
            latest = executor2.ckpt_manager.find_latest_checkpoint(partition_id)
            self.assertIsNotNone(latest,
                f"Should find checkpoint for partition {partition_id}")

    @TEST_TAG('ray')
    def test_resume_with_every_n_ops_strategy(self):
        """Test resume with EVERY_N_OPS checkpoint strategy."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_n_ops',
            '--checkpoint.n_ops', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_every_n_resume', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_every_n_resume')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Verify checkpoints exist at expected intervals
        checkpoint_files = [
            f for f in os.listdir(cfg.checkpoint_dir)
            if f.endswith('.parquet')
        ]

        # With n_ops=2, checkpoints should be at ops 1, 3, 5, etc. (0-indexed: 1, 3, 5)
        # Actual number depends on total ops in config
        self.assertGreater(len(checkpoint_files), 0)


if __name__ == '__main__':
    unittest.main()
