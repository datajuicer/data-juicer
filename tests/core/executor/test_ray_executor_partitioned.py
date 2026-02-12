import os
import tempfile
import unittest
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.config import init_configs
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class PartitionedRayExecutorTest(DataJuicerTestCaseBase):
    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def setUp(self) -> None:
        super().setUp()
        # Create temporary directory
        self.tmp_dir = tempfile.mkdtemp(prefix='test_ray_executor_partitioned_')

    def tearDown(self) -> None:
        super().tearDown()
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @TEST_TAG('ray')
    def test_end2end_execution_manual_partitioning(self):
        """Test end-to-end execution with manual partitioning mode."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution_manual', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution_manual')
        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # check result files
        self.assertTrue(os.path.exists(cfg.export_path))

    @TEST_TAG('ray')
    def test_end2end_execution_with_checkpointing(self):
        """Test end-to-end execution with checkpointing enabled."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_op'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution_checkpointing', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution_checkpointing')
        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # check result files
        self.assertTrue(os.path.exists(cfg.export_path))
        
        # check checkpoint directory exists
        checkpoint_dir = cfg.checkpoint_dir
        self.assertTrue(os.path.exists(checkpoint_dir))
        
        # check that checkpoint files were created
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.parquet')]
        self.assertGreater(len(checkpoint_files), 0, "No checkpoint files were created")
        
        # verify checkpoint file naming convention
        for checkpoint_file in checkpoint_files:
            self.assertTrue(checkpoint_file.startswith('checkpoint_op_'), 
                          f"Checkpoint file {checkpoint_file} doesn't follow naming convention")
            self.assertTrue('_partition_' in checkpoint_file, 
                          f"Checkpoint file {checkpoint_file} doesn't contain partition info")
            self.assertTrue(checkpoint_file.endswith('.parquet'), 
                          f"Checkpoint file {checkpoint_file} doesn't have .parquet extension")
        
        # test checkpoint loading functionality
        executor2 = PartitionedRayExecutor(cfg)
        
        # test find_latest_checkpoint method (on checkpoint manager)
        for partition_id in range(2):
            latest_checkpoint = executor2.ckpt_manager.find_latest_checkpoint(partition_id)
            if latest_checkpoint:
                op_idx, _, checkpoint_path = latest_checkpoint
                self.assertIsInstance(op_idx, int)
                self.assertTrue(os.path.exists(checkpoint_path))
                self.assertTrue(checkpoint_path.endswith('.parquet'))

        # test resolve_checkpoint_filename method (on checkpoint manager)
        test_filename = executor2.ckpt_manager.resolve_checkpoint_filename(0, 1)
        expected_pattern = 'checkpoint_op_0000_partition_0001.parquet'
        self.assertEqual(test_filename, expected_pattern)


    @TEST_TAG('ray')
    def test_dag_execution_initialization(self):
        """Test DAG execution initialization and strategy selection."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '4'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_dag_initialization', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_dag_initialization')
        
        executor = PartitionedRayExecutor(cfg)
        
        # Test DAG initialization
        executor._initialize_dag_execution(cfg)
        
        # Verify DAG is initialized
        self.assertIsNotNone(executor.pipeline_dag)
        self.assertIsNotNone(executor.dag_execution_strategy)
        
        # Verify partitioned strategy is used
        from data_juicer.core.executor.dag_execution_strategies import PartitionedDAGStrategy
        self.assertIsInstance(executor.dag_execution_strategy, PartitionedDAGStrategy)
        
        # Verify DAG nodes are created
        self.assertGreater(len(executor.pipeline_dag.nodes), 0)

    @TEST_TAG('ray')
    def test_convergence_point_detection(self):
        """Test convergence point detection for global operations."""
        # Create a simple config without loading from file
        from jsonargparse import Namespace
        cfg = Namespace()
        cfg.process = [
            {'text_length_filter': {'min_len': 10}},
            {'text_length_filter': {'max_len': 1000}}
        ]
        cfg.job_id = 'test_convergence_123'  # Required for event logging
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_convergence')
        cfg.event_logging = {'enabled': False}  # Disable event logging for this test

        # Create executor without running full initialization
        executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
        executor.cfg = cfg
        executor.executor_type = 'ray_partitioned'
        executor.work_dir = cfg.work_dir
        executor.num_partitions = 2

        # Initialize only the necessary components
        from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
        from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
        EventLoggingMixin.__init__(executor, cfg)
        DAGExecutionMixin.__init__(executor)
        executor._override_strategy_methods()

        convergence_points = executor._detect_convergence_points(cfg)

        # Should not detect any convergence points for non-global operations
        self.assertEqual(len(convergence_points), 0)

    @TEST_TAG('ray')
    def test_partition_configuration_manual_mode(self):
        """Test manual partition configuration."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '6'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_manual_config', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_manual_config')
        
        executor = PartitionedRayExecutor(cfg)
        
        # Verify manual mode configuration
        self.assertEqual(executor.partition_mode, 'manual')
        self.assertEqual(executor.num_partitions, 6)

    @TEST_TAG('ray')
    def test_partition_configuration_auto_mode(self):
        """Test auto partition configuration."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'auto'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_auto_config', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_auto_config')
        
        executor = PartitionedRayExecutor(cfg)
        
        # Verify auto mode configuration
        self.assertEqual(executor.partition_mode, 'auto')
        # num_partitions should be set to a default value initially
        self.assertIsNotNone(executor.num_partitions)

    @TEST_TAG('ray')
    def test_checkpoint_strategies(self):
        """Test different checkpoint strategies."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true'
        ])

        # Test EVERY_OP strategy
        cfg.checkpoint = {'strategy': 'every_op'}
        executor = PartitionedRayExecutor(cfg)
        self.assertEqual(executor.ckpt_manager.checkpoint_strategy.value, 'every_op')

        # Test EVERY_N_OPS strategy
        cfg.checkpoint = {'strategy': 'every_n_ops', 'n_ops': 2}
        executor = PartitionedRayExecutor(cfg)
        self.assertEqual(executor.ckpt_manager.checkpoint_strategy.value, 'every_n_ops')
        self.assertEqual(executor.ckpt_manager.checkpoint_n_ops, 2)

        # Test MANUAL strategy
        cfg.checkpoint = {'strategy': 'manual', 'op_names': ['text_length_filter']}
        executor = PartitionedRayExecutor(cfg)
        self.assertEqual(executor.ckpt_manager.checkpoint_strategy.value, 'manual')
        self.assertIn('text_length_filter', executor.ckpt_manager.checkpoint_op_names)

        # Test DISABLED strategy
        cfg.checkpoint = {'strategy': 'disabled'}
        executor = PartitionedRayExecutor(cfg)
        self.assertEqual(executor.ckpt_manager.checkpoint_strategy.value, 'disabled')
        self.assertFalse(executor.ckpt_manager.checkpoint_enabled)

    @TEST_TAG('ray')
    def test_dag_node_generation(self):
        """Test DAG node generation for partitioned execution."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '3'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_dag_nodes', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_dag_nodes')
        
        executor = PartitionedRayExecutor(cfg)
        executor._initialize_dag_execution(cfg)
        
        # Test DAG node ID generation for different partitions
        node_id_0 = executor._get_dag_node_for_operation_partitioned('text_length_filter', 0, partition_id=0)
        node_id_1 = executor._get_dag_node_for_operation_partitioned('text_length_filter', 0, partition_id=1)
        node_id_2 = executor._get_dag_node_for_operation_partitioned('text_length_filter', 0, partition_id=2)
        
        # All should be different for different partitions
        self.assertNotEqual(node_id_0, node_id_1)
        self.assertNotEqual(node_id_1, node_id_2)
        self.assertNotEqual(node_id_0, node_id_2)
        
        # All should contain the partition ID
        self.assertIn('_partition_0', node_id_0)
        self.assertIn('_partition_1', node_id_1)
        self.assertIn('_partition_2', node_id_2)

    @TEST_TAG('ray')
    def test_global_operation_detection(self):
        """Test detection of global operations that require convergence."""
        from data_juicer.core.executor.dag_execution_strategies import is_global_operation
        
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        
        executor = PartitionedRayExecutor(cfg)
        
        # Test deduplicator detection
        from data_juicer.ops.deduplicator.ray_bts_minhash_deduplicator import RayBTSMinhashDeduplicator
        deduplicator = RayBTSMinhashDeduplicator(hash_func='sha1', threshold=0.7)
        self.assertTrue(is_global_operation(deduplicator))
        
        # Test non-global operation
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        text_filter = TextLengthFilter(min_len=10)
        self.assertFalse(is_global_operation(text_filter))

    @TEST_TAG('ray')
    def test_executor_initialization_with_legacy_config(self):
        """Test executor initialization with legacy num_partitions config."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml')
        ])
        # Use legacy num_partitions instead of partition config
        cfg.num_partitions = 5
        cfg.export_path = os.path.join(self.tmp_dir, 'test_legacy_config', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_legacy_config')
        
        executor = PartitionedRayExecutor(cfg)
        
        # Should fall back to manual mode with legacy config
        self.assertEqual(executor.partition_mode, 'manual')
        self.assertEqual(executor.num_partitions, 5)

    @TEST_TAG('ray')
    def test_job_resumption_workflow(self):
        """Test job resumption workflow with user-provided job_id."""
        from unittest.mock import Mock, patch, MagicMock
        import json

        # Create a simple config without loading from file
        from jsonargparse import Namespace
        cfg = Namespace()
        cfg.process = [{'text_length_filter': {'min_len': 10}}]
        cfg.dataset_path = 'test.jsonl'
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_job_resumption')
        cfg.export_path = os.path.join(self.tmp_dir, 'test_job_resumption', 'res.jsonl')
        cfg.partition = {'mode': 'manual', 'num_of_partitions': 2}
        cfg.checkpoint = {'enabled': True, 'strategy': 'every_op'}
        cfg._user_provided_job_id = False
        cfg.job_id = 'test_job_resumption_123'  # Required for event logging
        cfg.event_logging = {'enabled': True}  # Enable event logging for this test

        # Create work_dir first
        os.makedirs(cfg.work_dir, exist_ok=True)

        # Create executor without running full initialization
        executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
        executor.cfg = cfg
        executor.executor_type = 'ray_partitioned'
        executor.work_dir = cfg.work_dir
        executor.num_partitions = 2

        # Initialize only the necessary components
        from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
        from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
        EventLoggingMixin.__init__(executor, cfg)
        DAGExecutionMixin.__init__(executor)
        executor._override_strategy_methods()
        
        # Test 1: Check job resumption when no job exists
        cfg._user_provided_job_id = False
        result = executor._resume_job('nonexistent_job')
        self.assertEqual(result, "failed")
        
        # Test 2: Test job completion check with mock job directory
        job_id = 'test_job_123'
        job_dir = os.path.join(cfg.work_dir, f'20250101_120000_{job_id}')
        os.makedirs(job_dir, exist_ok=True)
        
        # Create events file directly in job directory (required for job completion check)
        events_file = os.path.join(job_dir, 'events_20250101_120000.jsonl')
        with open(events_file, 'w') as f:
            f.write('{"timestamp": "2025-01-01T12:00:00", "event_type": "job_start", "message": "Job started"}\n')
            f.write('{"timestamp": "2025-01-01T12:01:00", "event_type": "job_complete", "message": "Job completed"}\n')
        
        # Test job completion check directly
        is_completed = executor._check_job_completion(job_dir, job_id)
        self.assertTrue(is_completed)
        
        # Test 3: Test job completion check with incomplete job
        with open(events_file, 'w') as f:
            f.write('{"timestamp": "2025-01-01T12:00:00", "event_type": "job_start", "message": "Job started"}\n')
            f.write('{"timestamp": "2025-01-01T12:01:00", "event_type": "op_start", "message": "Operation started"}\n')
        
        is_completed = executor._check_job_completion(job_dir, job_id)
        self.assertFalse(is_completed)
        
        # Test 4: Test job resumption with proper job directory (mock the directory finding)
        cfg._user_provided_job_id = True
        cfg.job_id = job_id
        
        # Mock the work directory finding to return our test directory
        with patch.object(executor, '_find_work_directory', return_value=job_dir):
            result = executor._resume_job(job_id)
            # Should return "failed" due to config validation failure (we didn't save the config)
            self.assertEqual(result, "failed")


    # ==================== Edge Case Tests ====================

    @TEST_TAG('ray')
    def test_single_partition(self):
        """Test execution with single partition (edge case)."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '1'  # Single partition
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_single_partition', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_single_partition')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Verify execution completes
        self.assertTrue(os.path.exists(cfg.export_path))
        self.assertEqual(executor.num_partitions, 1)

    @TEST_TAG('ray')
    def test_checkpoint_every_n_ops_strategy(self):
        """Test checkpointing with EVERY_N_OPS strategy."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_n_ops',
            '--checkpoint.n_ops', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_every_n_ops', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_every_n_ops')

        executor = PartitionedRayExecutor(cfg)

        # Verify strategy configuration
        self.assertEqual(executor.ckpt_manager.checkpoint_strategy.value, 'every_n_ops')
        self.assertEqual(executor.ckpt_manager.checkpoint_n_ops, 2)

        # Run and verify
        executor.run()
        self.assertTrue(os.path.exists(cfg.export_path))

    @TEST_TAG('ray')
    def test_checkpoint_disabled(self):
        """Test execution with checkpointing disabled."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'false'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_no_checkpoint', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_no_checkpoint')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Verify execution completes without checkpoints
        self.assertTrue(os.path.exists(cfg.export_path))

        # Checkpoint directory might exist but should be empty or not created
        checkpoint_dir = cfg.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.parquet')]
            self.assertEqual(len(checkpoint_files), 0, "Checkpoints should not be created when disabled")

    @TEST_TAG('ray')
    def test_partition_target_size_configuration(self):
        """Test configurable partition target size."""
        for target_size in [128, 256, 512]:
            cfg = init_configs([
                '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
                '--partition.mode', 'auto',
                '--partition.target_size_mb', str(target_size)
            ])
            cfg.export_path = os.path.join(self.tmp_dir, f'test_target_{target_size}', 'res.jsonl')
            cfg.work_dir = os.path.join(self.tmp_dir, f'test_target_{target_size}')

            executor = PartitionedRayExecutor(cfg)

            # Verify target size is set
            self.assertEqual(cfg.partition.target_size_mb, target_size)

    @TEST_TAG('ray')
    def test_event_logging_disabled(self):
        """Test execution with event logging disabled."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.event_logging = {'enabled': False}
        cfg.export_path = os.path.join(self.tmp_dir, 'test_no_events', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_no_events')

        executor = PartitionedRayExecutor(cfg)
        executor.run()

        # Verify execution completes
        self.assertTrue(os.path.exists(cfg.export_path))

        # Event logger should be None
        self.assertIsNone(executor.event_logger)

    @TEST_TAG('ray')
    def test_work_directory_creation(self):
        """Test that work directory and subdirectories are created."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_work_dir', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_work_dir')

        executor = PartitionedRayExecutor(cfg)

        # Verify work directory exists
        self.assertTrue(os.path.exists(cfg.work_dir))

    @TEST_TAG('ray')
    def test_dag_execution_status(self):
        """Test DAG execution status reporting."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_dag_status', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_dag_status')

        executor = PartitionedRayExecutor(cfg)
        executor._initialize_dag_execution(cfg)

        # Get DAG status
        status = executor.get_dag_execution_status()

        self.assertIsNotNone(status)
        # Check that status is not "not_initialized" (meaning DAG is initialized)
        self.assertIn('status', status)
        self.assertNotEqual(status['status'], 'not_initialized')
        # Check expected keys exist in initialized status
        self.assertIn('summary', status)
        self.assertIn('execution_plan_length', status)

    @TEST_TAG('ray')
    def test_operation_grouping_integration(self):
        """Test that operation grouping works correctly in execution."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2',
            '--checkpoint.enabled', 'true',
            '--checkpoint.strategy', 'every_n_ops',
            '--checkpoint.n_ops', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_op_grouping', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_op_grouping')

        executor = PartitionedRayExecutor(cfg)

        # Get operation groups from checkpoint manager
        # Note: This tests the grouping logic is accessible
        from data_juicer.utils.ckpt_utils import CheckpointStrategy
        self.assertEqual(executor.ckpt_manager.checkpoint_strategy, CheckpointStrategy.EVERY_N_OPS)


class PartitionedRayExecutorEdgeCasesTest(DataJuicerTestCaseBase):
    """Additional edge case tests for PartitionedRayExecutor."""

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_ray_executor_edge_')

    def tearDown(self) -> None:
        super().tearDown()
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @TEST_TAG('ray')
    def test_many_partitions(self):
        """Test execution with many partitions (stress test)."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '8'  # Many partitions
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_many_partitions', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_many_partitions')

        executor = PartitionedRayExecutor(cfg)
        self.assertEqual(executor.num_partitions, 8)

        # Should initialize successfully
        executor._initialize_dag_execution(cfg)

        # DAG should have nodes for each partition
        num_ops = len(cfg.process)
        expected_nodes = num_ops * 8  # ops * partitions
        self.assertEqual(len(executor.pipeline_dag.nodes), expected_nodes)

    @TEST_TAG('ray')
    def test_checkpoint_file_naming_consistency(self):
        """Test checkpoint file naming is consistent across partitions."""
        from data_juicer.utils.ckpt_utils import RayCheckpointManager, CheckpointStrategy

        ckpt_dir = os.path.join(self.tmp_dir, 'test_ckpt_naming')
        os.makedirs(ckpt_dir, exist_ok=True)

        mgr = RayCheckpointManager(
            ckpt_dir=ckpt_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )

        # Test filename generation for various op/partition combinations
        test_cases = [
            (0, 0, "checkpoint_op_0000_partition_0000.parquet"),
            (0, 1, "checkpoint_op_0000_partition_0001.parquet"),
            (5, 3, "checkpoint_op_0005_partition_0003.parquet"),
            (99, 15, "checkpoint_op_0099_partition_0015.parquet"),
        ]

        for op_idx, partition_id, expected in test_cases:
            filename = mgr.resolve_checkpoint_filename(op_idx, partition_id)
            self.assertEqual(filename, expected,
                f"Mismatch for op={op_idx}, partition={partition_id}")

    @TEST_TAG('ray')
    def test_checkpoint_manual_with_nonexistent_ops(self):
        """Test MANUAL checkpoint strategy with non-existent operation names."""
        from data_juicer.utils.ckpt_utils import RayCheckpointManager, CheckpointStrategy

        ckpt_dir = os.path.join(self.tmp_dir, 'test_ckpt_manual_nonexistent')
        os.makedirs(ckpt_dir, exist_ok=True)

        mgr = RayCheckpointManager(
            ckpt_dir=ckpt_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.MANUAL,
            checkpoint_op_names=["nonexistent_op_1", "nonexistent_op_2"],
        )

        # Should not checkpoint any operation that doesn't match
        self.assertFalse(mgr.should_checkpoint(0, "text_filter"))
        self.assertFalse(mgr.should_checkpoint(1, "mapper"))

        # Should checkpoint matching operations
        self.assertTrue(mgr.should_checkpoint(2, "nonexistent_op_1"))

    @TEST_TAG('ray')
    def test_auto_mode_execution(self):
        """Test end-to-end auto mode execution."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'auto',
            '--partition.target_size_mb', '256'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_auto_mode_exec', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_auto_mode_exec')

        executor = PartitionedRayExecutor(cfg)

        # Verify auto mode is set
        self.assertEqual(executor.partition_mode, 'auto')

        # Run execution
        executor.run()

        # Verify output
        self.assertTrue(os.path.exists(cfg.export_path))

    @TEST_TAG('ray')
    def test_dag_node_status_transitions(self):
        """Test DAG node status transitions during execution."""
        cfg = init_configs([
            '--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml'),
            '--partition.mode', 'manual',
            '--partition.num_of_partitions', '2'
        ])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_dag_status_trans', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_dag_status_trans')

        executor = PartitionedRayExecutor(cfg)
        executor._initialize_dag_execution(cfg)

        # Get a node ID
        if executor.pipeline_dag.nodes:
            node_id = list(executor.pipeline_dag.nodes.keys())[0]
            node = executor.pipeline_dag.nodes[node_id]

            # Initial status should be pending
            self.assertEqual(node["status"], "pending")

            # Mark as started
            executor._mark_dag_node_started(node_id)
            self.assertEqual(executor.pipeline_dag.nodes[node_id]["status"], "running")

            # Mark as completed
            executor._mark_dag_node_completed(node_id, duration=1.0)
            self.assertEqual(executor.pipeline_dag.nodes[node_id]["status"], "completed")


if __name__ == '__main__':
    unittest.main()
