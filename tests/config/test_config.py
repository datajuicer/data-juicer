import os
import sys
import copy
import unittest
import tempfile
import yaml
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from jsonargparse import Namespace, namespace_to_dict

from data_juicer.config import init_configs, get_default_cfg, validate_work_dir_config, resolve_job_id, resolve_job_directories
from data_juicer.ops import load_ops
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from data_juicer.utils.constant import RAY_JOB_ENV_VAR


test_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'demo_4_test.yaml')

test_bad_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'demo_4_test_bad_val.yaml')

test_text_keys_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'demo_4_test_multiple_text_keys.yaml')

test_same_ops_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       'demo_4_test_same_ops.yaml')

WORKDIR = os.path.join(os.getcwd(), 'outputs/demo')

class ConfigTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        self.tmp_dir = 'tmp/test_config/'
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()

        if os.path.exists(self.tmp_dir):
            os.system(f'rm -rf {self.tmp_dir}')

        os.environ[RAY_JOB_ENV_VAR] = "0"

    def test_help_info(self):
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            _ = init_configs(args=['--help'])
        out_str = out.getvalue()
        # self.assertIn('usage:', out_str, 'lacks message for command beginning')
        self.assertIn('--config CONFIG', out_str,
                      'lacks message for positional argument')
        self.assertIn('--project_name PROJECT_NAME', out_str,
                      'lacks message for optional argument')
        self.assertIn(
            'Number of processes to process dataset. (type:', out_str,
            'the help message of `np` argument does not show as expected')

    def test_yaml_cfg_file(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_yaml_path}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'test_demo')
            self.assertDictEqual(
                cfg.process[0], {
                    'whitespace_normalization_mapper': {
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'cpu_required': None,
                        'mem_required': None,
                        'gpu_required': None,
                        'turbo': False,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                }, 'nested dict load fail, for nonparametric op')
            self.assertDictEqual(
                cfg.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'min_closed_interval': True,
                        'max_closed_interval': True,
                        'reversed_range': False,
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': None,
                        'mem_required': None,
                        'turbo': False,
                        'gpu_required': None,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                }, 'nested dict load fail, un-expected internal value')

            ops_from_cfg = load_ops(cfg.process)
            self.assertTrue(len(ops_from_cfg) == 4)

    def test_val_range_check_cmd(self):
        out = StringIO()
        err_msg_head = ("remove_table_text_mapper.min_col")
        err_msg = ("Input should be greater than or equal to 2")
        with redirect_stdout(out), redirect_stderr(out):
            with self.assertRaises(SystemExit) as cm:
                init_configs(
                    args=f'--config {test_yaml_path} '
                          '--remove_table_text_mapper.min_col 1'.split())
            self.assertEqual(cm.exception.code, 2)
        out_str = out.getvalue()
        self.assertIn(err_msg_head, out_str)
        self.assertIn(err_msg, out_str)

    def _test_val_range_check_yaml(self):
        out = StringIO()
        err_msg_head = ("remove_table_text_mapper.max_col")
        err_msg = ("Input should be less than or equal to 20")
        with redirect_stdout(out), redirect_stderr(out):
            with self.assertRaises(SystemExit) as cm:
                init_configs(args=f'--config {test_bad_yaml_path}'.split())
            self.assertEqual(cm.exception.code, 2)
        out_str = out.getvalue()
        self.assertIn(err_msg_head, out_str)
        self.assertIn(err_msg, out_str)

    def test_mixture_cfg(self):
        out = StringIO()
        with redirect_stdout(out):
            ori_cfg = init_configs(args=f'--config {test_yaml_path}'.split())
            mixed_cfg_1 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang en'.split())
            mixed_cfg_2 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang=fr'.split())
            mixed_cfg_3 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang zh '
                '--language_id_score_filter.min_score 0.6'.split())
            mixed_cfg_4 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang=en '
                '--language_id_score_filter.min_score=0.5'.split())
            print(f'ori_cfg.process[1] = {ori_cfg.process[1]}')
            self.assertDictEqual(
                ori_cfg.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'min_closed_interval': True,
                        'max_closed_interval': True,
                        'reversed_range': False,
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': None,
                        'mem_required': None,
                        'gpu_required': None,
                        'turbo': False,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_1.process[1], {
                    'language_id_score_filter': {
                        'lang': 'en',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'min_closed_interval': True,
                        'max_closed_interval': True,
                        'reversed_range': False,
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': None,
                        'mem_required': None,
                        'turbo': False,
                        'gpu_required': None,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_2.process[1], {
                    'language_id_score_filter': {
                        'lang': 'fr',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'min_closed_interval': True,
                        'max_closed_interval': True,
                        'reversed_range': False,
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': None,
                        'mem_required': None,
                        'turbo': False,
                        'gpu_required': None,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_3.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.6,
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'min_closed_interval': True,
                        'max_closed_interval': True,
                        'reversed_range': False,
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': None,
                        'mem_required': None,
                        'turbo': False,
                        'gpu_required': None,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_4.process[1], {
                    'language_id_score_filter': {
                        'lang': 'en',
                        'min_score': 0.5,
                        'text_key': 'text',
                        'image_key': 'images',
                        'image_bytes_key': 'image_bytes',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'system_key': 'system',
                        'instruction_key': 'instruction',
                        'prompt_key': 'prompt',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'min_closed_interval': True,
                        'max_closed_interval': True,
                        'reversed_range': False,
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': None,
                        'mem_required': None,
                        'turbo': False,
                        'gpu_required': None,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })

    def test_op_params_parsing(self):
        from jsonargparse import ArgumentParser
        from data_juicer.config.config import (sort_op_by_types_and_names, _collect_config_info_from_class_docs)
        from data_juicer.ops.base_op import OPERATORS

        base_class_params = {
            'text_key', 'image_key', 'image_bytes_key', 'audio_key', 'video_key', 'query_key', 'response_key',
            'history_key', 'accelerator', 'turbo', 'batch_size', 'num_proc', 'cpu_required', 'mem_required', 'work_dir',
        }

        parser = ArgumentParser(default_env=True, default_config_files=None)
        ops_sorted_by_types = sort_op_by_types_and_names(
            OPERATORS.modules.items())
        op_params = _collect_config_info_from_class_docs(ops_sorted_by_types,
                                                         parser)

        for op_name, params in op_params.items():
            for base_param in base_class_params:
                base_param_key = f'{op_name}.{base_param}'
                self.assertIn(base_param_key, params)

    def test_get_default_cfg(self):
        """Test getting default configuration from config_all.yaml"""
        # Get default config
        cfg = get_default_cfg()
        
        # Verify basic default values
        self.assertIsInstance(cfg, Namespace)
        
        # Test essential defaults
        self.assertEqual(cfg.executor_type, 'default')
        self.assertEqual(cfg.ray_address, 'auto')
        self.assertEqual(cfg.text_keys, 'text')
        self.assertEqual(cfg.add_suffix, False)
        self.assertEqual(cfg.export_path, './outputs/')
        self.assertEqual(cfg.suffixes, None)
        
        # Test default values are of correct type
        self.assertIsInstance(cfg.executor_type, str)
        self.assertIsInstance(cfg.add_suffix, bool)
        self.assertIsInstance(cfg.export_path, str)

    def test_cli_override(self):
        """Test that command line arguments correctly override YAML config values."""
        out = StringIO()
        with redirect_stdout(out):
            # Test with multiple operators and nested parameters
            cfg = init_configs(args=[
                '--config', test_yaml_path,
                '--language_id_score_filter.lang', 'en',
                '--language_id_score_filter.min_score', '0.5',
                '--whitespace_normalization_mapper.batch_size', '2000',
                '--remove_table_text_mapper.min_col', '3'
            ])

            # Verify language_id_score_filter overrides
            lang_filter = next(op for op in cfg.process if 'language_id_score_filter' in op)
            self.assertEqual(lang_filter['language_id_score_filter']['lang'], 'en')
            self.assertEqual(lang_filter['language_id_score_filter']['min_score'], 0.5)

            # Verify whitespace_normalization_mapper override
            whitespace_mapper = next(op for op in cfg.process if 'whitespace_normalization_mapper' in op)
            self.assertEqual(whitespace_mapper['whitespace_normalization_mapper']['batch_size'], 2000)

            # Verify remove_table_text_mapper override
            table_mapper = next(op for op in cfg.process if 'remove_table_text_mapper' in op)
            self.assertEqual(table_mapper['remove_table_text_mapper']['min_col'], 3)

            # Verify other parameters remain unchanged
            self.assertEqual(whitespace_mapper['whitespace_normalization_mapper']['text_key'], 'text')
            self.assertEqual(lang_filter['language_id_score_filter']['text_key'], 'text')

    def test_cli_override_with_equals(self):
        """Test command line overrides using equals sign syntax."""
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=[
                '--config', test_yaml_path,
                '--language_id_score_filter.lang=en',
                '--language_id_score_filter.min_score=0.5',
                '--whitespace_normalization_mapper.batch_size=2000'
            ])

            # Verify overrides
            lang_filter = next(op for op in cfg.process if 'language_id_score_filter' in op)
            self.assertEqual(lang_filter['language_id_score_filter']['lang'], 'en')
            self.assertEqual(lang_filter['language_id_score_filter']['min_score'], 0.5)

            whitespace_mapper = next(op for op in cfg.process if 'whitespace_normalization_mapper' in op)
            self.assertEqual(whitespace_mapper['whitespace_normalization_mapper']['batch_size'], 2000)

    def test_cli_override_invalid_value(self):
        """Test that invalid command line override values are properly caught."""
        out = StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            with self.assertRaises(SystemExit) as cm:
                init_configs(args=[
                    '--config', test_yaml_path,
                    '--language_id_score_filter.min_score', 'invalid'  # Should be a float
                ])
            self.assertEqual(cm.exception.code, 2)
            out_str = out.getvalue()
            self.assertIn('language_id_score_filter.min_score', out_str)
            self.assertIn('float', out_str)

    def test_validate_work_dir_config_valid_cases(self):
        """Test validate_work_dir_config with valid configurations."""
        valid_configs = [
            './outputs/my_project/{job_id}',
            '/data/experiments/{job_id}',
            'outputs/{job_id}',
            './{job_id}',
            'C:/data/projects/{job_id}',
            '/home/user/data/{job_id}',
            'relative/path/to/{job_id}',
            '{job_id}',  # Just job_id alone
        ]
        
        for work_dir in valid_configs:
            with self.subTest(work_dir=work_dir):
                # Should not raise any exception
                validate_work_dir_config(work_dir)

    def test_validate_work_dir_config_invalid_cases(self):
        """Test validate_work_dir_config with invalid configurations."""
        invalid_configs = [
            './outputs/{job_id}/results',
            './{job_id}/outputs/data',
            'outputs/{job_id}/intermediate/stuff',
            'data/{job_id}/processed/results',
            '/home/user/{job_id}/data/outputs',
            'C:/data/{job_id}/projects/results',
            'relative/{job_id}/path/to/data',
            'outputs/data/{job_id}/processed',
        ]
        
        for work_dir in invalid_configs:
            with self.subTest(work_dir=work_dir):
                with self.assertRaises(ValueError) as cm:
                    validate_work_dir_config(work_dir)
                
                # Check that the error message is helpful
                error_msg = str(cm.exception)
                self.assertIn('{job_id}', error_msg)
                self.assertIn('must be the last part', error_msg)
                self.assertIn('Expected format', error_msg)

    def test_validate_work_dir_config_no_job_id(self):
        """Test validate_work_dir_config with configurations that don't contain {job_id}."""
        no_job_id_configs = [
            './outputs/my_project',
            '/data/experiments',
            'outputs',
            './',
            'C:/data/projects',
            '/home/user/data',
            'relative/path/to',
            '',  # Empty string
        ]
        
        for work_dir in no_job_id_configs:
            with self.subTest(work_dir=work_dir):
                # Should not raise any exception
                validate_work_dir_config(work_dir)

    def test_resolve_job_id_with_placeholder(self):
        """Test resolve_job_id when {job_id} placeholder is present."""
        cfg = Namespace()
        cfg.work_dir = './outputs/my_project/{job_id}'
        cfg.export_path = './outputs/{job_id}/results.jsonl'
        
        # Should auto-generate job_id
        cfg = resolve_job_id(cfg)
        
        self.assertIsNotNone(cfg.job_id)
        self.assertFalse(cfg._user_provided_job_id)
        self.assertIsInstance(cfg.job_id, str)
        # Job ID should be in format: YYYYMMDD_HHMMSS_xxxxxx
        self.assertRegex(cfg.job_id, r'^\d{8}_\d{6}_[a-f0-9]{6}$')

    def test_resolve_job_id_without_placeholder(self):
        """Test resolve_job_id when no {job_id} placeholder is present."""
        cfg = Namespace()
        cfg.work_dir = './outputs/my_project'
        cfg.export_path = './outputs/results.jsonl'
        
        # Should still auto-generate job_id (fallback behavior)
        cfg = resolve_job_id(cfg)
        
        self.assertIsNotNone(cfg.job_id)
        self.assertFalse(cfg._user_provided_job_id)
        self.assertIsInstance(cfg.job_id, str)
        self.assertRegex(cfg.job_id, r'^\d{8}_\d{6}_[a-f0-9]{6}$')

    def test_resolve_job_id_user_provided(self):
        """Test resolve_job_id when user provides job_id."""
        cfg = Namespace()
        cfg.job_id = 'my_custom_job_123'
        cfg.work_dir = './outputs/my_project/{job_id}'
        
        cfg = resolve_job_id(cfg)
        
        self.assertEqual(cfg.job_id, 'my_custom_job_123')
        self.assertTrue(cfg._user_provided_job_id)

    def test_resolve_job_directories_with_job_id_at_end(self):
        """Test resolve_job_directories when {job_id} is at the end of work_dir."""
        cfg = Namespace()
        cfg.work_dir = './outputs/my_project/{job_id}'
        cfg.job_id = '20250804_143022_abc123'
        
        cfg = resolve_job_directories(cfg)
        
        # work_dir should be substituted
        self.assertEqual(cfg.work_dir, './outputs/my_project/20250804_143022_abc123')
        # job_dir should equal work_dir since job_id is at the end
        self.assertEqual(cfg.job_dir, './outputs/my_project/20250804_143022_abc123')
        # Other directories should be under job_dir
        self.assertEqual(cfg.event_log_dir, './outputs/my_project/20250804_143022_abc123/logs')
        self.assertEqual(cfg.checkpoint_dir, './outputs/my_project/20250804_143022_abc123/checkpoints')
        self.assertEqual(cfg.partition_dir, './outputs/my_project/20250804_143022_abc123/partitions')
        self.assertEqual(cfg.metadata_dir, './outputs/my_project/20250804_143022_abc123/metadata')
        self.assertEqual(cfg.results_dir, './outputs/my_project/20250804_143022_abc123/results')
        self.assertEqual(cfg.event_log_file, './outputs/my_project/20250804_143022_abc123/events.jsonl')

    def test_resolve_job_directories_without_job_id_placeholder(self):
        """Test resolve_job_directories when work_dir doesn't contain {job_id}."""
        cfg = Namespace()
        cfg.work_dir = './outputs/my_project'
        cfg.job_id = '20250804_143022_abc123'
        
        cfg = resolve_job_directories(cfg)
        
        # work_dir should remain unchanged
        self.assertEqual(cfg.work_dir, './outputs/my_project')
        # job_dir should be work_dir + job_id
        self.assertEqual(cfg.job_dir, './outputs/my_project/20250804_143022_abc123')
        # Other directories should be under job_dir
        self.assertEqual(cfg.event_log_dir, './outputs/my_project/20250804_143022_abc123/logs')
        self.assertEqual(cfg.checkpoint_dir, './outputs/my_project/20250804_143022_abc123/checkpoints')

    def test_resolve_job_directories_placeholder_substitution(self):
        """Test that placeholders are properly substituted in all relevant paths."""
        cfg = Namespace()
        cfg.work_dir = './outputs/{job_id}'
        cfg.export_path = '{work_dir}/results.jsonl'
        cfg.event_log_dir = '{work_dir}/logs'
        cfg.checkpoint_dir = '{work_dir}/checkpoints'
        cfg.partition_dir = '{work_dir}/partitions'
        cfg.job_id = '20250804_143022_abc123'
        
        cfg = resolve_job_directories(cfg)
        
        # All placeholders should be substituted
        self.assertEqual(cfg.work_dir, './outputs/20250804_143022_abc123')
        self.assertEqual(cfg.export_path, './outputs/20250804_143022_abc123/results.jsonl')
        # Note: event_log_dir is overridden by the system to use standard 'logs' directory
        self.assertEqual(cfg.event_log_dir, './outputs/20250804_143022_abc123/logs')
        self.assertEqual(cfg.checkpoint_dir, './outputs/20250804_143022_abc123/checkpoints')
        self.assertEqual(cfg.partition_dir, './outputs/20250804_143022_abc123/partitions')
        self.assertEqual(cfg.metadata_dir, './outputs/20250804_143022_abc123/metadata')
        self.assertEqual(cfg.results_dir, './outputs/20250804_143022_abc123/results')
        self.assertEqual(cfg.event_log_file, './outputs/20250804_143022_abc123/events.jsonl')

    def test_resolve_job_directories_missing_job_id(self):
        """Test resolve_job_directories when job_id is not set."""
        cfg = Namespace()
        cfg.work_dir = './outputs/my_project'
        
        with self.assertRaises(ValueError) as cm:
            resolve_job_directories(cfg)
        
        self.assertIn('job_id must be set', str(cm.exception))

    def test_resolve_job_directories_invalid_work_dir(self):
        """Test resolve_job_directories with invalid work_dir containing {job_id} in middle."""
        cfg = Namespace()
        cfg.work_dir = './outputs/{job_id}/results'
        cfg.job_id = '20250804_143022_abc123'
        
        with self.assertRaises(ValueError) as cm:
            resolve_job_directories(cfg)
        
        error_msg = str(cm.exception)
        self.assertIn('{job_id}', error_msg)
        self.assertIn('must be the last part', error_msg)

    def test_full_config_loading_with_job_id_placeholder(self):
        """Test full config loading with {job_id} placeholder in work_dir."""
        # Create a temporary config file
        config_data = {
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'work_dir': './outputs/test_project/{job_id}',
            'export_path': '{work_dir}/results.jsonl',
            'process': [
                {'whitespace_normalization_mapper': {'text_key': 'text'}}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            out = StringIO()
            with redirect_stdout(out):
                cfg = init_configs(args=['--config', temp_config_path])
                
                # Verify job_id was auto-generated
                self.assertIsNotNone(cfg.job_id)
                self.assertRegex(cfg.job_id, r'^\d{8}_\d{6}_[a-f0-9]{6}$')
                
                # Verify work_dir was substituted
                self.assertIn(cfg.job_id, cfg.work_dir)
                self.assertNotIn('{job_id}', cfg.work_dir)
                
                # Verify job_dir is correct
                self.assertEqual(cfg.job_dir, cfg.work_dir)
                
                # Verify export_path was substituted
                self.assertIn(cfg.job_id, cfg.export_path)
                self.assertNotIn('{work_dir}', cfg.export_path)
                
        finally:
            os.unlink(temp_config_path)

    def test_full_config_loading_without_job_id_placeholder(self):
        """Test full config loading without {job_id} placeholder in work_dir."""
        # Create a temporary config file
        config_data = {
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'work_dir': './outputs/test_project',
            'export_path': '{work_dir}/results.jsonl',
            'process': [
                {'whitespace_normalization_mapper': {'text_key': 'text'}}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            out = StringIO()
            with redirect_stdout(out):
                cfg = init_configs(args=['--config', temp_config_path])
                
                # Verify job_id was auto-generated
                self.assertIsNotNone(cfg.job_id)
                self.assertRegex(cfg.job_id, r'^\d{8}_\d{6}_[a-f0-9]{6}$')
                
                # Verify work_dir was not changed
                self.assertEqual(cfg.work_dir, './outputs/test_project')
                
                # Verify job_dir is work_dir + job_id
                self.assertEqual(cfg.job_dir, f'./outputs/test_project/{cfg.job_id}')
                
                # Note: When there's no {job_id} placeholder, {work_dir} in export_path is still substituted
                # The system substitutes {work_dir} with the actual work_dir value
                self.assertNotIn('{work_dir}', cfg.export_path)
                self.assertIn('./outputs/test_project', cfg.export_path)
                self.assertNotIn(cfg.job_id, cfg.export_path)
                
        finally:
            os.unlink(temp_config_path)

    def test_full_config_loading_invalid_work_dir(self):
        """Test full config loading with invalid work_dir containing {job_id} in middle."""
        # Create a temporary config file with invalid work_dir
        config_data = {
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'work_dir': './outputs/{job_id}/results',  # Invalid: {job_id} not at end
            'export_path': '{work_dir}/results.jsonl',
            'process': [
                {'whitespace_normalization_mapper': {'text_key': 'text'}}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            out = StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                with self.assertRaises(ValueError) as cm:
                    init_configs(args=['--config', temp_config_path])
                
                error_msg = str(cm.exception)
                self.assertIn('{job_id}', error_msg)
                self.assertIn('must be the last part', error_msg)
                
        finally:
            os.unlink(temp_config_path)

    def test_user_provided_job_id(self):
        """Test config loading with user-provided job_id."""
        # Create a temporary config file
        config_data = {
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'work_dir': './outputs/test_project/{job_id}',
            'export_path': '{work_dir}/results.jsonl',
            'process': [
                {'whitespace_normalization_mapper': {'text_key': 'text'}}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            out = StringIO()
            with redirect_stdout(out):
                # Test with user-provided job_id
                cfg = init_configs(args=[
                    '--config', temp_config_path,
                    '--job_id', 'my_custom_job_123'
                ])
                
                # Verify user-provided job_id was used
                self.assertEqual(cfg.job_id, 'my_custom_job_123')
                self.assertTrue(cfg._user_provided_job_id)
                
                # Verify work_dir was substituted
                self.assertEqual(cfg.work_dir, './outputs/test_project/my_custom_job_123')
                self.assertEqual(cfg.job_dir, './outputs/test_project/my_custom_job_123')
                
        finally:
            os.unlink(temp_config_path)

if __name__ == '__main__':
    unittest.main()
