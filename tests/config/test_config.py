import os
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from jsonargparse import Namespace, namespace_to_dict

from data_juicer.config import init_configs, get_default_cfg, update_op_attr, export_config, merge_config, prepare_side_configs
from data_juicer.ops import load_ops
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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
                        'audio_special_token': '<__dj__audio>',
                        'eoc_special_token': '<|__dj__eoc|>',
                        'image_special_token': '<__dj__image>',
                        'video_special_token': '<__dj__video>',
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

    def test_auto_mode(self):
        out = StringIO()
        with redirect_stdout(out):
            # not in analyzer
            with self.assertRaises(NotImplementedError):
                init_configs(args=[
                    '--auto',
                ], which_entry="NoneAnalyzerClass")

            # in analyzer
            from data_juicer.core import Analyzer
            cfg = init_configs(args=[
                '--config', test_yaml_path,
            ])
            analyzer = Analyzer(cfg)

            cfg_auto = init_configs(args=[
                '--auto',
            ], which_entry=analyzer)
            self.assertTrue(cfg_auto.auto)
            self.assertGreater(len(cfg_auto.process), 0)

    def test_debug_mode(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=[
                '--config', test_yaml_path,
                '--debug',
            ])
            self.assertEqual(cfg.debug, True)

    def test_different_np(self):
        out = StringIO()
        with redirect_stdout(out):
            # too many
            cfg = init_configs(args=[
                '--config', test_yaml_path,
                '--np', f'{os.cpu_count() + 100}',
            ])
            self.assertEqual(cfg.np, os.cpu_count())

    def test_op_fusion(self):
        out = StringIO()
        with redirect_stdout(out):
            with self.assertRaises(NotImplementedError):
                init_configs(args=[
                    '--config', test_yaml_path,
                    '--op_fusion', 'True',
                    '--fusion_strategy', 'invalid',
                ])

    def test_multiple_text_keys(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=[
                '--config', test_text_keys_yaml_path,
            ])
            self.assertEqual(cfg.text_keys, ['text1', 'text2'])
            first_op = cfg.process[0]
            first_op_name = list(first_op.keys())[0]
            self.assertEqual(first_op[first_op_name]['text_key'], 'text1')

    def test_update_op_attr(self):
        ori_ops = [
            {'text_mapper': {'text_key': 'text'}},
            {'language_id_score_filter': {'lang': 'en', 'min_score': 0.5}},
            {'whitespace_normalization_mapper': {'batch_size': 2000}},
            {'remove_table_text_mapper': {'min_col': 3}}
        ]
        op_attrs = {
            'text_key': 'text2'
        }
        res_ops = update_op_attr(ori_ops, op_attrs)
        self.assertEqual(res_ops, [
            {'text_mapper': {'text_key': 'text'}},
            {'language_id_score_filter': {'lang': 'en', 'min_score': 0.5, 'text_key': 'text2'}},
            {'whitespace_normalization_mapper': {'batch_size': 2000, 'text_key': 'text2'}},
            {'remove_table_text_mapper': {'min_col': 3, 'text_key': 'text2'}}
        ])

        self.assertEqual(update_op_attr(ori_ops, None), ori_ops)

    def test_same_ops(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=[
                '--config', test_same_ops_yaml_path,
            ])
            op_name_groups = {}
            for op_cfg in cfg.process:
                op_name = list(op_cfg.keys())[0]
                op_name_groups.setdefault(op_name, []).append(op_cfg)
            self.assertEqual(len(op_name_groups['language_id_score_filter']), 2)
            self.assertEqual(op_name_groups['language_id_score_filter'][0]['language_id_score_filter']['lang'], 'zh')
            self.assertEqual(op_name_groups['language_id_score_filter'][1]['language_id_score_filter']['lang'], 'en')

    def test_export_config(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=[
                '--config', test_yaml_path,
            ])
            export_path = os.path.join(self.tmp_dir, 'export_config.json')
            export_config(cfg, export_path, format='json', skip_none=False)
            self.assertTrue(os.path.exists(export_path))
            import json
            exported_json = json.load(open(export_path))
            if isinstance(cfg, Namespace):
                cfg = namespace_to_dict(cfg)
            for key in exported_json:
                self.assertIn(key, cfg)
                self.assertEqual(exported_json[key], cfg[key])

    def test_merge_config(self):
        ori_cfg = Namespace({
            'export_path': os.path.join(self.tmp_dir, 'res.jsonl'),
            'work_dir': self.tmp_dir,
            'process': [
                {'text_mapper': {'text_key': 'text'}},
                {'language_id_score_filter': {'lang': 'en', 'min_score': 0.5}},
                {'whitespace_normalization_mapper': {'batch_size': 2000}},
                {'remove_table_text_mapper': {'min_col': 3}}
            ]
        })
        new_cfg = Namespace({
            'process': [
                {'text_mapper': {'text_key': 'text2'}},
                {'language_id_score_filter': {'lang': 'zh'}},
                {'whitespace_normalization_mapper': {'batch_size': 2000}},
                {'remove_table_text_mapper': {'min_col': 3}}
            ]
        })
        res_cfg = merge_config(ori_cfg, new_cfg)
        for i, op in enumerate(res_cfg.process):
            op_name = list(op.keys())[0]
            op_cfg = op[op_name]
            ori_op_cfg = ori_cfg.process[i][op_name]
            new_op_cfg = new_cfg.process[i][op_name]
            for key in op_cfg:
                if key in ori_op_cfg:
                    self.assertEqual(op_cfg[key], ori_op_cfg[key])
                else:
                    self.assertEqual(op_cfg[key], new_op_cfg[key])

    def test_prepare_side_configs(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = prepare_side_configs(test_yaml_path)
            self.assertEqual(cfg['np'], 4)

            cfg = prepare_side_configs({'key': 'value'})
            self.assertEqual(cfg['key'], 'value')

            with self.assertRaises(TypeError):
                prepare_side_configs(1)

            with self.assertRaises(TypeError):
                prepare_side_configs('xxx.txt')

    def test_cli_custom_operator_paths(self):
        """Test arg custom_operator_paths"""

        new_ops_dir = f'{WORKDIR}/custom_ops'
        new_op_path1 = os.path.join(new_ops_dir, 'new_op1.py')
        new_op_path2 = os.path.join(new_ops_dir, 'test_dir_module/new_op2.py')
        os.makedirs(os.path.dirname(new_op_path1), exist_ok=True)
        os.makedirs(os.path.dirname(new_op_path2), exist_ok=True)

        with open(new_op_path1, 'w') as f:
            f.write("""
from data_juicer.ops.base_op import OPERATORS, Mapper
                                              
@OPERATORS.register_module('custom_mapper1')
class CustomMapper1(Mapper):
    def process_single(self, data):
        return data
""")
        with open(new_op_path2, 'w') as f:
            f.write("""
from data_juicer.ops.base_op import OPERATORS, Mapper
                                              
@OPERATORS.register_module('custom_mapper2')
class CustomMapper2(Mapper):
    def process_single(self, data):
        return data
""")
            
        with open(os.path.join(os.path.dirname(new_op_path2), '__init__.py'), 'w') as f:
            f.write("""
from . import new_op2
""")

        init_configs(args=[
            '--config', test_yaml_path,
            '--custom-operator-paths', new_op_path1, os.path.dirname(new_op_path2)
        ])
        from data_juicer.ops.base_op import OPERATORS
        self.assertIn('custom_mapper1', list(OPERATORS.modules.keys()))
        self.assertIn('custom_mapper2', list(OPERATORS.modules.keys()))
        
        OPERATORS.modules.pop('custom_mapper1')
        OPERATORS.modules.pop('custom_mapper2')

    @TEST_TAG("ray")
    def test_cli_custom_operator_paths_ray(self):
        """Test arg custom_operator_paths"""

        new_ops_dir = f'{WORKDIR}/custom_ops'
        new_op_path1 = os.path.join(new_ops_dir, 'new_op3.py')
        new_op_path2 = os.path.join(new_ops_dir, 'test_dir_module2/new_op4.py')
        os.makedirs(os.path.dirname(new_op_path1), exist_ok=True)
        os.makedirs(os.path.dirname(new_op_path2), exist_ok=True)
        tmp_yaml_path = f'{WORKDIR}/demo_4_test_ray_tmp.yaml'
        
        with open(tmp_yaml_path, 'w') as f:
            f.write("""
project_name: 'test_demo'
dataset_path: './demos/data/demo-dataset.jsonl'
executor_type: ray
ray_address: auto
export_path: './outputs/demo/demo-processed.parquet'
process:
  - custom_mapper3:
  - custom_mapper4:
""")

        with open(new_op_path1, 'w') as f:
            f.write("""
from data_juicer.ops.base_op import OPERATORS, Mapper
                                              
@OPERATORS.register_module('custom_mapper3')
class CustomMapper3(Mapper):
    def process_single(self, data):
        data['text'] += 'tag1'
        return data
""")
        with open(new_op_path2, 'w') as f:
            f.write("""
from data_juicer.ops.base_op import OPERATORS, Mapper
                                              
@OPERATORS.register_module('custom_mapper4')
class CustomMapper4(Mapper):
    def process_single(self, data):
        data['text'] += 'tag2'
        return data
""")
            
        with open(os.path.join(os.path.dirname(new_op_path2), '__init__.py'), 'w') as f:
            f.write("""
from . import new_op4
""")

        cfg = init_configs(args=[
            '--config', tmp_yaml_path,
            '--custom-operator-paths', new_op_path1, os.path.dirname(new_op_path2)
        ])
        from data_juicer.core.executor.ray_executor import RayExecutor

        executor = RayExecutor(cfg)
        ds = executor.run()
        for data in ds.to_list():
            self.assertTrue(data['text'].endswith('tag1tag2'))


if __name__ == '__main__':
    unittest.main()
