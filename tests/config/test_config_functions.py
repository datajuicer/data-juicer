"""Tests for pure-logic utility functions in data_juicer/config/config.py.

Covers: timing_context, _generate_module_name, load_custom_operators,
sort_op_by_types_and_names, _parse_cli_to_config, _parse_value,
config_backup, validate_config_for_resumption, prepare_side_configs (edge),
namespace_to_arg_list, build_base_parser, namespace_to_arg_list,
save_cli_arguments, load_ops_with_stats_meta, prepare_cfgs_for_export,
display_config.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import yaml
from jsonargparse import Namespace

from data_juicer.config.config import (
    _generate_module_name,
    _parse_cli_to_config,
    _parse_value,
    build_base_parser,
    config_backup,
    display_config,
    load_custom_operators,
    load_ops_with_stats_meta,
    namespace_to_arg_list,
    prepare_cfgs_for_export,
    prepare_side_configs,
    resolve_job_directories,
    resolve_job_id,
    save_cli_arguments,
    sort_op_by_types_and_names,
    timing_context,
    validate_config_for_resumption,
    validate_work_dir_config,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TimingContextTest(DataJuicerTestCaseBase):

    def test_timing_context_runs_block(self):
        executed = False
        with timing_context("test block"):
            executed = True
        self.assertTrue(executed)

    def test_timing_context_measures_time(self):
        start = time.time()
        with timing_context("sleep"):
            time.sleep(0.05)
        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.04)


class GenerateModuleNameTest(DataJuicerTestCaseBase):

    def test_simple_path(self):
        self.assertEqual(_generate_module_name("/foo/bar/my_module.py"), "my_module")

    def test_nested_path(self):
        self.assertEqual(_generate_module_name("/a/b/c/ops.py"), "ops")

    def test_no_extension(self):
        self.assertEqual(_generate_module_name("/foo/bar/module"), "module")

    def test_basename_only(self):
        self.assertEqual(_generate_module_name("script.py"), "script")


class LoadCustomOperatorsTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up any modules we loaded
        for key in list(sys.modules.keys()):
            if key.startswith("_test_custom_op"):
                del sys.modules[key]
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_load_single_file(self):
        op_file = os.path.join(self.tmp_dir, "_test_custom_op_file.py")
        with open(op_file, "w") as f:
            f.write("LOADED = True\n")

        load_custom_operators([op_file])
        self.assertIn("_test_custom_op_file", sys.modules)
        self.assertTrue(sys.modules["_test_custom_op_file"].LOADED)

    def test_load_package(self):
        pkg_dir = os.path.join(self.tmp_dir, "_test_custom_op_pkg")
        os.makedirs(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("PKG_LOADED = True\n")

        load_custom_operators([pkg_dir])
        self.assertIn("_test_custom_op_pkg", sys.modules)

    def test_load_package_missing_init_raises(self):
        pkg_dir = os.path.join(self.tmp_dir, "_test_custom_op_noinit")
        os.makedirs(pkg_dir)
        # No __init__.py

        with self.assertRaises(ValueError) as ctx:
            load_custom_operators([pkg_dir])
        self.assertIn("__init__.py", str(ctx.exception))

    def test_load_nonexistent_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            load_custom_operators(["/nonexistent/path/to/module.py"])
        self.assertIn("neither a file nor a directory", str(ctx.exception))

    def test_load_duplicate_module_raises(self):
        op_file = os.path.join(self.tmp_dir, "_test_custom_op_dup.py")
        with open(op_file, "w") as f:
            f.write("X = 1\n")

        load_custom_operators([op_file])
        with self.assertRaises(RuntimeError) as ctx:
            load_custom_operators([op_file])
        self.assertIn("already loaded", str(ctx.exception))

    def test_load_file_with_syntax_error_raises(self):
        op_file = os.path.join(self.tmp_dir, "_test_custom_op_bad.py")
        with open(op_file, "w") as f:
            f.write("def broken(\n")  # syntax error

        with self.assertRaises(RuntimeError) as ctx:
            load_custom_operators([op_file])
        self.assertIn("Error loading", str(ctx.exception))


class SortOpByTypesAndNamesTest(DataJuicerTestCaseBase):

    def test_sorts_by_type_then_name(self):
        ops = [
            ("z_filter", "FilterClass"),
            ("a_mapper", "MapperClass"),
            ("b_deduplicator", "DedupClass"),
            ("c_mapper", "MapperClass2"),
            ("a_selector", "SelectorClass"),
            ("b_grouper", "GrouperClass"),
            ("a_aggregator", "AggClass"),
        ]
        result = sort_op_by_types_and_names(ops)
        names = [name for name, _ in result]
        self.assertEqual(names, [
            "a_mapper", "c_mapper",          # mappers first, sorted
            "z_filter",                       # filters
            "b_deduplicator",                 # deduplicators
            "a_selector",                     # selectors
            "b_grouper",                      # groupers
            "a_aggregator",                   # aggregators
        ])

    def test_empty_list(self):
        result = sort_op_by_types_and_names([])
        self.assertEqual(result, [])

    def test_single_type(self):
        ops = [("b_mapper", "B"), ("a_mapper", "A")]
        result = sort_op_by_types_and_names(ops)
        self.assertEqual([n for n, _ in result], ["a_mapper", "b_mapper"])


class ParseValueTest(DataJuicerTestCaseBase):

    def test_parse_true(self):
        self.assertIs(_parse_value("true"), True)
        self.assertIs(_parse_value("True"), True)
        self.assertIs(_parse_value("TRUE"), True)

    def test_parse_false(self):
        self.assertIs(_parse_value("false"), False)
        self.assertIs(_parse_value("False"), False)

    def test_parse_integer(self):
        self.assertEqual(_parse_value("42"), 42)
        self.assertIsInstance(_parse_value("42"), int)

    def test_parse_negative_integer(self):
        self.assertEqual(_parse_value("-7"), -7)

    def test_parse_float(self):
        self.assertAlmostEqual(_parse_value("3.14"), 3.14)
        self.assertIsInstance(_parse_value("3.14"), float)

    def test_parse_float_scientific(self):
        self.assertAlmostEqual(_parse_value("1e-3"), 0.001)

    def test_parse_string(self):
        self.assertEqual(_parse_value("hello"), "hello")
        self.assertIsInstance(_parse_value("hello"), str)

    def test_parse_path(self):
        self.assertEqual(_parse_value("/path/to/file.yaml"), "/path/to/file.yaml")


class ParseCliToConfigTest(DataJuicerTestCaseBase):

    def test_empty_args(self):
        self.assertEqual(_parse_cli_to_config([]), {})

    def test_key_value_pair(self):
        result = _parse_cli_to_config(["--name", "test"])
        self.assertEqual(result.get("name"), "test")

    def test_key_equals_value(self):
        result = _parse_cli_to_config(["--count=5"])
        self.assertEqual(result.get("count"), 5)

    def test_boolean_flag(self):
        result = _parse_cli_to_config(["--debug"])
        self.assertEqual(result.get("debug"), True)

    def test_multiple_values(self):
        result = _parse_cli_to_config(["--items", "a", "b", "c"])
        self.assertEqual(result.get("items"), ["a", "b", "c"])

    def test_mixed_args(self):
        result = _parse_cli_to_config([
            "--name", "test",
            "--count=10",
            "--verbose",
        ])
        self.assertEqual(result.get("name"), "test")
        self.assertEqual(result.get("count"), 10)
        self.assertEqual(result.get("verbose"), True)


class ConfigBackupTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmp_dir, "work")
        os.makedirs(self.work_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_backup_copies_config_file(self):
        cfg_file = os.path.join(self.tmp_dir, "test.yaml")
        with open(cfg_file, "w") as f:
            f.write("key: value\n")

        target_path = os.path.join(self.work_dir, "test.yaml")
        cfg = Namespace(
            config=[cfg_file],
            work_dir=self.work_dir,
            backed_up_config_path=target_path,
            _original_args=[],
        )

        config_backup(cfg)
        self.assertTrue(os.path.exists(target_path))
        with open(target_path) as f:
            self.assertEqual(f.read(), "key: value\n")

    def test_backup_skips_if_exists(self):
        cfg_file = os.path.join(self.tmp_dir, "test.yaml")
        with open(cfg_file, "w") as f:
            f.write("original\n")

        target_path = os.path.join(self.work_dir, "test.yaml")
        with open(target_path, "w") as f:
            f.write("already_there\n")

        cfg = Namespace(
            config=[cfg_file],
            work_dir=self.work_dir,
            backed_up_config_path=target_path,
            _original_args=[],
        )
        config_backup(cfg)

        with open(target_path) as f:
            self.assertEqual(f.read(), "already_there\n")

    def test_backup_no_config_does_nothing(self):
        cfg = Namespace(config=None, work_dir=self.work_dir)
        # Should not raise
        config_backup(cfg)

    def test_backup_fallback_without_backed_up_path(self):
        cfg_file = os.path.join(self.tmp_dir, "fallback.yaml")
        with open(cfg_file, "w") as f:
            f.write("fallback_data\n")

        cfg = Namespace(
            config=[cfg_file],
            work_dir=self.work_dir,
            _original_args=[],
        )

        config_backup(cfg)
        expected_path = os.path.join(self.work_dir, "fallback.yaml")
        self.assertTrue(os.path.exists(expected_path))


class ValidateConfigForResumptionTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmp_dir, "work")
        os.makedirs(self.work_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_matching_configs_returns_true(self):
        config_content = "dataset_path: ./data\nprocess:\n  - clean_mapper:\n"
        # Write "original" config in work_dir
        orig_cfg = os.path.join(self.work_dir, "config.yaml")
        with open(orig_cfg, "w") as f:
            f.write(config_content)

        # Write a matching CLI args file with specific args
        cli_yaml = os.path.join(self.work_dir, "cli.yaml")
        with open(cli_yaml, "w") as f:
            yaml.dump({"arguments": ["--np", "4"]}, f)

        # Write "current" config somewhere else (same content)
        cur_cfg = os.path.join(self.tmp_dir, "current.yaml")
        with open(cur_cfg, "w") as f:
            f.write(config_content)

        cfg = Namespace(config=[cur_cfg])
        # Pass same CLI args as saved in cli.yaml
        result = validate_config_for_resumption(
            cfg, self.work_dir, original_args=["--np", "4"]
        )
        self.assertTrue(result)
        self.assertTrue(cfg._same_yaml_config)

    def test_mismatched_configs_returns_false(self):
        orig_cfg = os.path.join(self.work_dir, "config.yaml")
        with open(orig_cfg, "w") as f:
            f.write("dataset_path: ./old_data\n")

        cur_cfg = os.path.join(self.tmp_dir, "current.yaml")
        with open(cur_cfg, "w") as f:
            f.write("dataset_path: ./new_data\n")

        cfg = Namespace(config=[cur_cfg])
        result = validate_config_for_resumption(cfg, self.work_dir)
        self.assertFalse(result)
        self.assertFalse(cfg._same_yaml_config)

    def test_no_config_files_returns_false(self):
        empty_dir = os.path.join(self.tmp_dir, "empty")
        os.makedirs(empty_dir)

        cfg = Namespace()
        result = validate_config_for_resumption(cfg, empty_dir)
        self.assertFalse(result)

    def test_no_current_config_returns_false(self):
        orig_cfg = os.path.join(self.work_dir, "config.yaml")
        with open(orig_cfg, "w") as f:
            f.write("x: 1\n")

        cfg = Namespace()  # no config attribute
        result = validate_config_for_resumption(cfg, self.work_dir)
        self.assertFalse(result)

    def test_cli_yaml_comparison(self):
        config_content = "dataset_path: ./data\n"
        orig_cfg = os.path.join(self.work_dir, "config.yaml")
        with open(orig_cfg, "w") as f:
            f.write(config_content)

        # Write CLI args
        cli_yaml = os.path.join(self.work_dir, "cli.yaml")
        with open(cli_yaml, "w") as f:
            yaml.dump({"arguments": ["--np", "4"]}, f)

        cur_cfg = os.path.join(self.tmp_dir, "current.yaml")
        with open(cur_cfg, "w") as f:
            f.write(config_content)

        cfg = Namespace(config=[cur_cfg])
        # Pass different CLI args
        result = validate_config_for_resumption(
            cfg, self.work_dir, original_args=["--np", "8"]
        )
        self.assertFalse(result)


class PrepareSideConfigsEdgeCasesTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_json_file(self):
        cfg_file = os.path.join(self.tmp_dir, "config.json")
        with open(cfg_file, "w") as f:
            json.dump({"key": "value"}, f)
        result = prepare_side_configs(cfg_file)
        self.assertEqual(result, {"key": "value"})

    def test_unsupported_extension_raises(self):
        cfg_file = os.path.join(self.tmp_dir, "config.toml")
        with open(cfg_file, "w") as f:
            f.write("")
        with self.assertRaises(TypeError):
            prepare_side_configs(cfg_file)

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            prepare_side_configs(12345)

    def test_namespace_input(self):
        ns = Namespace(a=1, b=2)
        result = prepare_side_configs(ns)
        self.assertEqual(result, ns)


class ResolveJobIdTest(DataJuicerTestCaseBase):
    """Test resolve_job_id: auto-generates or preserves user-provided job_id."""

    def test_auto_generates_job_id_when_missing(self):
        cfg = Namespace()
        result = resolve_job_id(cfg)
        self.assertTrue(hasattr(result, "job_id"))
        self.assertIsInstance(result.job_id, str)
        self.assertGreater(len(result.job_id), 0)
        self.assertFalse(result._user_provided_job_id)

    def test_auto_generated_format(self):
        """Auto-generated job_id should be timestamp_hash format."""
        cfg = Namespace()
        resolve_job_id(cfg)
        parts = cfg.job_id.split("_")
        # Format: YYYYMMDD_HHMMSS_hexhash
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[0]), 8)  # date
        self.assertEqual(len(parts[1]), 6)  # time
        self.assertEqual(len(parts[2]), 6)  # hex hash

    def test_preserves_user_provided_job_id(self):
        cfg = Namespace(job_id="my_custom_job")
        result = resolve_job_id(cfg)
        self.assertEqual(result.job_id, "my_custom_job")
        self.assertTrue(result._user_provided_job_id)

    def test_uniqueness(self):
        """Two calls should produce different job_ids."""
        cfg1 = Namespace()
        cfg2 = Namespace()
        resolve_job_id(cfg1)
        resolve_job_id(cfg2)
        self.assertNotEqual(cfg1.job_id, cfg2.job_id)


class ValidateWorkDirConfigTest(DataJuicerTestCaseBase):
    """Test validate_work_dir_config: ensures {job_id} is at end of path."""

    def test_valid_job_id_at_end(self):
        # Should not raise
        validate_work_dir_config("./outputs/project/{job_id}")

    def test_valid_absolute_path(self):
        validate_work_dir_config("/data/experiments/{job_id}")

    def test_valid_no_job_id_placeholder(self):
        """Paths without {job_id} are valid (job_id appended later)."""
        validate_work_dir_config("./outputs/project")

    def test_invalid_job_id_not_at_end(self):
        with self.assertRaises(ValueError) as ctx:
            validate_work_dir_config("./outputs/{job_id}/results")
        self.assertIn("last part", str(ctx.exception))

    def test_invalid_job_id_in_middle(self):
        with self.assertRaises(ValueError) as ctx:
            validate_work_dir_config("./{job_id}/outputs/data")
        self.assertIn("last part", str(ctx.exception))

    def test_trailing_slash_still_valid(self):
        validate_work_dir_config("./outputs/{job_id}/")


class ResolveJobDirectoriesTest(DataJuicerTestCaseBase):
    """Test resolve_job_directories: sets up all job-specific directories."""

    def test_basic_directory_resolution(self):
        cfg = Namespace(
            work_dir="./outputs/project",
            job_id="test_job_123",
            config=["test.yaml"],
            event_log_dir=None,
            checkpoint_dir=None,
            partition_dir=None,
        )
        result = resolve_job_directories(cfg)
        self.assertTrue(result.work_dir.endswith("test_job_123"))
        self.assertEqual(result.event_log_dir,
                         os.path.join(result.work_dir, "logs"))
        self.assertEqual(result.checkpoint_dir,
                         os.path.join(result.work_dir, "checkpoints"))
        self.assertEqual(result.partition_dir,
                         os.path.join(result.work_dir, "partitions"))
        self.assertEqual(result.metadata_dir,
                         os.path.join(result.work_dir, "metadata"))
        self.assertEqual(result.results_dir,
                         os.path.join(result.work_dir, "results"))

    def test_job_id_placeholder_substitution(self):
        cfg = Namespace(
            work_dir="./outputs/{job_id}",
            job_id="abc123",
            config=["cfg.yaml"],
            event_log_dir=None,
            checkpoint_dir=None,
            partition_dir=None,
        )
        result = resolve_job_directories(cfg)
        self.assertTrue(result.work_dir.endswith("abc123"))
        self.assertNotIn("{job_id}", result.work_dir)

    def test_work_dir_placeholder_in_other_paths(self):
        cfg = Namespace(
            work_dir="./outputs/proj",
            job_id="j1",
            config=["c.yaml"],
            event_log_dir="{work_dir}/my_logs",
            checkpoint_dir=None,
            partition_dir=None,
        )
        result = resolve_job_directories(cfg)
        self.assertIn("my_logs", result.event_log_dir)
        self.assertNotIn("{work_dir}", result.event_log_dir)

    def test_no_config_uses_default_backup_path(self):
        cfg = Namespace(
            work_dir="./outputs",
            job_id="j2",
            config=None,
            event_log_dir=None,
            checkpoint_dir=None,
            partition_dir=None,
        )
        result = resolve_job_directories(cfg)
        self.assertTrue(result.backed_up_config_path.endswith("config.yaml"))

    def test_missing_job_id_raises(self):
        cfg = Namespace(
            work_dir="./outputs",
            job_id="",
            config=None,
            event_log_dir=None,
            checkpoint_dir=None,
            partition_dir=None,
        )
        with self.assertRaises(ValueError):
            resolve_job_directories(cfg)

    def test_custom_dirs_preserved(self):
        """Custom event_log_dir/checkpoint_dir should not be overridden."""
        cfg = Namespace(
            work_dir="./outputs",
            job_id="j3",
            config=["c.yaml"],
            event_log_dir="/custom/logs",
            checkpoint_dir="/custom/ckpts",
            partition_dir="/custom/parts",
        )
        result = resolve_job_directories(cfg)
        self.assertEqual(result.event_log_dir, "/custom/logs")
        self.assertEqual(result.checkpoint_dir, "/custom/ckpts")
        self.assertEqual(result.partition_dir, "/custom/parts")

    def test_event_log_file_set(self):
        cfg = Namespace(
            work_dir="./outputs",
            job_id="j4",
            config=["c.yaml"],
            event_log_dir=None,
            checkpoint_dir=None,
            partition_dir=None,
        )
        result = resolve_job_directories(cfg)
        self.assertTrue(result.event_log_file.endswith("events.jsonl"))
        self.assertTrue(result.job_summary_file.endswith("job_summary.json"))


class BuildBaseParserTest(DataJuicerTestCaseBase):

    def test_has_required_args(self):
        parser = build_base_parser()
        self.assertIsNotNone(parser)
        action_dests = {a.dest for a in parser._actions}
        self.assertIn('project_name', action_dests)
        self.assertIn('executor_type', action_dests)
        self.assertIn('dataset_path', action_dests)

    def test_executor_type_choices(self):
        parser = build_base_parser()
        for action in parser._actions:
            if action.dest == 'executor_type':
                self.assertIn('default', action.choices)
                self.assertIn('ray', action.choices)
                self.assertIn('ray_partitioned', action.choices)
                break

    def test_config_and_auto_mutually_exclusive(self):
        parser = build_base_parser()
        groups = [g for g in parser._mutually_exclusive_groups]
        self.assertTrue(len(groups) >= 1)


class NamespaceToArgListTest(DataJuicerTestCaseBase):

    def test_flat_namespace(self):
        ns = Namespace(a=1, b='hello')
        result = namespace_to_arg_list(ns)
        self.assertIn('--a=1', result)
        self.assertIn('--b=hello', result)

    def test_skips_none_values(self):
        ns = Namespace(a=1, b=None)
        result = namespace_to_arg_list(ns)
        self.assertIn('--a=1', result)
        self.assertNotIn('--b=None', result)

    def test_excludes(self):
        ns = Namespace(a=1, b=2, c=3)
        result = namespace_to_arg_list(ns, excludes=['b'])
        keys = [r.split('=')[0] for r in result]
        self.assertNotIn('--b', keys)

    def test_includes(self):
        ns = Namespace(a=1, b=2, c=3)
        result = namespace_to_arg_list(ns, includes=['a', 'c'])
        keys = [r.split('=')[0] for r in result]
        self.assertIn('--a', keys)
        self.assertIn('--c', keys)
        self.assertNotIn('--b', keys)

    def test_exclude_prefixes(self):
        ns = Namespace(a=1, my_op=Namespace(x=10, y=20))
        result = namespace_to_arg_list(ns, exclude_prefixes=('my_op',))
        keys = [r.split('=')[0] for r in result]
        self.assertIn('--a', keys)
        self.assertNotIn('--my_op.x', keys)
        self.assertNotIn('--my_op.y', keys)

    def test_nested_namespace(self):
        ns = Namespace(top=Namespace(inner=42))
        result = namespace_to_arg_list(ns)
        self.assertIn('--top.inner=42', result)

    def test_process_key_uses_json(self):
        ns = Namespace(process=[{'op': {'k': 'v'}}])
        result = namespace_to_arg_list(ns)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].startswith('--process='))
        parsed = json.loads(result[0].split('=', 1)[1])
        self.assertEqual(parsed, [{'op': {'k': 'v'}}])

    def test_prefix(self):
        ns = Namespace(x=5)
        result = namespace_to_arg_list(ns, prefix='sub.')
        self.assertIn('--sub.x=5', result)


class SaveCliArgumentsTest(DataJuicerTestCaseBase):

    def test_saves_to_cli_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Namespace(
                work_dir=tmpdir,
                _original_args=['--config', 'test.yaml', '--np', '4'],
            )
            save_cli_arguments(cfg)
            cli_path = os.path.join(tmpdir, 'cli.yaml')
            self.assertTrue(os.path.exists(cli_path))
            with open(cli_path) as f:
                data = yaml.safe_load(f)
            self.assertEqual(data['arguments'],
                             ['--config', 'test.yaml', '--np', '4'])

    def test_no_work_dir_does_nothing(self):
        cfg = Namespace(_original_args=['--x', '1'])
        save_cli_arguments(cfg)

    def test_no_args_saves_sys_argv_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Namespace(work_dir=tmpdir)
            save_cli_arguments(cfg)
            cli_path = os.path.join(tmpdir, 'cli.yaml')
            self.assertTrue(os.path.exists(cli_path))
            with open(cli_path) as f:
                data = yaml.safe_load(f)
            self.assertIn('arguments', data)
            self.assertIsInstance(data['arguments'], list)


class LoadOpsWithStatsMetaTest(DataJuicerTestCaseBase):

    def test_returns_known_ops(self):
        result = load_ops_with_stats_meta()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        names = [list(d.keys())[0] for d in result]
        for expected in [
            'alphanumeric_filter',
            'audio_duration_filter',
            'text_length_filter',
        ]:
            self.assertIn(expected, names,
                          f'{expected} should be in ops with stats meta')

    def test_each_item_is_single_key_dict(self):
        result = load_ops_with_stats_meta()
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertEqual(len(item), 1)
            op_name = list(item.keys())[0]
            self.assertIsInstance(op_name, str)
            self.assertGreater(len(op_name), 0)


class PrepareCfgsForExportTest(DataJuicerTestCaseBase):

    def test_converts_config_paths_to_str(self):
        from pathlib import Path
        cfg = {'config': [Path('/a/b.yaml')], 'process': []}
        result = prepare_cfgs_for_export(cfg)
        self.assertEqual(result['config'], ['/a/b.yaml'])

    def test_removes_op_keys(self):
        from data_juicer.ops.base_op import OPERATORS
        some_op = next(iter(OPERATORS.modules.keys()))
        cfg = {
            'process': [{'some_filter': {}}],
            some_op: {'param': 'val'},
        }
        result = prepare_cfgs_for_export(cfg)
        self.assertNotIn(some_op, result)
        self.assertIn('process', result)

    def test_no_config_key(self):
        cfg = {'process': [], 'other': 'val'}
        result = prepare_cfgs_for_export(cfg)
        self.assertEqual(result['other'], 'val')

    def test_mutates_in_place(self):
        from pathlib import Path
        cfg = {'config': [Path('/a/b.yaml')], 'process': []}
        result = prepare_cfgs_for_export(cfg)
        self.assertIs(result, cfg)


class DisplayConfigTest(DataJuicerTestCaseBase):

    def test_output_contains_config_keys(self):
        cfg = Namespace(
            project_name='test',
            dataset_path='./data',
            process=[],
        )
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            display_config(cfg)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertIn('project_name', output)
        self.assertIn('dataset_path', output)


if __name__ == "__main__":
    unittest.main()
