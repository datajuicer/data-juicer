import json
import unittest
import os
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


class RayDatasetFuncsTest(DataJuicerTestCaseBase):

    def setUp(self):
        """Set up test data"""
        super().setUp()

        from data_juicer.core.data.ray_dataset import (
            get_abs_path,
            convert_to_absolute_paths,
            set_dataset_to_absolute_path,
            preprocess_dataset,
        )

        self.get_abs_path = get_abs_path
        self.convert_to_absolute_paths = convert_to_absolute_paths
        self.set_dataset_to_absolute_path = set_dataset_to_absolute_path
        self.preprocess_dataset = preprocess_dataset

        self.test_data = [
            {
                "text": "Hello",
                "images": ["image1.jpg", "subdir/image2.png"],
                "videos": ["video1.mp4"],
                "audios": ["audio1.wav", "audio2.mp3"],
            },
            {"text": "World", "images": ["image3.jpg"], "videos": ["subdir/video2.mp4"], "audios": ["audio3.wav"]},
        ]

        self.tmp_dir = "tmp/test_ray_executor/"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil

            shutil.rmtree(self.tmp_dir)

    def _touch_a_file(self, path):
        """Create a file at the given path"""
        with open(path, "w") as f:
            f.write("test")

    @TEST_TAG("ray")
    def test_get_abs_path_local(self):
        """Test get_abs_path function for local paths"""
        import os

        # Test relative path
        dataset_dir = self.tmp_dir
        rel_path = "image.jpg"
        full_path = os.path.join(dataset_dir, rel_path)
        self._touch_a_file(full_path)
        expected = os.path.abspath(os.path.join(dataset_dir, rel_path))
        result = self.get_abs_path(rel_path, dataset_dir)
        self.assertEqual(result, expected)

        # Test absolute path (should remain unchanged)
        abs_path = os.path.abspath(full_path)
        result = self.get_abs_path(abs_path, dataset_dir)
        self.assertEqual(result, abs_path)

        # Test remote path (should remain unchanged)
        remote_path = "http://bucket/file.jpg"
        result = self.get_abs_path(remote_path, dataset_dir)
        self.assertEqual(result, remote_path)

    @TEST_TAG("ray")
    def test_convert_to_absolute_paths(self):
        """Test convert_to_absolute_paths function"""
        import pyarrow as pa

        # Create a PyArrow table similar to what would be passed to the function

        sample_data = {
            "images": [["image1.jpg", "subdir/image2.png"], ["image3.jpg"]],
            "videos": [["video1.mp4"], ["subdir/video2.mp4"]],
        }

        for key, value_list in sample_data.items():
            for sub_list in value_list:
                for path in sub_list:
                    full_path = os.path.join(self.tmp_dir, path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    self._touch_a_file(full_path)

        table = pa.Table.from_pydict(sample_data)

        dataset_dir = self.tmp_dir
        path_keys = ["images", "videos"]

        result_table = self.convert_to_absolute_paths(table, dataset_dir, path_keys)

        result_dict = result_table.to_pydict()

        # Check that images were converted to absolute paths
        self.assertTrue(result_dict["images"][0][0].startswith("/"))
        self.assertTrue(result_dict["images"][0][1].startswith("/"))
        self.assertTrue(result_dict["images"][1][0].startswith("/"))

        # Check that videos were converted to absolute paths
        self.assertTrue(result_dict["videos"][0][0].startswith("/"))
        self.assertTrue(result_dict["videos"][1][0].startswith("/"))

    @TEST_TAG("ray")
    def test_get_abs_path_with_nonexistent_local_path(self):
        """Test get_abs_path when local path doesn't exist"""
        # When the joined path doesn't exist, it should return the current path
        dataset_dir = "./nonexistent_dataset"
        path = "existing_file.txt"
        tgt_path = os.path.join(dataset_dir, path)
        non_tgt_path = os.path.abspath(tgt_path)
        result = self.get_abs_path(path, dataset_dir)
        self.assertEqual(result, tgt_path)
        self.assertNotEqual(result, non_tgt_path)

    @TEST_TAG("ray")
    def test_read_json_stream(self):
        """Test reading JSON stream with RayDataset"""
        from data_juicer.core.data.ray_dataset import read_json_stream
        import pyarrow.json as js

        LONG_TEXT_MULTIPLIER = 100_000
        BLOCK_SIZE_MB = 3
        BLOCK_SIZE_BYTES = BLOCK_SIZE_MB * 1024 * 1024

        _text = "I have a very long text. " * LONG_TEXT_MULTIPLIER  # Create a long text to test large JSON objects
        self.test_data.append({"text": _text, "images": ["image4.jpg"], "videos": [], "audios": []})
        # Create a temporary JSONL file
        jsonl_path = os.path.join(self.tmp_dir, "test.jsonl")
        with open(jsonl_path, "w") as f:
            for item in self.test_data:
                f.write(f"{json.dumps(item)}\n")

        # In PyArrow 20.0.0+, a large record can cause a 'straddling object' error.
        # We set a larger block size to allow PyArrow to read larger records.
        read_options = js.ReadOptions(block_size=BLOCK_SIZE_BYTES)  # Set block size to 3MB
        dataset = read_json_stream(jsonl_path, read_options=read_options)
        self.assertEqual(len(dataset.take(3)[2]["text"]), len(_text))

    @TEST_TAG("ray")
    def test_read_json_stream_schema_evolution(self):
        """Regression test for #936: null -> concrete type schema evolution."""
        from data_juicer.core.data.ray_dataset import read_json_stream
        import pyarrow.json as js

        jsonl_path = os.path.join(self.tmp_dir, "schema_evolution.jsonl")
        rows = [{"id": i, "meta": {"url": None}} for i in range(30)]
        rows.append({"id": 999, "meta": {"url": "https://example.com"}})
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        read_options = js.ReadOptions(use_threads=False, block_size=256)
        dataset = read_json_stream(jsonl_path, override_num_blocks=1, read_options=read_options)
        result = dataset.take_all()
        self.assertEqual(len(result), 31)
        self.assertEqual(result[-1]["id"], 999)
        self.assertEqual(result[-1]["meta"]["url"], "https://example.com")

    @TEST_TAG("ray")
    def test_read_json_stream_schema_evolution_with_filesystem(self):
        """Regression test: schema evolution fallback works with filesystem abstraction."""
        from data_juicer.core.data.ray_dataset import read_json_stream
        import pyarrow.json as js
        import pyarrow.fs as pafs

        jsonl_path = os.path.join(self.tmp_dir, "schema_evolution_fs.jsonl")
        rows = [{"id": i, "meta": {"url": None}} for i in range(30)]
        rows.append({"id": 999, "meta": {"url": "https://example.com"}})
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        fs = pafs.SubTreeFileSystem(self.tmp_dir, pafs.LocalFileSystem())
        relative_path = "schema_evolution_fs.jsonl"

        read_options = js.ReadOptions(use_threads=False, block_size=256)
        dataset = read_json_stream(
            relative_path,
            filesystem=fs,
            override_num_blocks=1,
            read_options=read_options,
        )
        result = dataset.take_all()
        self.assertEqual(len(result), 31)
        self.assertEqual(result[-1]["id"], 999)
        self.assertEqual(result[-1]["meta"]["url"], "https://example.com")

    @TEST_TAG("ray")
    def test_read_json_stream_stable_schema_no_fallback(self):
        """Verify stable-schema files stream without buffering (no fallback)."""
        from data_juicer.core.data.ray_dataset import read_json_stream
        import pyarrow.json as js

        jsonl_path = os.path.join(self.tmp_dir, "stable_schema.jsonl")
        rows = [{"id": i, "meta": {"url": f"https://example.com/{i}"}} for i in range(100)]
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        read_options = js.ReadOptions(use_threads=False, block_size=256)
        dataset = read_json_stream(jsonl_path, override_num_blocks=1, read_options=read_options)
        result = dataset.take_all()
        self.assertEqual(len(result), 100)
        self.assertEqual(result[0]["id"], 0)
        self.assertEqual(result[99]["meta"]["url"], "https://example.com/99")

    def _write_rows(self, filename, rows):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    @TEST_TAG("ray")
    def test_read_json_stream_read_options_dict_and_unset(self):
        """read_options accepts dicts, and unset values keep the reader default."""
        from data_juicer.core.data.ray_dataset import read_json_stream

        rows = [{"id": i, "text": f"row {i}"} for i in range(20)]
        jsonl_path = self._write_rows("read_options_forms.jsonl", rows)

        # A dict is converted to ReadOptions; an empty dict and None are treated
        # as unset so Ray's ReadOptions(use_threads=False) default still applies.
        for read_options in ({"block_size": 1 << 20}, {}, None):
            with self.subTest(read_options=read_options):
                dataset = read_json_stream(jsonl_path, read_options=read_options)
                result = dataset.take_all()
                self.assertEqual(len(result), 20)
                self.assertEqual(result[0]["text"], "row 0")

    @TEST_TAG("ray")
    def test_read_json_stream_read_options_keeps_use_threads_disabled(self):
        """use_threads defaults to False, and an explicit value is preserved."""
        import pyarrow.json as js

        from data_juicer.core.data import ray_dataset

        real_read_options = js.ReadOptions
        calls = []

        def _capture(**kwargs):
            calls.append(kwargs)
            return real_read_options(**kwargs)

        rows = [{"id": i} for i in range(5)]
        jsonl_path = self._write_rows("read_options_threads.jsonl", rows)

        # Ray's datasource also builds a ReadOptions(use_threads=False) default
        # eagerly, so only the first call comes from read_json_stream itself.
        for read_options, expected in (
            ({"block_size": 4096}, {"block_size": 4096, "use_threads": False}),
            ({"use_threads": True}, {"use_threads": True}),
        ):
            with self.subTest(read_options=read_options):
                calls.clear()
                with patch.object(js, "ReadOptions", _capture):
                    ray_dataset.read_json_stream(jsonl_path, read_options=read_options)
                self.assertEqual(calls[0], expected)

    @TEST_TAG("ray")
    def test_read_json_stream_old_pyarrow_fallback(self):
        """Regression test: the pre-20.0.0 PyArrow fallback must materialize.

        The fallback delegates to ray.data.read_json, which has no meta_provider
        parameter and rejects read_options=None. Both failures only surface when
        the blocks are read, so this test takes rows rather than just building
        the dataset.
        """
        import pyarrow.json as js

        from data_juicer.core.data.ray_dataset import read_json_stream

        rows = [{"id": i, "text": f"row {i}"} for i in range(10)]
        jsonl_path = self._write_rows("old_pyarrow_fallback.jsonl", rows)

        # Hide open_json so read_json_stream takes the older-PyArrow branch.
        open_json = js.__dict__.pop("open_json")
        try:
            self.assertFalse(hasattr(js, "open_json"))
            for read_options in (None, {}, {"block_size": 1 << 20}):
                with self.subTest(read_options=read_options):
                    dataset = read_json_stream(jsonl_path, read_options=read_options)
                    result = dataset.take_all()
                    self.assertEqual(len(result), 10)
                    self.assertEqual(result[0]["text"], "row 0")
        finally:
            js.__dict__["open_json"] = open_json


class TestRayDataset(DataJuicerTestCaseBase):
    def setUp(self):
        """Set up test data"""
        super().setUp()

        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        self.data = [
            {"text": "Hello", "score": 1, "metadata": {"lang": "en"}, "labels": [1, 2, 3]},
            {"text": "World", "score": 2, "metadata": {"lang": "es"}, "labels": [4, 5, 6]},
            {"text": "Test", "score": 3, "metadata": {"lang": "fr"}, "labels": [7, 8, 9]},
        ]

        # Create fresh dataset for each test
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up test data"""
        self.dataset = None
        super().tearDown()

    @TEST_TAG("ray")
    def test_get_column_basic(self):
        """Test basic column retrieval"""
        # Test string column
        texts = self.dataset.get_column("text")
        self.assertEqual(texts, ["Hello", "World", "Test"])

        # Test numeric column
        scores = self.dataset.get_column("score")
        self.assertEqual(scores, [1, 2, 3])

        # Test dict column
        metadata = self.dataset.get_column("metadata")
        self.assertEqual(metadata, [{"lang": "en"}, {"lang": "es"}, {"lang": "fr"}])

        # Test list column
        labels = self.dataset.get_column("labels")
        self.assertEqual(labels, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    @TEST_TAG("ray")
    def test_get_column_with_k(self):
        """Test column retrieval with k limit"""
        # Test k=2
        texts = self.dataset.get_column("text", k=2)
        self.assertEqual(texts, ["Hello", "World"])

        # Test k larger than dataset
        texts = self.dataset.get_column("text", k=5)
        self.assertEqual(texts, ["Hello", "World", "Test"])

        # Test k=0
        texts = self.dataset.get_column("text", k=0)
        self.assertEqual(texts, [])

        # Test k=1
        texts = self.dataset.get_column("text", k=1)
        self.assertEqual(texts, ["Hello"])

        # Ray's take() defaults to 20 rows, so explicitly requesting or
        # retrieving all rows must not truncate larger datasets.
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        large_data = [{"text": str(i)} for i in range(25)]
        large_dataset = RayDataset(ray.data.from_items(large_data))
        self.assertEqual(large_dataset.get_column("text", k=25), [str(i) for i in range(25)])
        self.assertEqual(large_dataset.get_column("text"), [str(i) for i in range(25)])

    @TEST_TAG("ray")
    def test_get_column_errors(self):
        """Test error handling"""
        # Test non-existent column
        with self.assertRaises(KeyError) as context:
            self.dataset.get_column("nonexistent")
        self.assertIn("not found in dataset", str(context.exception))

        # Test negative k
        with self.assertRaises(ValueError) as context:
            self.dataset.get_column("text", k=-1)
        self.assertIn("must be non-negative", str(context.exception))

    @TEST_TAG("ray")
    def test_get_column_empty_dataset(self):
        """Test with empty dataset"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        empty_dataset = RayDataset(ray.data.from_items([]))

        # Should raise ValuError for empty dataset/columns
        with self.assertRaises(KeyError):
            empty_dataset.get_column("text")

    @TEST_TAG("ray")
    def test_get_column_types(self):
        """Test return type consistency"""
        # All elements should be strings
        texts = self.dataset.get_column("text")
        self.assertTrue(all(isinstance(x, str) for x in texts))

        # All elements should be ints
        scores = self.dataset.get_column("score")
        self.assertTrue(all(isinstance(x, int) for x in scores))

        # All elements should be dicts
        metadata = self.dataset.get_column("metadata")
        self.assertTrue(all(isinstance(x, dict) for x in metadata))

        # All elements should be lists
        labels = self.dataset.get_column("labels")
        self.assertTrue(all(isinstance(x, list) for x in labels))

    @TEST_TAG("ray")
    def test_get_column_preserve_order(self):
        """Test that column order is preserved"""
        texts = self.dataset.get_column("text")
        self.assertEqual(texts[0], "Hello")
        self.assertEqual(texts[1], "World")
        self.assertEqual(texts[2], "Test")

        # Test with k
        texts = self.dataset.get_column("text", k=2)
        self.assertEqual(texts[0], "Hello")
        self.assertEqual(texts[1], "World")

    @TEST_TAG("ray")
    def test_get(self):
        """Test get method for RayDataset"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        # Test with simple data
        simple_data = [{"text": "hello", "score": 1}, {"text": "world", "score": 2}, {"text": "test", "score": 3}]
        dataset = RayDataset(ray.data.from_items(simple_data))

        # Basic get
        rows = dataset.get(2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"text": "hello", "score": 1})
        self.assertEqual(rows[1], {"text": "world", "score": 2})

        # Test with nested structures
        nested_data = [
            {"text": "hello", "metadata": {"lang": "en", "source": "web"}, "tags": [1, 2, 3]},
            {"text": "world", "metadata": {"lang": "es", "source": "book"}, "tags": [4, 5, 6]},
        ]
        nested_dataset = RayDataset(ray.data.from_items(nested_data))

        # Test nested structure preservation
        rows = nested_dataset.get(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metadata"]["lang"], "en")
        self.assertEqual(rows[0]["tags"], [1, 2, 3])

        # Test edge cases
        self.assertEqual(dataset.get(0), [])
        self.assertEqual(len(dataset.get(10)), 3)  # More than dataset size
        with self.assertRaises(ValueError):
            dataset.get(-1)

        # Test type preservation
        row = dataset.get(1)[0]
        self.assertIsInstance(row, dict)
        self.assertIsInstance(row["text"], str)
        self.assertIsInstance(row["score"], int)

        large_data = [{"text": str(i)} for i in range(25)]
        large_dataset = RayDataset(ray.data.from_items(large_data))
        self.assertEqual(large_dataset.get(25), large_data)

    @TEST_TAG("ray")
    def test_process_does_not_count_before_building_plan(self):
        from data_juicer.core.data.ray_dataset import RayDataset

        ray_data = MagicMock()
        ray_data.columns.return_value = ["text"]
        dataset = RayDataset.__new__(RayDataset)
        dataset.data = ray_data
        dataset._auto_proc = False
        dataset._run_single_op = MagicMock(return_value={"text"})

        result = dataset.process(MagicMock())

        self.assertIs(result, dataset)
        ray_data.count.assert_not_called()
        ray_data.columns.assert_called_once()

    @TEST_TAG("ray")
    def test_process_skips_empty_dataset_with_known_schema(self):
        """Empty datasets with a known schema (columns() returns []) must be
        skipped instead of running operators on zero rows."""
        import pyarrow
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.ops.mapper.punctuation_normalization_mapper import (
            PunctuationNormalizationMapper,
        )

        empty_data = pyarrow.table({"text": []})
        dataset = RayDataset(ray.data.from_arrow(empty_data))
        self.assertEqual(dataset.data.columns(), ["text"])
        self.assertEqual(dataset.data.count(), 0)

        result = dataset.process([PunctuationNormalizationMapper()])

        self.assertIs(result, dataset)


class RayComputeStrategyTest(DataJuicerTestCaseBase):
    def _get_compute_strategy(self, op):
        from data_juicer.core.data.ray_dataset import RayDataset

        ray_data = MagicMock()
        ray_data.map_batches.return_value = ray_data
        dataset = RayDataset.__new__(RayDataset)
        dataset.data = ray_data

        dataset._run_single_op(op, {"text"})

        compute_strategies = [
            call.kwargs["compute"] for call in ray_data.map_batches.call_args_list if "compute" in call.kwargs
        ]
        self.assertEqual(len(compute_strategies), 1)
        return compute_strategies[0]

    @TEST_TAG("ray")
    def test_public_compute_strategies(self):
        from ray.data import ActorPoolStrategy, TaskPoolStrategy

        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.mapper.python_lambda_mapper import PythonLambdaMapper

        for op_class in [PythonLambdaMapper, TextLengthFilter]:
            with self.subTest(op=op_class.__name__, mode="task"):
                op = op_class(auto_op_parallelism=False, num_proc=2, ray_execution_mode="task")
                compute = self._get_compute_strategy(op)
                self.assertIsInstance(compute, TaskPoolStrategy)
                self.assertEqual(compute.size, 2)

            with self.subTest(op=op_class.__name__, mode="actor"):
                op = op_class(auto_op_parallelism=False, num_proc=2, ray_execution_mode="actor")
                compute = self._get_compute_strategy(op)
                self.assertIsInstance(compute, ActorPoolStrategy)
                self.assertEqual(compute.min_size, 2)
                self.assertEqual(compute.max_size, 2)

            with self.subTest(op=op_class.__name__, mode="actor", concurrency="elastic"):
                op = op_class(auto_op_parallelism=False, num_proc=(1, 3), ray_execution_mode="actor")
                compute = self._get_compute_strategy(op)
                self.assertIsInstance(compute, ActorPoolStrategy)
                self.assertEqual(compute.min_size, 1)
                self.assertEqual(compute.max_size, 3)


class RayBatchExecutionTest(unittest.TestCase):
    def test_native_actor_uses_original_operator_call(self):
        import pyarrow as pa

        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.ops.base_op import Mapper

        class CustomCallMapper(Mapper):
            def __call__(self, table):
                return table.append_column("custom", pa.array([True] * table.num_rows))

            def process_single(self, sample):
                raise AssertionError("Original __call__ must be preserved")

        op = CustomCallMapper(num_proc=1, ray_execution_mode="actor")
        dataset = RayDataset.__new__(RayDataset)
        dataset.data = MagicMock()
        dataset.data.map_batches.return_value = dataset.data
        dataset._run_single_op(op, {"text"})
        call = dataset.data.map_batches.call_args
        self.assertIs(call.args[0], CustomCallMapper)
        actor = call.args[0](*call.kwargs["fn_constructor_args"], **call.kwargs["fn_constructor_kwargs"])
        self.assertEqual(actor(pa.table({"text": ["one"]})).to_pylist(), [{"text": "one", "custom": True}])
        self.assertEqual(actor(pa.table({"text": ["two"]})).to_pylist(), [{"text": "two", "custom": True}])

    def test_preparation_handles_tagging_filters_and_preserves_existing_columns(self):
        import pyarrow as pa
        from unittest.mock import patch

        from data_juicer.core.data.ray_dataset import operator_batch_columns, prepare_batch_columns
        from data_juicer.ops.base_op import TAGGING_OPS
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.utils.constant import Fields

        op = TextLengthFilter()
        with patch.dict(TAGGING_OPS.modules, {op._name: type(op)}):
            columns = operator_batch_columns(op)
        self.assertEqual(columns, [Fields.meta, Fields.stats])
        table = pa.table({"text": ["one"], Fields.meta: [{"tag": "keep"}]})
        prepared = prepare_batch_columns(table, columns)
        self.assertEqual(prepared[Fields.meta].to_pylist(), [{"tag": "keep"}])
        self.assertEqual(prepared[Fields.stats].to_pylist(), [{}])
        self.assertIs(prepare_batch_columns(prepared, columns), prepared)

    def test_non_stats_filter_does_not_create_empty_struct(self):
        from data_juicer.core.data.ray_dataset import operator_batch_columns
        from data_juicer.ops.filter.suffix_filter import SuffixFilter

        self.assertEqual(operator_batch_columns(SuffixFilter()), [])

    def test_nonbatched_filter_preserves_native_tensor_shape(self):
        import numpy as np
        import pyarrow as pa

        from data_juicer.core.data.ray_dataset import filter_batch

        values = np.arange(12).reshape(3, 2, 2)
        table = pa.table({"tensor": pa.FixedShapeTensorArray.from_numpy_ndarray(values)})
        selected = filter_batch(
            table, lambda row: row["tensor"].shape == (2, 2) and row["tensor"][0, 0] == 4, is_batched=False
        )
        self.assertTrue(selected.equals(table.slice(1, 1)))

    def test_forced_batch_non_stats_filter_keeps_required_stats_preparation(self):
        from data_juicer.core.data.ray_dataset import operator_batch_columns
        from data_juicer.ops.filter.suffix_filter import SuffixFilter
        from data_juicer.utils.constant import Fields

        self.assertEqual(operator_batch_columns(SuffixFilter(batch_mode=True)), [Fields.stats])

    def test_batched_and_single_filter_use_same_arrow_selection(self):
        import pyarrow as pa

        from data_juicer.core.data.ray_dataset import filter_batch

        table = pa.table({"value": [0, 1, 2]})
        batched = filter_batch(table, lambda samples: [value % 2 == 0 for value in samples["value"]])
        single = filter_batch(table, lambda row: row["value"] % 2 == 0, is_batched=False)
        self.assertTrue(batched.equals(single))
        self.assertEqual(single["value"].to_pylist(), [0, 2])
        with self.assertRaises(pa.ArrowInvalid):
            filter_batch(table, lambda samples: [True])
        empty = table.slice(0, 0)
        self.assertIs(filter_batch(empty, lambda samples: self.fail("Empty filters must not run")), empty)

    def test_checkpoint_stage_keeps_native_compute_strategy(self):
        from types import SimpleNamespace

        from ray.data import ActorPoolStrategy, TaskPoolStrategy

        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.core.executor import ray_partition_runner
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper

        for op_class in (TextLengthFilter, WhitespaceNormalizationMapper):
            for mode, concurrency in (("actor", 3), ("actor", (1, 3)), ("task", 3)):
                with self.subTest(op=op_class, mode=mode, concurrency=concurrency):
                    dataset = RayDataset(MagicMock(), auto_op_parallelism=False)
                    dataset.data._map_batches_without_batch_size_validation.return_value = dataset.data
                    context = object()
                    recovery = SimpleNamespace(context=context, register_stats=lambda export: None)
                    op = op_class(num_proc=concurrency, ray_execution_mode=mode)
                    dataset._run_single_op(op, defer_schema_probe=True, recovery=recovery)
                    args, options = dataset.data._map_batches_without_batch_size_validation.call_args
                    self.assertIsNone(options["batch_size"])
                    dataset.data.map_batches.assert_not_called()
                    compute = options["compute"]
                    if mode == "task":
                        self.assertIs(args[0], ray_partition_runner.run_checkpoint_task)
                        self.assertIs(options["fn_args"][0], context)
                        self.assertIsInstance(compute, TaskPoolStrategy)
                        self.assertEqual(compute.size, 3)
                    else:
                        self.assertIs(args[0], ray_partition_runner.CheckpointStageActor)
                        self.assertIs(options["fn_constructor_args"][0], context)
                        self.assertIsInstance(compute, ActorPoolStrategy)
                        self.assertEqual(compute.min_size, 1 if isinstance(concurrency, tuple) else 3)
                        self.assertEqual(compute.max_size, 3)
                    self.assertEqual(options["num_cpus"], op.num_cpus)
                    self.assertEqual(options["num_gpus"], op.num_gpus)
                    self.assertEqual(options["runtime_env"], op.runtime_env)

    def test_partition_batches_preserve_requested_mutability(self):
        import numpy as np
        import pyarrow as pa

        from data_juicer.core.data.ray_batch_plan import BatchPlanWorker, RayBatchPlan

        def mutate(batch):
            batch["value"] += 1
            return batch

        for batch_size in (2, None):
            with self.subTest(batch_size=batch_size):
                table = pa.table({"value": np.arange(3)})
                plan = RayBatchPlan().map_batches(
                    mutate, batch_size=batch_size, batch_format="numpy", zero_copy_batch=False
                )
                result = [
                    row for block in BatchPlanWorker(plan).process_partition([table]) for row in block.to_pylist()
                ]
                self.assertEqual(result, [{"value": 1}, {"value": 2}, {"value": 3}])
                self.assertEqual(table["value"].to_pylist(), [0, 1, 2])

    def test_partition_batches_copy_only_readonly_numpy_columns(self):
        import numpy as np
        import pyarrow as pa

        from data_juicer.core.data.ray_batch_plan import BatchPlanWorker, RayBatchPlan

        for zero_copy_batch in (False, True):
            with self.subTest(zero_copy_batch=zero_copy_batch):
                readonly = np.arange(3)
                readonly.flags.writeable = False
                writable = np.arange(3)
                batch = {"readonly": readonly, "writable": writable}

                def inspect_batch(output):
                    self.assertIs(output["writable"], writable)
                    if zero_copy_batch:
                        self.assertIs(output, batch)
                        self.assertIs(output["readonly"], readonly)
                        self.assertFalse(output["readonly"].flags.writeable)
                    else:
                        self.assertTrue(output["readonly"].flags.writeable)
                        self.assertFalse(np.shares_memory(output["readonly"], readonly))
                        np.testing.assert_array_equal(output["readonly"], readonly)
                    return output

                plan = RayBatchPlan().map_batches(
                    inspect_batch, batch_size=2, batch_format="numpy", zero_copy_batch=zero_copy_batch
                )
                with patch("data_juicer.core.data.ray_batch_adapter.batch_blocks", return_value=iter([batch])) as batching:
                    list(BatchPlanWorker(plan).process_partition([pa.table({"value": np.arange(3)})]))
                self.assertEqual(batching.call_args.kwargs["ensure_copy"], not zero_copy_batch)
                self.assertFalse(readonly.flags.writeable)

    def test_partition_worker_rejects_global_barrier(self):
        from data_juicer.core.data.ray_batch_plan import BatchPlanWorker, RayBatchPlan

        with self.assertRaisesRegex(RuntimeError, "global materialization barrier"):
            list(BatchPlanWorker(RayBatchPlan().materialize()).process_partition([]))

    def test_native_and_partition_plans_reject_invalid_business_batches(self):
        from types import SimpleNamespace

        from data_juicer.core.data.ray_batch_plan import RayBatchPlan
        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter

        for size in (0, -1):
            for recovery in (None, SimpleNamespace(recovery=SimpleNamespace(partitions=(0,)))):
                with self.subTest(size=size, recovery=recovery):
                    dataset = RayDataset(MagicMock(), auto_op_parallelism=False)
                    op = TextLengthFilter(batch_size=size, num_proc=1, ray_execution_mode="actor")
                    with self.assertRaisesRegex(ValueError, "negative or 0"):
                        dataset._run_single_op(op, defer_schema_probe=True, recovery=recovery)
                    dataset.data.map_batches.assert_not_called()
        for size in (None, "auto"):
            with self.assertRaisesRegex(ValueError, "explicit batch_size"):
                RayBatchPlan().map_batches(lambda batch: batch, num_gpus=1, batch_size=size)

    def test_native_shared_state_keeps_task_override_and_global_barrier(self):
        from ray.data import TaskPoolStrategy

        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter

        events = []

        class SharedFilter(TextLengthFilter):
            def _prepare_for_ray_map_batches(self):
                events.append("prepare_shared_state")
                return True

        ray_data = MagicMock()

        def map_batches(function, **options):
            if "compute" in options:
                self.assertIsInstance(options["compute"], TaskPoolStrategy)
                events.append("compute")
            else:
                events.append("filter")
            return ray_data

        def materialize():
            events.append("materialize")
            return ray_data

        ray_data.map_batches.side_effect = map_batches
        ray_data.materialize.side_effect = materialize
        ray_data.write_json.side_effect = lambda *args, **kwargs: events.append("stats_export")
        op = SharedFilter(num_proc=1, ray_execution_mode="actor", stats_export_path="stats")
        dataset = RayDataset(ray_data, auto_op_parallelism=False)
        dataset._run_single_op(op, {"text", "__dj__stats__"})
        self.assertEqual(events, ["prepare_shared_state", "compute", "materialize", "stats_export", "filter"])


if __name__ == "__main__":
    unittest.main()
