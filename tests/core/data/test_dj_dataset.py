import unittest
from datasets import Dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from data_juicer.core.data import NestedDataset, wrap_func_with_nested_access
from data_juicer.core.data.dj_dataset import nested_obj_factory, NestedDatasetDict, NestedQueryDict
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestNestedDataset(DataJuicerTestCaseBase):
    def setUp(self):
        """Set up test data"""
        super().setUp()

        self.data = [
            {
                'text': 'Hello',
                'score': 1,
                'metadata': {'lang': 'en'},
                'labels': [1, 2, 3]
            },
            {
                'text': 'World',
                'score': 2,
                'metadata': {'lang': 'es'},
                'labels': [4, 5, 6]
            },
            {
                'text': 'Test',
                'score': 3,
                'metadata': {'lang': 'fr'},
                'labels': [7, 8, 9]
            }
        ]
        self.dataset = NestedDataset(Dataset.from_list(self.data))

    def test_get_column_basic(self):
        """Test basic column retrieval"""
        # Test string column
        texts = self.dataset.get_column('text')
        self.assertEqual(texts, ['Hello', 'World', 'Test'])
        
        # Test numeric column
        scores = self.dataset.get_column('score')
        self.assertEqual(scores, [1, 2, 3])
        
        # Test dict column
        metadata = self.dataset.get_column('metadata')
        self.assertEqual(metadata, [
            {'lang': 'en'},
            {'lang': 'es'},
            {'lang': 'fr'}
        ])
        
        # Test list column
        labels = self.dataset.get_column('labels')
        self.assertEqual(labels, [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # contain_column
        self.assertTrue(self.dataset.contain_column('text'))
        self.assertFalse(self.dataset.contain_column('nonexistent'))

    def test_get_column_with_k(self):
        """Test column retrieval with k limit"""
        # Test k=2
        texts = self.dataset.get_column('text', k=2)
        self.assertEqual(texts, ['Hello', 'World'])
        
        # Test k larger than dataset
        texts = self.dataset.get_column('text', k=5)
        self.assertEqual(texts, ['Hello', 'World', 'Test'])
        
        # Test k=0
        texts = self.dataset.get_column('text', k=0)
        self.assertEqual(texts, [])
        
        # Test k=1
        texts = self.dataset.get_column('text', k=1)
        self.assertEqual(texts, ['Hello'])

    def test_get_column_errors(self):
        """Test error handling"""
        # Test non-existent column
        with self.assertRaises(KeyError) as context:
            self.dataset.get_column('nonexistent')
        self.assertIn("not found in dataset", str(context.exception))
        
        # Test negative k
        with self.assertRaises(ValueError) as context:
            self.dataset.get_column('text', k=-1)
        self.assertIn("must be non-negative", str(context.exception))

    def test_get_column_empty_dataset(self):
        """Test with empty dataset"""
        empty_dataset = NestedDataset(Dataset.from_list([]))
        
        # Should return empty list for existing column
        with self.assertRaises(KeyError):
            empty_dataset.get_column('text')

    def test_get_column_types(self):
        """Test return type consistency"""
        # All elements should be strings
        texts = self.dataset.get_column('text')
        self.assertTrue(all(isinstance(x, str) for x in texts))
        
        # All elements should be ints
        scores = self.dataset.get_column('score')
        self.assertTrue(all(isinstance(x, int) for x in scores))
        
        # All elements should be dicts
        metadata = self.dataset.get_column('metadata')
        self.assertTrue(all(isinstance(x, dict) for x in metadata))
        
        # All elements should be lists
        labels = self.dataset.get_column('labels')
        self.assertTrue(all(isinstance(x, list) for x in labels))

    def test_get_column_preserve_order(self):
        """Test that column order is preserved"""
        texts = self.dataset.get_column('text')
        self.assertEqual(texts[0], 'Hello')
        self.assertEqual(texts[1], 'World')
        self.assertEqual(texts[2], 'Test')
        
        # Test with k
        texts = self.dataset.get_column('text', k=2)
        self.assertEqual(texts[0], 'Hello')
        self.assertEqual(texts[1], 'World')

    def test_get(self):
        """Test get method for NestedDataset"""
        # Test with simple data
        simple_data = [
            {'text': 'hello', 'score': 1},
            {'text': 'world', 'score': 2},
            {'text': 'test', 'score': 3}
        ]
        dataset = NestedDataset(Dataset.from_list(simple_data))
        
        # Basic get
        rows = dataset.get(2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {'text': 'hello', 'score': 1})
        self.assertEqual(rows[1], {'text': 'world', 'score': 2})
        
        # Test with nested structures
        nested_data = [
            {
                'text': 'hello',
                'metadata': {'lang': 'en', 'source': 'web'},
                'tags': [1, 2, 3]
            },
            {
                'text': 'world',
                'metadata': {'lang': 'es', 'source': 'book'},
                'tags': [4, 5, 6]
            }
        ]
        nested_dataset = NestedDataset(Dataset.from_list(nested_data))
        
        # Test nested structure preservation
        rows = nested_dataset.get(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['metadata']['lang'], 'en')
        self.assertEqual(rows[0]['tags'], [1, 2, 3])
        
        # Test edge cases
        self.assertEqual(dataset.get(0), [])
        self.assertEqual(len(dataset.get(10)), 3)  # More than dataset size
        with self.assertRaises(ValueError):
            dataset.get(-1)
            
        # Test type preservation
        row = dataset.get(1)[0]
        self.assertIsInstance(row, dict)
        self.assertIsInstance(row['text'], str)
        self.assertIsInstance(row['score'], int)

    def test_nested_access(self):
        # simpale test function
        def sample_function(*args, **kwargs):
            return args, kwargs

        # basic function: wrap for simple args
        wrapped_func = wrap_func_with_nested_access(sample_function)

        args = (1, {"key": "value"}, [1, 2, 3])
        kwargs = {"a": 4, "b": {"c": 5}}

        result_args, result_kwargs = wrapped_func(*args, **kwargs)

        for i in range(len(args)):
            self.assertEqual(result_args[i], nested_obj_factory(args[i]))

        for key in kwargs:
            self.assertEqual(result_kwargs[key], nested_obj_factory(kwargs[key]))

        # test nested qeury
        inner_func = lambda x: x
        wrapped_inner_func = wrap_func_with_nested_access(inner_func)

        def outer_func(*args, **kwargs):
            return args, kwargs

        wrapped_outer_func = wrap_func_with_nested_access(outer_func)

        result_args, result_kwargs = wrapped_outer_func(wrapped_inner_func, key=wrapped_inner_func)

        self.assertTrue(callable(result_args[0]))
        self.assertTrue(callable(result_kwargs["key"]))

        # test mixed types
        wrapped_func = wrap_func_with_nested_access(sample_function)

        args = (1, {"meta": {"date": "2023-01-01"}}, [1, 2, 3])
        kwargs = {"a": 4, "b": {"c": 5}, "list_of_dicts": [{"x": 1}, {"y": 2}]}

        result_args, result_kwargs = wrapped_func(*args, **kwargs)

        for i in range(len(args)):
            self.assertEqual(result_args[i], nested_obj_factory(args[i]))

        for key in kwargs:
            self.assertEqual(result_kwargs[key], nested_obj_factory(kwargs[key]))

    def test_nested_obj_factory(self):
        # test dataset
        ds = Dataset.from_dict({"text": ["Hello", "World"]})
        result = nested_obj_factory(ds)
        self.assertIsInstance(result, NestedDataset)

        # test dataset dict
        ds_dict = DatasetDict({"train": ds})
        result = nested_obj_factory(ds_dict)
        self.assertIsInstance(result, NestedDatasetDict)
        self.assertEqual(result["train"], ds)
        mapped_ds_dict = result.map(function=lambda x: {"text": x["text"] + "1"})
        self.assertEqual(mapped_ds_dict["train"]["text"], ["Hello1", "World1"])
        mapped_ds_dict = result.map()
        self.assertEqual(mapped_ds_dict["train"]["text"], ["Hello", "World"])

        # test dict
        input_dict = {"key": "value"}
        result = nested_obj_factory(input_dict)
        self.assertIsInstance(result, NestedQueryDict)
        self.assertEqual(result["key"], "value")

        # test lazy batch
        class MockLazyBatch(LazyBatch):
            def __init__(self):
                self.data = {"key1": {"key2": "value"}}

        lazy_batch = MockLazyBatch()
        result = nested_obj_factory(lazy_batch)
        self.assertIsInstance(result.data, NestedQueryDict)
        self.assertEqual(result.data["key1.key2"], "value")

        # test list of dict
        input_list = [{"a": 1}, {"b": 2}]
        result = nested_obj_factory(input_list)
        self.assertIsInstance(result[0], NestedQueryDict)
        self.assertIsInstance(result[1], NestedQueryDict)
        self.assertEqual(result[0]["a"], 1)
        self.assertEqual(result[1]["b"], 2)

        # test nested list
        input_list = [[{"a": 1}], [{"b": 2}]]
        result = nested_obj_factory(input_list)
        self.assertIsInstance(result[0][0], NestedQueryDict)
        self.assertEqual(result[0][0]["a"], 1)

        # test simple types
        result = nested_obj_factory(42)
        self.assertEqual(result, 42)
        result = nested_obj_factory("hello")
        self.assertEqual(result, "hello")
        result = nested_obj_factory(3.14)
        self.assertEqual(result, 3.14)
        result = nested_obj_factory(None)
        self.assertIsNone(result)

    def test_nested_dataset(self):
        import pyarrow as pa
        table = pa.Table.from_pydict({"text": ["hello", "world"]})
        ds = NestedDataset(table)
        self.assertEqual(ds[0], {"text": "hello"})
        self.assertEqual(ds[1], {"text": "world"})

        # test empty ops
        res = ds.process([])
        self.assertEqual(res, ds)


if __name__ == '__main__':
    unittest.main()
