import unittest
from data_juicer.utils.unittest_utils import TEST_TAG
from datasets import Dataset
from typing import List, Any
from data_juicer.core.data.schema import Schema
from data_juicer.core.data import NestedDataset
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestSchema(DataJuicerTestCaseBase):

    def test_schema_single_dj_dataset(self):
        """Test schema for single dataset"""
        data = [{'text': 'hello', 'score': 1}]
        dataset = NestedDataset(Dataset.from_list(data))

        schema = dataset.schema()
        self.assertIn('text', schema.columns)
        self.assertEqual(schema.column_types['text'], str)
        self.assertEqual(schema.column_types['score'], int)

    def test_schema_multiple_dj_datasets(self):
        """Test schema consistency across multiple datasets"""
        data1 = [{'text': 'hello', 'score': 1}]
        data2 = [{'text': 'world', 'score': 2}]

        dataset1 = NestedDataset(Dataset.from_list(data1))
        dataset2 = NestedDataset(Dataset.from_list(data2))

        # Verify schemas match
        schema1 = dataset1.schema()
        schema2 = dataset2.schema()

        self.assertEqual(schema1.columns, schema2.columns)
        self.assertEqual(schema1.column_types, schema2.column_types)

    def test_schema_validation_dj_dataset(self):
        """Test schema validation"""
        # Test with invalid schema (mixed types)
        data = [
            {'text': 'hello', 'value': 1},
            {'text': 'world', 'value': 'string'}  # Mixed int/str
        ]

        with self.assertRaises(Exception) as context:
            dataset = NestedDataset(Dataset.from_list(data))
            _ = dataset.schema()

        # Should raise error when dealing with mixed types
        self.assertIn("could not convert", str(context.exception).lower())

    def test_schema_nested_structures_dj_dataset(self):
        """Test schema with nested data structures"""
        data = [{
            'text': 'hello',
            'int_value': 1,
            'float_value': 1.0,
            'bool_value': True,
            'metadata': {'lang': 'en', 'score': 1},
            'tags': ['tag1', 'tag2'],
            'nested': {'a': {'b': {'c': 1}}}
        }]

        dataset = NestedDataset(Dataset.from_list(data))
        schema = dataset.schema()

        self.assertEqual(schema.column_types['text'], str)
        self.assertEqual(schema.column_types['int_value'], int)
        self.assertEqual(schema.column_types['float_value'], float)
        self.assertEqual(schema.column_types['bool_value'], bool)
        self.assertIsInstance(schema.column_types['metadata'], Schema)
        self.assertEqual(schema.column_types['tags'], List[str])
        self.assertIsInstance(schema.column_types['nested'], Schema)

    def test_schema_empty_dj_dataset(self):
        """Test schema with empty dataset"""
        dataset = NestedDataset(Dataset.from_list([]))
        schema = dataset.schema()

        self.assertEqual(len(schema.columns), 0)
        self.assertEqual(len(schema.column_types), 0)

    def test_schema_special_characters_dj_dataset(self):
        """Test schema with special characters in column names"""
        data = [{
            'normal': 1,
            'with.dot': 2,
            'with-dash': 3,
            '_underscore': 4,
            'with space': 5
        }]

        dataset = NestedDataset(Dataset.from_list(data))
        schema = dataset.schema()

        expected_columns = {
            'normal', 'with.dot', 'with-dash',
            '_underscore', 'with space'
        }
        self.assertEqual(set(schema.columns), expected_columns)

    def test_schema_type_consistency_dj_dataset(self):
        """Test schema type consistency across rows"""
        data = [
            {'text': 'hello', 'score': 1, 'flag': True},
            {'text': 'world', 'score': 2, 'flag': False},
            {'text': 'test', 'score': 3, 'flag': True}
        ]

        dataset = NestedDataset(Dataset.from_list(data))
        schema = dataset.schema()

        # Verify types are consistent
        self.assertTrue(all(
            isinstance(row['text'], schema.column_types['text'])
            for row in dataset
        ))
        self.assertTrue(all(
            isinstance(row['score'], schema.column_types['score'])
            for row in dataset
        ))
        self.assertTrue(all(
            isinstance(row['flag'], schema.column_types['flag'])
            for row in dataset
        ))

    @TEST_TAG('ray')
    def test_schema_single_dataset(self):
        """Test schema for single dataset"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [{'text': 'hello', 'score': 1}]
        dataset = RayDataset(ray.data.from_items(data))

        schema = dataset.schema()
        self.assertIn('text', schema.columns)
        self.assertEqual(schema.column_types['text'], str)
        self.assertEqual(schema.column_types['score'], int)

    @TEST_TAG('ray')
    def test_schema_multiple_datasets(self):
        """Test schema consistency across multiple datasets"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data1 = [{'text': 'hello', 'score': 1}]
        data2 = [{'text': 'world', 'score': 2}]

        dataset1 = RayDataset(ray.data.from_items(data1))
        dataset2 = RayDataset(ray.data.from_items(data2))

        # Verify schemas match
        schema1 = dataset1.schema()
        schema2 = dataset2.schema()

        self.assertEqual(schema1.columns, schema2.columns)
        self.assertEqual(schema1.column_types, schema2.column_types)

    @TEST_TAG('ray')
    def test_schema_validation(self):
        """Test schema validation"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        # Test with invalid schema (mixed types)
        data = [
            {'text': 'hello', 'value': 1},
            {'text': 'world', 'value': 'string'}  # Mixed int/str
        ]

        with self.assertRaises(Exception) as context:
            dataset = RayDataset(ray.data.from_items(data))
            _ = dataset.schema()

        # Ray might choose either type
        self.assertIn("unable to merge", str(context.exception).lower())

    @TEST_TAG('ray')
    def test_schema_nested_structures(self):
        """Test schema with nested data structures"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [{
            'text': 'hello',
            'int_value': 1,
            'float_value': 1.0,
            'bool_value': True,
            'metadata': {'lang': 'en', 'score': 1},
            'tags': ['tag1', 'tag2'],
            'nested': {'a': {'b': {'c': 1}}}
        }]

        dataset = RayDataset(ray.data.from_items(data))
        schema = dataset.schema()

        self.assertEqual(schema.column_types['text'], str)
        self.assertEqual(schema.column_types['int_value'], int)
        self.assertEqual(schema.column_types['float_value'], float)
        self.assertEqual(schema.column_types['bool_value'], bool)
        self.assertIsInstance(schema.column_types['metadata'], Schema)
        self.assertEqual(schema.column_types['tags'], List[str])
        self.assertIsInstance(schema.column_types['nested'], Schema)

    @TEST_TAG('ray')
    def test_schema_empty_dataset(self):
        """Test schema with empty dataset"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset

        with self.assertRaises(ValueError) as context:
            dataset = RayDataset(ray.data.from_items([]))
            _ = dataset.schema()

        self.assertIn("empty", str(context.exception).lower())

    @TEST_TAG('ray')
    def test_schema_special_characters(self):
        """Test schema with special characters in column names"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [{
            'normal': 1,
            'with.dot': 2,
            'with-dash': 3,
            '_underscore': 4,
            'with space': 5
        }]

        dataset = RayDataset(ray.data.from_items(data))
        schema = dataset.schema()

        expected_columns = {
            'normal', 'with.dot', 'with-dash',
            '_underscore', 'with space'
        }
        self.assertEqual(set(schema.columns), expected_columns)

    @TEST_TAG('ray')
    def test_schema_type_consistency(self):
        """Test schema type consistency across rows"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [
            {'text': 'hello', 'score': 1, 'flag': True},
            {'text': 'world', 'score': 2, 'flag': False},
            {'text': 'test', 'score': 3, 'flag': True}
        ]

        dataset = RayDataset(ray.data.from_items(data))
        schema = dataset.schema()

        # Get all rows for type checking
        rows = list(dataset.data.take())

        # Verify types are consistent
        self.assertTrue(all(
            isinstance(row['text'], schema.column_types['text'])
            for row in rows
        ))
        self.assertTrue(all(
            isinstance(row['score'], schema.column_types['score'])
            for row in rows
        ))
        self.assertTrue(all(
            isinstance(row['flag'], schema.column_types['flag'])
            for row in rows
        ))

    def test_type_mapping(self):
        # hf type
        from datasets import Value, Sequence, Array2D, ClassLabel
        test_mapping = [
            (Value('string'), str),
            (Sequence(Value('int32')), List[int]),
            (Array2D((2, 3), dtype='float32'), List[float]),
            ({'a': Value('bool')}, Schema(column_types={'a': bool}, columns=['a'])),
            (ClassLabel(num_classes=2), int),
            (None, Any)
        ]
        for ori, tgt in test_mapping:
            self.assertEqual(Schema.map_hf_type_to_python(ori), tgt)

        # ray/arrow type
        import pyarrow as pa
        test_mapping = [
            (pa.string(), str),
            (pa.list_(pa.int32()), List[int]),
            (pa.struct({'a': pa.bool_()}), Schema(column_types={'a': bool}, columns=['a'])),
            (pa.binary(10), bytes),
            (pa.float64(), float),
            (pa.list_(pa.float64()), List[float]),
            (pa.map_(pa.string(), pa.int32()), dict),
            (pa.date32(), Any)
        ]
        for ori, tgt in test_mapping:
            self.assertEqual(Schema.map_ray_type_to_python(ori), tgt)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            Schema(column_types={'a': int}, columns=['a', 'b'])

    def test_convert_to_str(self):
        s = Schema(column_types={'a': int}, columns=['a'])
        ss = str(s)
        self.assertIsInstance(ss, str)
        self.assertIn("-" * 40, ss)
        self.assertIn("Dataset Schema", ss)
        for col in s.columns:
            self.assertIn(col, ss)


if __name__ == '__main__':
    unittest.main()
