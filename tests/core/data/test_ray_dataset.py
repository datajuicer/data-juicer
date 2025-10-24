import unittest
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

class TestRayDataset(DataJuicerTestCaseBase):
    def setUp(self):
        """Set up test data"""
        super().setUp()

        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

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

        # Create fresh dataset for each test
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up test data"""
        self.dataset = None
        super().tearDown()

    @TEST_TAG('ray')
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

    @TEST_TAG('ray')
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

    @TEST_TAG('ray')
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

    @TEST_TAG('ray')
    def test_get_column_empty_dataset(self):
        """Test with empty dataset"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        empty_dataset = RayDataset(ray.data.from_items([]))

        # Should raise ValuError for empty dataset/columns
        with self.assertRaises(KeyError):
            empty_dataset.get_column('text')

    @TEST_TAG('ray')
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

    @TEST_TAG('ray')
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

    @TEST_TAG('ray')
    def test_get(self):
        """Test get method for RayDataset"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        # Test with simple data
        simple_data = [
            {'text': 'hello', 'score': 1},
            {'text': 'world', 'score': 2},
            {'text': 'test', 'score': 3}
        ]
        dataset = RayDataset(ray.data.from_items(simple_data))

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
        nested_dataset = RayDataset(ray.data.from_items(nested_data))

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


if __name__ == '__main__':
    unittest.main()
