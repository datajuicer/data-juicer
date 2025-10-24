import os
import unittest
import tempfile
import shutil

from data_juicer.format.json_formatter import JsonFormatter
from data_juicer.format.load import load_formatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LoadFormatterTest(DataJuicerTestCaseBase):
    """Test cases specifically for the load_formatter function"""

    def setUp(self):
        super().setUp()

        # Setup test data paths
        self._path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'data', 'structured')
        self._json_file = os.path.join(self._path, 'demo-dataset.jsonl')
        self._csv_file = os.path.join(self._path, 'demo-dataset.csv')
        self._temp_dir = tempfile.mkdtemp()
        self._complex_ext_file = os.path.join(self._temp_dir, 'test.jsonl.zst')
        
        # Create a test file with complex extension
        with open(self._complex_ext_file, 'w') as f:
            f.write('{"text": "test"}\n')
        
        # Create a directory structure for relative path testing
        self._rel_path_dir = os.path.join(self._temp_dir, 'rel_path_test')
        os.makedirs(self._rel_path_dir, exist_ok=True)
        
        # Create a test JSONL file in the relative path directory
        self._rel_json_file = os.path.join(self._rel_path_dir, 'test_rel.jsonl')
        with open(self._rel_json_file, 'w') as f:
            f.write('{"text": "relative path test"}\n')
        
        # Save current working directory
        self._original_cwd = os.getcwd()

    def tearDown(self):
        # Restore original working directory
        os.chdir(self._original_cwd)
        
        # Clean up any temp files
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

        super().tearDown()

    def test_load_formatter_with_json_file(self):
        """Test loading a JSONL file directly"""
        formatter = load_formatter(self._json_file)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)

    def test_load_formatter_with_directory(self):
        """Test loading a directory with mixed file types"""
        formatter = load_formatter(self._path)
        # Should pick the formatter with most matching files
        ds = formatter.load_dataset()
        self.assertTrue(len(ds) > 0)

    def test_load_formatter_with_specific_suffix(self):
        """Test loading a file with specific suffix"""
        formatter = load_formatter(self._json_file, suffixes=['.jsonl'])
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        
    def test_load_formatter_with_complex_extension(self):
        """Test loading a file with a complex extension like jsonl.zst
            which is not supported by json formatter"""
        with self.assertRaises(ValueError):
            formatter = load_formatter(self._complex_ext_file)
        
    def test_load_formatter_with_nonexistent_file(self):
        """Test handling of nonexistent files"""
        with self.assertRaises(ValueError):
            load_formatter(os.path.join(self._temp_dir, 'nonexistent.jsonl'))
    
    def test_load_formatter_with_stacktrace_scenario(self):
        """Specifically test the scenario from the error stacktrace"""
        # Create a temp directory with the name that matches the error
        temp_path = os.path.join(self._temp_dir, 'data')
        os.makedirs(temp_path, exist_ok=True)
        
        # Create the file with the name from the error
        test_file = os.path.join(temp_path, 'redpajama-cc-2023-06-refined.jsonl')
        with open(test_file, 'w') as f:
            f.write('{"text": "test content"}\n')
        
        # Now try to load it
        formatter = load_formatter(test_file)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
        
        # Test with a directory containing the file
        formatter = load_formatter(temp_path)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
    
    def test_load_formatter_with_relative_path(self):
        """Test loading a file using a relative path"""
        # Change to the temp directory
        os.chdir(self._temp_dir)
        
        # Use a relative path to the test file
        rel_path = os.path.join('rel_path_test', 'test_rel.jsonl')
        
        # Try to load using the relative path
        formatter = load_formatter(rel_path)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]['text'], 'relative path test')
    
    def test_load_formatter_with_relative_directory_path(self):
        """Test loading a directory using a relative path"""
        # Change to the temp directory
        os.chdir(self._temp_dir)
        
        # Use a relative path to the directory
        rel_dir_path = 'rel_path_test'
        
        # Try to load using the relative directory path
        formatter = load_formatter(rel_dir_path)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]['text'], 'relative path test')


if __name__ == '__main__':
    unittest.main() 