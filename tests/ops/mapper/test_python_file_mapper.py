import unittest
import tempfile

from data_juicer.ops.mapper.python_file_mapper import PythonFileMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class TestPythonFileMapper(DataJuicerTestCaseBase):

    def test_function_execution(self):
        """Test the correct execution of a loadable function."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write(
                "def process_data(sample):\n"
                "    return {'result': sample['value'] + 10}\n"
            )
            temp_file.seek(0)  # Rewind the file so it can be read
            mapper = PythonFileMapper(temp_file.name, "process_data")
            result = mapper.process_single({'value': 5})
            self.assertEqual(result, {'result': 15})

    def test_function_batched(self):
        """Test for a function that processes a batch."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write(
                "def process_data(samples):\n"
                "    return {'result': samples['value'] + [10]}\n"
            )
            temp_file.seek(0)  # Rewind the file so it can be read
            mapper = PythonFileMapper(temp_file.name, "process_data", batched=True)
            result = mapper.process_batched({'value': [5]})
            self.assertEqual(result, {'result': [5, 10]})

    def test_function_with_import(self):
        """Test for a function that contains an import statement."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write(
                "import numpy as np\n"
                "def process_data(sample):\n"
                "    return {'result': np.sum([sample['value'], 10])}\n"
            )
            temp_file.seek(0)  # Rewind the file so it can be read
            mapper = PythonFileMapper(temp_file.name, "process_data")
            result = mapper.process_single({'value': 5})
            self.assertEqual(result, {'result': 15})

    def test_file_not_found(self):
        """Test for a non-existent file."""
        with self.assertRaises(FileNotFoundError) as cm:
            PythonFileMapper("non_existent.py", "process_data")
        self.assertIn("does not exist", str(cm.exception))

    def test_file_not_python_extension(self):
        """Test for a file that exists but is not a .py file."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt', mode='w+') as temp_file:
            temp_file.write("This is a text file.")
            temp_file.seek(0)  # Rewind the file so it can be read
            with self.assertRaises(ValueError) as cm:
                PythonFileMapper(temp_file.name, "some_function")
            self.assertIn("is not a Python file", str(cm.exception))

    def test_function_not_found(self):
        """Test for function not existing in the provided file."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write(
                "def existing_function(sample):\n"
                "    return sample\n"
            )
            temp_file.seek(0)  # Rewind the file so it can be read
            with self.assertRaises(ValueError) as cm:
                PythonFileMapper(temp_file.name, "non_existing_function")
            self.assertIn("not found", str(cm.exception))

    def test_function_not_callable(self):
        """Test for trying to load a non-callable function."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write("x = 42")
            temp_file.seek(0)  # Rewind the file so it can be read
            with self.assertRaises(ValueError) as cm:
                PythonFileMapper(temp_file.name, "x")
            self.assertIn("not callable", str(cm.exception))

    def test_function_mutiple_arguments(self):
        """Test for function that requires more than one argument."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write(
                "def multi_arg_function(arg1, arg2):\n"
                "    return arg1 + arg2\n"
            )
            temp_file.seek(0)  # Rewind the file so it can be read
            with self.assertRaises(ValueError) as cm:
                PythonFileMapper(temp_file.name, "multi_arg_function")
            self.assertIn("must take exactly one argument", str(cm.exception))

    def test_invalid_return_type(self):
        """Test for a function returning a non-dictionary."""
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+') as temp_file:
            temp_file.write(
                "def invalid_function(sample):\n"
                "    return sample['value'] + 5\n"
            )
            temp_file.seek(0)  # Rewind the file so it can be read
            mapper = PythonFileMapper(temp_file.name, "invalid_function")
            with self.assertRaises(ValueError) as cm:
                mapper.process_single({'value': 5})
            self.assertIn("Function must return a dictionary, got int instead.", str(cm.exception))

if __name__ == '__main__':
    unittest.main()