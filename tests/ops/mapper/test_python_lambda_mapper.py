import unittest

from data_juicer.ops.mapper.python_lambda_mapper import PythonLambdaMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class PythonLambdaMapperMapper(DataJuicerTestCaseBase):

    def test_lambda_function_batched(self):
        mapper = PythonLambdaMapper("lambda d: {'value': d['value'] + [6]}", batched=True)  # Append '6' to value
        result = mapper.process_batched({'value': [5]})
        self.assertEqual(result, {'value': [5, 6]})

    def test_lambda_modifies_values(self):
        mapper = PythonLambdaMapper("lambda d: {'value': d['value'] + 1}")  # '+1' to 'value'
        result = mapper.process_single({'value': 5})
        self.assertEqual(result, {'value': 6})

    def test_lambda_combines_values(self):
        mapper = PythonLambdaMapper("lambda d: {'combined': d['a'] + d['b']}")
        result = mapper.process_single({'a': 3, 'b': 7})
        self.assertEqual(result, {'combined': 10})

    def test_lambda_swaps_values(self):
        mapper = PythonLambdaMapper("lambda d: {'a': d['b'], 'b': d['a']}")
        result = mapper.process_single({'a': 1, 'b': 2})
        self.assertEqual(result, {'a': 2, 'b': 1})

    def test_lambda_result_is_not_dict(self):
        mapper = PythonLambdaMapper("lambda d: d['value'] + 1")  # This returns an int
        with self.assertRaises(ValueError) as cm:
            mapper.process_single({'value': 10})
        self.assertIn("Lambda function must return a dictionary, got int instead.", str(cm.exception))

    def test_invalid_syntax(self):
        with self.assertRaises(ValueError) as cm:
            PythonLambdaMapper("invalid lambda")  # Invalid syntax
        self.assertIn("Invalid lambda function", str(cm.exception))

    def test_invalid_expression(self):
        with self.assertRaises(ValueError) as cm:
            PythonLambdaMapper("3 + 5")  # Not a lambda
        self.assertIn("Input string must be a valid lambda function.", str(cm.exception))

    def test_lambda_with_multiple_arguments(self):
        with self.assertRaises(ValueError) as cm:
            PythonLambdaMapper("lambda x, y: {'sum': x + y}")  # Creating a lambda accepts two arguments
        self.assertIn("Lambda function must have exactly one argument.", str(cm.exception))

    def test_lambda_returning_unexpected_structure(self):
        mapper = PythonLambdaMapper("lambda d: ({'value': d['value']}, {'extra': d['extra']})")  # Invalid return type; too many dictionaries
        with self.assertRaises(ValueError) as cm:
            mapper.process_single({'value': 5, 'extra': 10})
        self.assertIn("Lambda function must return a dictionary, got tuple instead.", str(cm.exception))

    def test_lambda_modifies_in_place_and_returns(self):
        mapper = PythonLambdaMapper("lambda d: d.update({'new_key': 'added_value'}) or d")  # Returns the modified dictionary
        sample_dict = {'value': 3}
        result = mapper.process_single(sample_dict)
        self.assertEqual(result, {'value': 3, 'new_key': 'added_value'})  # Ensure the update worked

    def test_lambda_function_with_no_operation(self):
        mapper = PythonLambdaMapper("lambda d: d")  # Simply returns the input dictionary
        sample_dict = {'key': 'value'}
        result = mapper.process_single(sample_dict)
        self.assertEqual(result, {'key': 'value'})  # Unchanged

if __name__ == '__main__':
    unittest.main()