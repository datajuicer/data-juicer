from unittest import main
import datasets
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from data_juicer.core.data import NestedDataset
from data_juicer.core.data.data_validator import (
    DataValidationError, 
    RequiredFieldsValidator,
    SwiftMessagesValidator,
    DataJuicerFormatValidator,
    DataValidatorRegistry
)


# Test RequiredFieldsValidator
class RequiredFieldsValidatorTest(DataJuicerTestCaseBase):
    
    def setUp(self):
        super().setUp()

        # Create sample DataFrame
        self.data = [
            {
                'text': 'Hello',
                'metadata': {'lang': 'en'},
                'score': 1.0
            },
            {
                'text': 'World',
                'metadata': {'lang': 'es'},
                'score': 2.0
            },
            {
                'text': None,
                'metadata': {'lang': 'fr'},
                'score': 3.0
            },
            {
                'text': 'Test',
                'metadata': None,
                'score': 4.0
            }
        ]

        # Create dataset
        self.dataset = NestedDataset(datasets.Dataset.from_list(self.data))

    def test_basic_validation(self):
        """Test basic field validation"""
        config = {
            'type': 'required_fields',
            'required_fields': ['text', 'metadata'],
            'allow_missing': .25
        }
        validator = DataValidatorRegistry.get_validator(config['type'])(config)
        
        # Should pass
        validator.validate(self.dataset)
        
        # Should fail with missing field
        config['required_fields'].append('nonexistent')
        validator = RequiredFieldsValidator(config)
        with self.assertRaises(DataValidationError) as exc:
            validator.validate(self.dataset)
        self.assertIn("missing required fields", str(exc.exception).lower())

    def test_type_validation(self):
        """Test field type validation"""
        # Should pass        
        config = {
            'required_fields': ['text', 'score'],
            'field_types': {
                'text': str,
                'score': float
            },
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        validator.validate(self.dataset)

        # Should fail with wrong type
        config['field_types']['score'] = str
        validator = RequiredFieldsValidator(config)
        with self.assertRaises(DataValidationError) as exc:
            validator.validate(self.dataset)
        self.assertIn("incorrect type", str(exc.exception).lower())

    @TEST_TAG('ray')
    def test_ray_dataset_support(self):
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        
        # Create ray dataset
        ray_dataset = RayDataset(ray.data.from_items(self.data))

        """Test validation with RayDataset"""
        config = {
            'required_fields': ['text', 'metadata'],
            'field_types': {
                'text': str,
                'metadata': dict
            },
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        
        # Should pass
        validator.validate(ray_dataset)

    def test_invalid_dataset_type(self):
        """Test validation with unsupported dataset type"""
        config = {
            'required_fields': ['text']
        }
        validator = RequiredFieldsValidator(config)
        
        with self.assertRaises(DataValidationError) as exc:
            validator.validate([1, 2, 3])  # Invalid dataset type
        self.assertIn("unsupported dataset type", str(exc.exception).lower())

    def test_empty_required_fields(self):
        """Test validation with empty required fields"""
        config = {
            'required_fields': []
        }
        validator = RequiredFieldsValidator(config)
        
        # Should pass as no fields are required
        validator.validate(self.dataset)

    @TEST_TAG('ray')
    def test_multiple_dataset_types(self):
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset

        # Create ray dataset
        ray_dataset = RayDataset(ray.data.from_items(self.data))

        """Test validation works with different dataset types"""
        datasets_to_test = [
            ('nested', self.dataset),
            ('ray', ray_dataset)
        ]
        
        config = {
            'required_fields': ['text', 'metadata', 'score'],
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        
        for name, dataset in datasets_to_test:
            with self.subTest(dataset_type=name):
                validator.validate(dataset)


class TestSwiftMessagesValidator(DataJuicerTestCaseBase):
    def setUp(self):
        """Setup test validator"""
        super().setUp()
        self.config = {'min_turns': 1, 'max_turns': 5}
        self.validator = SwiftMessagesValidator(self.config)

    def test_valid_conversation_with_system(self):
        """Test valid conversation with system message"""
        data = {
            'messages': [
                {'role': 'system', 'content': 'Be helpful'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there'},
                {'role': 'user', 'content': 'How are you?'},
                {'role': 'assistant', 'content': 'I am good!'}
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        self.validator.validate(dataset)  # Should not raise

    def test_valid_conversation_without_system(self):
        """Test valid conversation without system message"""
        data = {
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there'}
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        self.validator.validate(dataset)  # Should not raise

    def test_missing_messages(self):
        """Test conversation with missing messages field"""
        data = {'random': 'random_value'}
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("missing 'messages' field", str(exc.exception).lower())

    def test_invalid_messages_type(self):
        """Test conversation with non-array messages"""
        data = {'messages': 'not an array'}
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("must be an array", str(exc.exception).lower())

    def test_empty_messages(self):
        """Test conversation with empty messages"""
        data = {'messages': []}
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("must have between", str(exc.exception).lower())

    def test_too_many_messages(self):
        """Test conversation exceeding max_turns"""
        messages = []
        for i in range(6):  # 6 pairs = 12 messages
            messages.extend([
                {'role': 'user', 'content': f'user_{i}'},
                {'role': 'assistant', 'content': f'assistant_{i}'}
            ])
        data = {'messages': messages}
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("must have between", str(exc.exception).lower())

    def test_missing_content(self):
        """Test message with missing content"""
        data = {
            'messages': [
                {'role': 'user'},  # Missing content field
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("Missing 'content' field", str(exc.exception))

    def test_missing_role(self):
        """Test message with missing role"""
        data = {
            'messages': [
                {'content': 'Hello'},  # Missing role field
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("Missing 'role' field", str(exc.exception))

    def test_invalid_role(self):
        """Test message with invalid role"""
        data = {
            'messages': [
                {'role': 'invalid', 'content': 'Hello'},  # Invalid role
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("Invalid 'role' field", str(exc.exception))

    def test_non_string_content(self):
        """Test message with non-string content"""
        data = {
            'messages': [
                {'role': 'user', 'content': 123},
                {'role': 'assistant', 'content': 'Hi'}
            ]
        }
        with self.assertRaises(Exception) as exc:
            # from_list will raise an exception if the content has inconsistent types
            dataset = NestedDataset(datasets.Dataset.from_list([data]))
            # self.validator.validate(dataset)
        self.assertIn("could not convert", str(exc.exception).lower())


class TestDataJuicerFormatValidator(DataJuicerTestCaseBase):
    def setUp(self):
        """Setup test validator"""
        super().setUp()
        self.config = {'min_turns': 1, 'max_turns': 5}
        self.validator = DataJuicerFormatValidator(self.config)

    def test_valid_conversation_with_system(self):
        """Test valid conversation with system message"""
        data = {
            'system': 'Be helpful',
            'instruction': 'Help me with this',
            'query': 'How do I code?',
            'response': 'Here is how...',
            'history': [
                ['What is Python?', 'Python is a programming language']
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        self.validator.validate(dataset)  # Should not raise

    def test_valid_conversation_without_system(self):
        """Test valid conversation without optional fields"""
        data = {
            'instruction': 'Help me',
            'query': 'How do I code?',
            'response': 'Here is how...'
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        self.validator.validate(dataset)  # Should not raise

    def test_missing_required_field(self):
        """Test conversation with missing required field"""
        data = {
            'instruction': 'Help me',
            'query': 'How do I code?'
            # missing 'response'
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("Missing 'response' field", str(exc.exception))

    def test_invalid_field_type(self):
        """Test conversation with invalid field type"""
        data = {
            'instruction': 'Help me',
            'query': 123,  # should be string
            'response': 'Here is how...'
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("'query' must be string", str(exc.exception))

    def test_invalid_history_format(self):
        """Test conversation with invalid history format"""
        data = {
            'instruction': 'Help me',
            'query': 'How do I code?',
            'response': 'Here is how...',
            'history': [
                ['Single element']  # should be [query, response] pair
            ]
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("must be [query, response] pair", str(exc.exception))

    def test_too_many_turns(self):
        """Test conversation exceeding max_turns"""
        history = [['Q1', 'A1'], ['Q2', 'A2'], ['Q3', 'A3'], 
                  ['Q4', 'A4'], ['Q5', 'A5']]  # 5 history turns + 1 current = 6
        data = {
            'instruction': 'Help me',
            'query': 'How do I code?',
            'response': 'Here is how...',
            'history': history
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("must have between", str(exc.exception))

    def test_invalid_history_types(self):
        """Test conversation with invalid types in history"""
        data = {
            'instruction': 'Help me',
            'query': 'How do I code?',
            'response': 'Here is how...',
            'history': [
                [123, 456]  # query should be string
            ]
        }

        with self.assertRaises(DataValidationError) as exc:
            # from_list will raise an exception if the content has inconsistent types
            dataset = NestedDataset(datasets.Dataset.from_list([data]))
            self.validator.validate(dataset)
        self.assertIn("must be string", str(exc.exception).lower())

    def test_invalid_system_type(self):
        """Test conversation with invalid system type"""
        data = {
            'system': 123,  # should be string
            'instruction': 'Help me',
            'query': 'How do I code?',
            'response': 'Here is how...'
        }
        dataset = NestedDataset(datasets.Dataset.from_list([data]))
        with self.assertRaises(DataValidationError) as exc:
            self.validator.validate(dataset)
        self.assertIn("'system' must be string", str(exc.exception))

    def test_invalid_history_array(self):
        """Test valid conversation with system message"""
        data = {
            'system': 'Be helpful',
            'instruction': 'Help me with this',
            'query': 'How do I code?',
            'response': 'Here is how...',
            'history': 'history_str',
        }
        with self.assertRaises(DataValidationError) as exc:
            # from_list will raise an exception if the content has inconsistent types
            dataset = NestedDataset(datasets.Dataset.from_list([data]))
            self.validator.validate(dataset)
        self.assertIn("must be an array", str(exc.exception).lower())

    def test_missing_rate_exceeded(self):
        data = {
            'instruction': ['Help me with this', 'other inst', None, None, None],
            'query': ['How do I code?', None, None, None, None],
            'response': ['Here is how...', 'Here is how...', 'Here is how...', 'Here is how...', 'Here is how...'],
        }
        config = {
            'required_fields': ['query', 'response'],
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        with self.assertRaises(DataValidationError) as exc:
            # from_list will raise an exception if the content has inconsistent types
            dataset = NestedDataset(datasets.Dataset.from_dict(data))
            validator.validate(dataset)
        self.assertIn("missing values, exceeding allowed", str(exc.exception).lower())


if __name__ == '__main__':
    main()
