from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from data_juicer.core.data.dj_dataset import DJDataset


class DataValidator(ABC):
    """Base class for data validation"""

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def validate(self, dataset: DJDataset) -> None:
        """
        Validate dataset content

        Args:
            dataset: The dataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(dataset, DJDataset):
            raise DataValidationError("unsupported dataset type; must be a DJDataset")


class DataValidationError(Exception):
    """Custom exception for data validation errors"""

    pass


class DataValidatorRegistry:
    """Registry for data validators"""

    _validators: Dict[str, Type[DataValidator]] = {}

    @classmethod
    def register(cls, validator_type: str):
        def decorator(validator_class: Type[DataValidator]):
            cls._validators[validator_type] = validator_class
            return validator_class

        return decorator

    @classmethod
    def get_validator(cls, validator_type: str) -> Optional[Type[DataValidator]]:
        return cls._validators.get(validator_type)


class BaseConversationValidator(DataValidator):
    """Base class for conversation validators"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_turns = config.get("min_turns", 1)
        self.max_turns = config.get("max_turns", 100)
        self.sample_size = config.get("sample_size", 100)

    def validate(self, dataset: DJDataset) -> None:
        """Base validation for all conversation formats"""
        super().validate(dataset)

        for item in dataset.get(self.sample_size):
            self.validate_conversation(item)

    @abstractmethod
    def validate_conversation(self, data: Dict) -> None:
        """Validate specific conversation format"""


@DataValidatorRegistry.register("swift_messages")
class SwiftMessagesValidator(BaseConversationValidator):
    """Validator for Swift Messages conversation format.

    This validator ensures conversations follow the Swift Messages format with
    proper message structure and role assignments.

    Args:
        config (Dict): Configuration dictionary containing:
            min_turns (int, optional): Minimum number of messages.
                Defaults to 1.
            max_turns (int, optional): Maximum number of messages.
                Defaults to 100.
            sample_size (int, optional): Number of samples to validate.
                Defaults to 100.

    Example Format:
        .. code-block:: python

            {
                "messages": [
                    {"role": "system", "content": "<system>"},
                    {"role": "user", "content": "<query>"},
                    {"role": "assistant", "content": "<response>"},
                    ...
                ]
            }

    Raises:
        DataValidationError: If validation fails due to:
            - Missing 'messages' field
            - Invalid message structure
            - Invalid role values
            - Missing content
            - Message count outside allowed range
    """

    def validate_conversation(self, data: Dict) -> None:
        if "messages" not in data:
            raise DataValidationError("Missing 'messages' field")

        messages = data["messages"]
        if not isinstance(messages, list):
            raise DataValidationError("'messages' must be an array")

        if not (self.min_turns <= len(messages) <= self.max_turns):
            raise DataValidationError(
                f"Conversation must have between {self.min_turns} and " f"{self.max_turns} messages"
            )

        # each message should have a role and content
        # and role should be one of system, user, assistant
        for msg in messages:
            if "role" not in msg or msg["role"] is None:
                raise DataValidationError("Missing 'role' field")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise DataValidationError("Invalid 'role' field")
            if "content" not in msg or msg["content"] is None:
                raise DataValidationError("Missing 'content' field")


@DataValidatorRegistry.register("dj_conversation")
class DataJuicerFormatValidator(BaseConversationValidator):
    """Validator for Data-Juicer default conversation format.

    This validator ensures conversations follow the Data-Juicer format with
    proper fields and structure.

    Args:
        config (Dict): Configuration dictionary containing:
            min_turns (int, optional): Minimum number of conversation turns.
                Defaults to 1.
            max_turns (int, optional): Maximum number of conversation turns.
                Defaults to 100.
            sample_size (int, optional): Number of samples to validate.
                Defaults to 100.

    Example Format:
        .. code-block:: python

            {
                "system": "<system>",  # Optional
                "instruction": "<query-inst>",
                "query": "<query2>",
                "response": "<response2>",
                "history": [  # Optional
                    ["<query1>", "<response1>"],
                    ...
                ]
            }

    Raises:
        DataValidationError: If validation fails due to:
            - Missing required fields
            - Invalid field types
            - Invalid conversation structure
            - Turn count outside allowed range
    """

    def validate_conversation(self, data: Dict) -> None:
        # Validate system if present
        if "system" in data:
            if not isinstance(data["system"], str):
                raise DataValidationError("'system' must be string")

        # Validate required fields
        for field in ["instruction", "query", "response"]:
            if field not in data:
                raise DataValidationError(f"Missing '{field}' field")
            if not isinstance(data[field], str):
                raise DataValidationError(f"'{field}' must be string")

        # Validate history if present
        if "history" in data:
            if not isinstance(data["history"], list):
                raise DataValidationError("'history' must be an array")

            # Count total turns including current query/response
            total_turns = len(data["history"]) + 1
            if not (self.min_turns <= total_turns <= self.max_turns):
                raise DataValidationError(
                    f"Conversation must have between {self.min_turns} and " f"{self.max_turns} turns including history"
                )

            # Validate each history turn
            for i, turn in enumerate(data["history"]):
                if not isinstance(turn, list) or len(turn) != 2:
                    raise DataValidationError(f"History turn {i} must be [query, response] pair")
                if not isinstance(turn[0], str):
                    raise DataValidationError(f"Query in history turn {i} must be string")
                if not isinstance(turn[1], str):
                    raise DataValidationError(f"Response in history turn {i} must be string")


@DataValidatorRegistry.register("code")
class CodeDataValidator(DataValidator):
    """Validator for code data"""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.required_columns = ["code", "language"]
        self.supported_languages = config.get("supported_languages", [])

    def validate(self, dataset: DJDataset) -> None:
        # Implement code-specific validation logic...
        pass


@DataValidatorRegistry.register("required_fields")
class RequiredFieldsValidator(DataValidator):
    """Validator that checks for required fields in dataset.

    This validator ensures that specified fields exist in the dataset and
    optionally checks their types and missing value ratios.

    Args:
        config (Dict): Configuration dictionary containing:
            required_fields (List[str]): List of field names that must exist
            field_types (Dict[str, type], optional): Map of field names to
            expected types allow_missing (float, optional): Maximum ratio of
            missing values allowed. Defaults to 0.0.

    Example Config:
        .. code-block:: python

            {
                "required_fields": ["field1", "field2"],
                "field_types": {"field1": str, "field2": int},
                "allow_missing": 0.0
            }

    Raises:
        DataValidationError: If validation fails
    """

    def __init__(self, config: Dict):
        """
        Initialize validator with config

        Args:
            config: Dict containing:
                - required_fields: List of field names that must exist
                - field_types: Optional map of field names to expected types
                - allow_missing: Optional float for max ratio missing allowed
        """
        super().__init__(config)

        self.required_fields = config["required_fields"]
        self.field_types = config.get("field_types", {})
        # Default no missing allowed
        self.allow_missing = config.get("allow_missing", 0.0)
        self.sample_size = config.get("sample_size", 100)

    def validate(self, dataset: DJDataset) -> None:
        """
        Validate dataset has required fields with correct types

        Args:
            dataset: NestedDataset or RayDataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        super().validate(dataset)

        # Check if fields exist in dataset
        available_fields = set(dataset.schema().columns)

        missing_fields = set(self.required_fields) - available_fields
        if missing_fields:
            raise DataValidationError(f"Dataset missing required fields: {missing_fields}")

        # Check field types and missing values
        for field in self.required_fields:
            # Get expected type if specified
            expected_type = self.field_types.get(field)

            # Sample head part of data for validation
            sample_values = dataset.get_column(field, self.sample_size)

            # Check for missing values
            missing_count = sum(1 for v in sample_values if v is None)
            missing_ratio = missing_count / len(sample_values)
            if missing_ratio > self.allow_missing:
                raise DataValidationError(
                    f"Field '{field}' has {missing_ratio:.1%} missing values, "
                    f"exceeding allowed {self.allow_missing:.1%}"
                )

            # Check types if specified
            if expected_type:
                invalid_types = [type(v) for v in sample_values if v is not None and not isinstance(v, expected_type)]
                if invalid_types:
                    raise DataValidationError(
                        f"Field '{field}' contains values of incorrect type. "
                        f"Expected {expected_type.__name__}, "
                        f"got {set(t.__name__ for t in invalid_types)}"
                    )
