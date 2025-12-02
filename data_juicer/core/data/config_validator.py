from typing import Dict


class ConfigValidationError(Exception):
    """Custom exception for validation errors"""

    pass


class ConfigValidator:
    """Mixin class for configuration validation"""

    # Define validation rules for each strategy type
    CONFIG_VALIDATION_RULES = {
        "required_fields": [],  # Fields that must be present
        "optional_fields": [],  # Fields that are optional
        "field_types": {},  # Expected types for fields
        "custom_validators": {},  # Custom validation functions
    }

    def validate_config(self, ds_config: Dict) -> None:
        """
        Validate the configuration dictionary.

        Args:
            ds_config: Configuration dictionary to validate

        Raises:
            ValidationError: If validation fails
        """
        # Check required fields
        missing_fields = [field for field in self.CONFIG_VALIDATION_RULES["required_fields"] if field not in ds_config]
        if missing_fields:
            raise ConfigValidationError(f"Missing required fields: {', '.join(missing_fields)}")

        # Optional fields
        # no need for any special checks

        # Check field types
        for field, expected_type in self.CONFIG_VALIDATION_RULES["field_types"].items():
            if field in ds_config:
                value = ds_config[field]
                if not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Field '{field}' must be of "
                        f"type '{expected_type.__name__}', "
                        f"got '{type(value).__name__}'"
                    )

        # Run custom validators
        for field, validator in self.CONFIG_VALIDATION_RULES["custom_validators"].items():
            if field in ds_config:
                try:
                    validator(ds_config[field])
                except Exception as e:
                    raise ConfigValidationError(f"Validation failed for field '{field}': {str(e)}")
