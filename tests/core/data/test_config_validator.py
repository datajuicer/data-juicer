import unittest
from data_juicer.core.data.config_validator import ConfigValidator, ConfigValidationError
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class TestConfigValidator(DataJuicerTestCaseBase):

    def test_config_validator(self):
        cfg = {
            "key1": "val1",
            "key2": 2,
            "key3": [1, 2, 3],
        }
        validator = ConfigValidator()
        # no raise
        validator.validate_config(cfg)

        # required fields
        validator.CONFIG_VALIDATION_RULES["required_fields"] = ["key1", "key2", "key3", "key4"]
        with self.assertRaises(ConfigValidationError) as ctx:
            validator.validate_config(cfg)
        self.assertIn("Missing required fields", str(ctx.exception))
        validator.CONFIG_VALIDATION_RULES["required_fields"] = []

        # field type
        validator.CONFIG_VALIDATION_RULES["field_types"] = {"key1": int}
        with self.assertRaises(ConfigValidationError) as ctx:
            validator.validate_config(cfg)
        self.assertIn("must be of type", str(ctx.exception))
        validator.CONFIG_VALIDATION_RULES["field_types"] = {}

        # custom validator
        validator.CONFIG_VALIDATION_RULES["custom_validators"] = {"key1": lambda x: x == "val1", "key2": lambda x: x / 0}
        with self.assertRaises(ConfigValidationError) as ctx:
            validator.validate_config(cfg)
        self.assertIn("Validation failed", str(ctx.exception))
        validator.CONFIG_VALIDATION_RULES["custom_validators"] = {}


if __name__ == '__main__':
    unittest.main()
