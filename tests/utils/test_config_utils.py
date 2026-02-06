"""
Tests for configuration utilities.

Tests cover:
- ConfigAccessor.get() method for dict and object configs
- ConfigAccessor.get_nested() method for nested configurations
- Edge cases (None config, missing keys, empty configs)
- Default value handling
- Type safety and error conditions
"""

import unittest
from dataclasses import dataclass
from typing import Any

from data_juicer.utils.config_utils import ConfigAccessor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@dataclass
class TestConfigObject:
    """Test configuration object for testing."""
    name: str = "test_config"
    value: int = 42
    nested: Any = None
    enabled: bool = True


class ConfigUtilsTest(DataJuicerTestCaseBase):
    """Tests for ConfigAccessor utility class."""

    def test_get_from_dict_existing_key(self):
        """Test getting existing key from dictionary."""
        config = {"name": "test", "value": 123}
        
        result = ConfigAccessor.get(config, "name")
        self.assertEqual(result, "test")
        
        result = ConfigAccessor.get(config, "value")
        self.assertEqual(result, 123)

    def test_get_from_dict_missing_key(self):
        """Test getting missing key from dictionary returns None."""
        config = {"name": "test"}
        
        result = ConfigAccessor.get(config, "missing_key")
        self.assertIsNone(result)

    def test_get_from_dict_with_default(self):
        """Test getting missing key with default value."""
        config = {"name": "test"}
        
        result = ConfigAccessor.get(config, "missing_key", "default_value")
        self.assertEqual(result, "default_value")
        
        result = ConfigAccessor.get(config, "name", "default_value")
        self.assertEqual(result, "test")  # Should return actual value, not default

    def test_get_from_object_existing_attribute(self):
        """Test getting existing attribute from object."""
        config = TestConfigObject(name="my_config", value=999)
        
        result = ConfigAccessor.get(config, "name")
        self.assertEqual(result, "my_config")
        
        result = ConfigAccessor.get(config, "value")
        self.assertEqual(result, 999)
        
        result = ConfigAccessor.get(config, "enabled")
        self.assertTrue(result)

    def test_get_from_object_missing_attribute(self):
        """Test getting missing attribute from object returns None."""
        config = TestConfigObject()
        
        result = ConfigAccessor.get(config, "missing_attr")
        self.assertIsNone(result)

    def test_get_from_object_with_default(self):
        """Test getting missing attribute with default value."""
        config = TestConfigObject()
        
        result = ConfigAccessor.get(config, "missing_attr", "fallback")
        self.assertEqual(result, "fallback")
        
        result = ConfigAccessor.get(config, "name", "fallback")
        self.assertEqual(result, "test_config")  # Should return actual value

    def test_get_none_config(self):
        """Test getting from None config returns default."""
        result = ConfigAccessor.get(None, "any_key")
        self.assertIsNone(result)
        
        result = ConfigAccessor.get(None, "any_key", "default")
        self.assertEqual(result, "default")

    def test_get_empty_dict(self):
        """Test getting from empty dictionary."""
        config = {}
        
        result = ConfigAccessor.get(config, "any_key")
        self.assertIsNone(result)
        
        result = ConfigAccessor.get(config, "any_key", "default")
        self.assertEqual(result, "default")

    def test_get_empty_object(self):
        """Test getting from object with no matching attributes."""
        @dataclass
        class EmptyObject:
            pass
        
        config = EmptyObject()
        
        result = ConfigAccessor.get(config, "any_attr")
        self.assertIsNone(result)
        
        result = ConfigAccessor.get(config, "any_attr", "default")
        self.assertEqual(result, "default")

    def test_get_nested_simple_path(self):
        """Test getting nested value with simple path."""
        # Dict nested structure
        config = {
            "level1": {
                "level2": {
                    "value": "nested_value"
                }
            }
        }
        
        result = ConfigAccessor.get_nested(config, "level1", "level2", "value")
        self.assertEqual(result, "nested_value")

    def test_get_nested_object_path(self):
        """Test getting nested value from object structure."""
        level2_obj = TestConfigObject(value=777)
        level1_obj = TestConfigObject(nested=level2_obj)
        config = level1_obj
        
        result = ConfigAccessor.get_nested(config, "nested", "value")
        self.assertEqual(result, 777)

    def test_get_nested_mixed_dict_object(self):
        """Test getting nested value from mixed dict/object structure."""
        level2_obj = TestConfigObject(value="mixed_value")
        config = {"level1": {"nested_obj": level2_obj}}
        
        result = ConfigAccessor.get_nested(config, "level1", "nested_obj", "value")
        self.assertEqual(result, "mixed_value")

    def test_get_nested_missing_intermediate_key(self):
        """Test nested access with missing intermediate key returns default."""
        config = {"level1": {"value": "present"}}
        
        result = ConfigAccessor.get_nested(config, "level1", "missing", "value")
        self.assertIsNone(result)
        
        result = ConfigAccessor.get_nested(config, "level1", "missing", "value", default="fallback")
        self.assertEqual(result, "fallback")

    def test_get_nested_none_intermediate(self):
        """Test nested access stops at None intermediate value."""
        config = {"level1": None}
        
        result = ConfigAccessor.get_nested(config, "level1", "any_key")
        self.assertIsNone(result)
        
        result = ConfigAccessor.get_nested(config, "level1", "any_key", default="fallback")
        self.assertEqual(result, "fallback")

    def test_get_nested_empty_path(self):
        """Test nested access with no keys returns the config itself."""
        config = {"some": "value"}
        
        result = ConfigAccessor.get_nested(config)
        self.assertEqual(result, config)
        
        result = ConfigAccessor.get_nested(config, default="fallback")
        self.assertEqual(result, config)  # Should return config, not default

    def test_get_nested_single_key(self):
        """Test nested access with single key behaves like regular get."""
        config = {"key": "value"}
        
        result = ConfigAccessor.get_nested(config, "key")
        self.assertEqual(result, "value")
        
        result = ConfigAccessor.get_nested(config, "missing", default="default")
        self.assertEqual(result, "default")

    def test_get_nested_deep_structure(self):
        """Test deeply nested structure access."""
        config = {
            "a": {
                "b": {
                    "c": {
                        "d": {
                            "value": "deep_value"
                        }
                    }
                }
            }
        }
        
        result = ConfigAccessor.get_nested(config, "a", "b", "c", "d", "value")
        self.assertEqual(result, "deep_value")

    def test_get_nested_with_none_config(self):
        """Test nested access with None config returns default."""
        result = ConfigAccessor.get_nested(None, "any", "path")
        self.assertIsNone(result)
        
        result = ConfigAccessor.get_nested(None, "any", "path", default="fallback")
        self.assertEqual(result, "fallback")

    def test_get_type_preservation(self):
        """Test that original types are preserved."""
        config = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"}
        }
        
        self.assertEqual(ConfigAccessor.get(config, "string_val"), "hello")
        self.assertEqual(ConfigAccessor.get(config, "int_val"), 42)
        self.assertEqual(ConfigAccessor.get(config, "float_val"), 3.14)
        self.assertEqual(ConfigAccessor.get(config, "bool_val"), True)
        self.assertEqual(ConfigAccessor.get(config, "list_val"), [1, 2, 3])
        self.assertEqual(ConfigAccessor.get(config, "dict_val"), {"nested": "value"})

    def test_get_nested_type_preservation(self):
        """Test that nested access preserves original types."""
        config = {
            "nested": {
                "data": [1, 2, 3],
                "settings": {"debug": True, "level": 5}
            }
        }
        
        result = ConfigAccessor.get_nested(config, "nested", "data")
        self.assertEqual(result, [1, 2, 3])
        
        result = ConfigAccessor.get_nested(config, "nested", "settings")
        self.assertEqual(result, {"debug": True, "level": 5})


if __name__ == '__main__':
    unittest.main()