"""
Configuration utilities for handling both dict and object-style configs.
"""

from typing import Any


class ConfigAccessor:
    """Utility for accessing configuration values that may be dicts or objects."""

    @staticmethod
    def get(config: Any, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from either a dict or object.

        Args:
            config: Configuration object (dict or object with attributes)
            key: Key/attribute name to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def get_nested(config: Any, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value.

        Example:
            get_nested(cfg, 'partition', 'mode', default='auto')

        Args:
            config: Configuration object
            keys: Series of keys to traverse
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        current = config
        for key in keys:
            if current is None:
                return default
            current = ConfigAccessor.get(current, key)
        return current if current is not None else default
