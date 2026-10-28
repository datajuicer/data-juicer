"""
Configuration utilities for handling both dict and object-style configs.
"""

import re
from typing import Any

from jsonargparse import Namespace


def redact_sensitive_config(value):
    """Return a display-only copy with credentials and URI userinfo hidden."""
    if isinstance(value, Namespace):
        value = value.as_dict()
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            tokens = str(key).lower().replace("-", "_").replace(".", "_").split("_")
            sensitive = any(
                token in {"password", "passwd", "secret", "token", "credential", "credentials", "authorization"}
                for token in tokens
            ) or any(
                tokens[index : index + 2] in (["access", "key"], ["account", "key"], ["api", "key"], ["private", "key"])
                for index in range(len(tokens) - 1)
            )
            redacted[key] = "***" if sensitive else redact_sensitive_config(item)
        return redacted
    if isinstance(value, str):
        return re.sub(r"(?<=://)[^/@\s]+@", "***@", value)
    if isinstance(value, list):
        return [redact_sensitive_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_config(item) for item in value)
    return value


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
