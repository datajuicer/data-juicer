#!/usr/bin/env python3
"""
Configuration management for benchmark framework.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger


class ConfigManager:
    """Manages configuration files for benchmark testing."""

    def __init__(self):
        self.config_cache = {}

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)

        # Check cache first
        if str(config_path) in self.config_cache:
            return self.config_cache[str(config_path)].copy()

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Cache the config
            self.config_cache[str(config_path)] = config.copy()

            logger.debug(f"Loaded config from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                if output_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif output_path.suffix.lower() == ".json":
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported output format: {output_path.suffix}")

            logger.debug(f"Saved config to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {output_path}: {e}")
            raise

    def apply_strategy_config(self, base_config: Dict[str, Any], strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strategy-specific configuration to base config."""

        # Deep copy to avoid modifying original
        result_config = self._deep_copy_dict(base_config)

        # Apply strategy modifications
        for key, value in strategy_config.items():
            self._set_nested_value(result_config, key, value)

        logger.debug(f"Applied strategy config: {strategy_config}")
        return result_config

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations."""
        result = {}

        for config in configs:
            result = self._deep_merge_dicts(result, config)

        return result

    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any] = None) -> List[str]:
        """Validate configuration against schema."""
        issues = []

        if schema is None:
            # Basic validation
            required_fields = ["dataset", "process"]
            for field in required_fields:
                if field not in config:
                    issues.append(f"Missing required field: {field}")
        else:
            # Schema-based validation
            issues.extend(self._validate_against_schema(config, schema))

        return issues

    def get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a dictionary."""
        return json.loads(json.dumps(d))

    def _deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set a nested value in configuration."""
        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        issues = []

        # This is a simplified schema validation
        # In practice, you might want to use a more robust schema validation library

        for field, field_schema in schema.items():
            if field_schema.get("required", False) and field not in config:
                issues.append(f"Missing required field: {field}")

            if field in config:
                field_type = field_schema.get("type", "string")
                if field_type == "boolean" and not isinstance(config[field], bool):
                    issues.append(f"Field {field} must be boolean")
                elif field_type == "integer" and not isinstance(config[field], int):
                    issues.append(f"Field {field} must be integer")
                elif field_type == "string" and not isinstance(config[field], str):
                    issues.append(f"Field {field} must be string")

        return issues
