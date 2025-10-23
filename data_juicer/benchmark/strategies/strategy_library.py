#!/usr/bin/env python3
"""
Library of optimization strategies for A/B testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class StrategyType(Enum):
    """Types of optimization strategies."""

    FUSION = "fusion"
    BATCHING = "batching"
    MEMORY = "memory"
    PARALLEL = "parallel"
    CACHING = "caching"
    ALGORITHM = "algorithm"


@dataclass
class StrategyConfig:
    """Configuration for an optimization strategy."""

    name: str
    enabled: bool
    parameters: Dict[str, Any]
    description: str = ""
    strategy_type: StrategyType = StrategyType.ALGORITHM


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.enabled = False
        self.parameters = {}

    @abstractmethod
    def apply_to_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this strategy to a configuration."""
        pass

    @abstractmethod
    def get_expected_impact(self) -> Dict[str, str]:
        """Get expected impact description."""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate strategy parameters."""
        return []

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for this strategy."""
        return {}


class StrategyLibrary:
    """Library of available optimization strategies."""

    def __init__(self):
        self.strategies = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize all available strategies."""
        from .config_strategies import BaselineStrategy, CoreOptimizerStrategy

        # Baseline Strategy (no optimizations)
        self.strategies["baseline"] = BaselineStrategy()

        # Core Optimizer Strategies - these configure the actual optimizer
        self.strategies["mapper_fusion"] = CoreOptimizerStrategy(
            "mapper_fusion", "Enable mapper operation fusion", ["mapper_fusion"]
        )
        self.strategies["filter_fusion"] = CoreOptimizerStrategy(
            "filter_fusion", "Enable filter operation fusion", ["filter_fusion"]
        )
        self.strategies["full_optimization"] = CoreOptimizerStrategy(
            "full_optimization", "Enable all core optimizations", ["mapper_fusion", "filter_fusion"]
        )

        # Additional configuration-based strategies (not core optimizer)
        class SimpleStrategy(OptimizationStrategy):
            def __init__(self, name, description, strategy_type, apply_func, impact_dict):
                super().__init__(name, description)
                self.strategy_type = strategy_type
                self._apply_func = apply_func
                self._impact_dict = impact_dict

            def apply_to_config(self, config):
                return self._apply_func(config)

            def get_expected_impact(self):
                return self._impact_dict

        # Configuration-only strategies (not core optimizer)
        self.strategies["large_batch_size"] = SimpleStrategy(
            "large_batch_size",
            "Use large batch sizes for better throughput",
            StrategyType.BATCHING,
            lambda config: {**config, "batch_size": 1000},
            {
                "performance": "Improved throughput with large batches",
                "memory": "Higher memory usage",
                "complexity": "Minimal configuration complexity",
            },
        )

        self.strategies["streaming_processing"] = SimpleStrategy(
            "streaming_processing",
            "Enable streaming processing to reduce memory usage",
            StrategyType.MEMORY,
            lambda config: {**config, "streaming": True},
            {
                "performance": "May reduce throughput for memory savings",
                "memory": "Significantly reduced memory usage",
                "complexity": "Minimal configuration complexity",
            },
        )

        self.strategies["ray_optimized"] = SimpleStrategy(
            "ray_optimized",
            "Optimize for Ray distributed processing",
            StrategyType.PARALLEL,
            lambda config: {**config, "executor": "ray", "ray_config": {"num_cpus": -1}},
            {
                "performance": "Improved throughput through distributed processing",
                "memory": "Distributed memory usage across nodes",
                "complexity": "Moderate configuration complexity",
            },
        )

        # Operation Reordering Strategy (from core optimizer)
        self.strategies["op_reorder"] = CoreOptimizerStrategy(
            "op_reorder", "Enable operation reordering for optimal execution order", ["op_reorder"]
        )

    def get_strategy(self, name: str) -> Optional[OptimizationStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(name)

    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[OptimizationStrategy]:
        """Get all strategies of a specific type."""
        return [s for s in self.strategies.values() if s.strategy_type == strategy_type]

    def get_all_strategies(self) -> List[OptimizationStrategy]:
        """Get all available strategies."""
        return list(self.strategies.values())

    def create_strategy_config(
        self, name: str, enabled: bool = True, parameters: Dict[str, Any] = None
    ) -> StrategyConfig:
        """Create a strategy configuration."""
        strategy = self.get_strategy(name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {name}")

        return StrategyConfig(
            name=name,
            enabled=enabled,
            parameters=parameters or {},
            description=strategy.description,
            strategy_type=strategy.strategy_type,
        )


# Global strategy library instance
STRATEGY_LIBRARY = StrategyLibrary()
