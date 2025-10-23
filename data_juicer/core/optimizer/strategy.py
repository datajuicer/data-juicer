from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from loguru import logger

from data_juicer.core.pipeline_ast import OpNode, PipelineAST


class OptimizationStrategy(ABC):
    """Base class for pipeline optimization strategies."""

    def __init__(self, name: str):
        """Initialize the optimization strategy.

        Args:
            name: Name of the optimization strategy
        """
        self.name = name

    @abstractmethod
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply the optimization strategy to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST
        """
        pass

    def _get_operation_chain(self, node: OpNode) -> List[OpNode]:
        """Get the linear chain of operations from a node.

        Args:
            node: The node to start from

        Returns:
            List of operations in the chain
        """
        chain = []
        current = node
        while current.children:
            current = current.children[0]
            chain.append(current)
        return chain

    def _rebuild_chain(self, root: OpNode, chain: List[OpNode]) -> None:
        """Rebuild the operation chain from a list of nodes.

        Args:
            root: The root node
            chain: List of operations to chain
        """
        current = root
        for node in chain:
            current.children = [node]
            node.parent = current
            current = node


class StrategyRegistry:
    """Registry for optimization strategies."""

    _strategies: Dict[str, Type[OptimizationStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[OptimizationStrategy]) -> None:
        """Register a strategy class with a name.

        Args:
            name: Name to register the strategy under
            strategy_class: The strategy class to register
        """
        cls._strategies[name] = strategy_class
        logger.debug(f"🔧 Registered optimization strategy: {name} -> {strategy_class.__name__}")

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[OptimizationStrategy]]:
        """Get a strategy class by name.

        Args:
            name: Name of the strategy

        Returns:
            The strategy class if found, None otherwise
        """
        return cls._strategies.get(name)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names.

        Returns:
            List of registered strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def create_strategy(cls, name: str, **kwargs) -> Optional[OptimizationStrategy]:
        """Create a strategy instance by name.

        Args:
            name: Name of the strategy
            **kwargs: Additional arguments to pass to strategy constructor

        Returns:
            Strategy instance if found and created successfully, None otherwise
        """
        strategy_class = cls.get_strategy_class(name)
        if strategy_class is None:
            logger.warning(f"⚠️ Unknown strategy '{name}'. Available strategies: {cls.get_available_strategies()}")
            return None

        try:
            return strategy_class(**kwargs)
        except Exception as e:
            logger.error(f"❌ Failed to create strategy '{name}': {e}")
            return None


def register_strategy(name: str):
    """Decorator to register a strategy class.

    Args:
        name: Name to register the strategy under

    Example:
        @register_strategy('op_reorder')
        class OpReorderStrategy(OptimizationStrategy):
            pass
    """

    def decorator(strategy_class: Type[OptimizationStrategy]) -> Type[OptimizationStrategy]:
        StrategyRegistry.register(name, strategy_class)
        return strategy_class

    return decorator
