#!/usr/bin/env python3
"""
Optimization Manager for Data-Juicer Pipeline Optimization.

This module provides a centralized way to apply optimization strategies
to data processing pipelines across different executors.
"""

from typing import Any, List

from loguru import logger

from data_juicer.core.optimizer.optimizer import PipelineOptimizer
from data_juicer.core.optimizer.strategy import StrategyRegistry
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST


class OptimizationManager:
    """
    Centralized manager for applying optimization strategies to data processing pipelines.

    This class provides a clean interface for executors to apply optimization
    without duplicating logic across different executor implementations.
    """

    def __init__(self, cfg=None):
        """Initialize the optimization manager."""
        self.cfg = cfg
        self._check_optimization_enabled()

    def _check_optimization_enabled(self):
        """Check if optimization is enabled via config."""
        # Check config for optimization settings
        if self.cfg and hasattr(self.cfg, "enable_optimizer"):
            self.optimization_enabled = self.cfg.enable_optimizer
        else:
            self.optimization_enabled = False

        if self.optimization_enabled:
            # Get strategies from config
            if self.cfg and hasattr(self.cfg, "optimizer_strategies"):
                self.optimization_strategies = self.cfg.optimizer_strategies
            else:
                self.optimization_strategies = ["op_reorder"]  # Default strategy

            # Ensure strategies is a list
            if isinstance(self.optimization_strategies, str):
                self.optimization_strategies = self.optimization_strategies.split(",")

            logger.info(f"🔧 Core optimizer enabled with strategies: {self.optimization_strategies}")
        else:
            self.optimization_strategies = []

    def apply_optimizations(self, ops: List[Any]) -> List[Any]:
        """
        Apply optimization strategies to a list of operations.

        Args:
            ops: List of operations to optimize

        Returns:
            Optimized list of operations
        """
        if not self.optimization_enabled:
            return ops

        try:
            logger.info("🔧 Applying core optimizer to operations...")

            # Create AST from operations
            ast = self._create_ast_from_ops(ops)

            # Print original operation order
            original_order = [getattr(op, "_name", getattr(op, "name", str(op))) for op in ops]
            logger.info("📋 Original operation order:")
            for i, op_name in enumerate(original_order, 1):
                logger.info(f"   {i}. {op_name}")

            # Print original AST structure
            logger.info("📋 Original AST structure:")
            self._print_ast_structure(ast, "BEFORE")

            # Apply core optimizer with properly initialized strategies
            strategy_objects = self._initialize_strategies()
            optimizer = PipelineOptimizer(strategy_objects)
            optimized_ast = optimizer.optimize(ast)

            # Print optimized AST structure
            logger.info("📋 Optimized AST structure:")
            self._print_ast_structure(optimized_ast, "AFTER")

            # Extract optimized operations from the AST
            optimized_ops = self._extract_ops_from_ast(optimized_ast, ops)

            # Print final optimized operation order
            optimized_order = [getattr(op, "_name", getattr(op, "name", str(op))) for op in optimized_ops]
            logger.info("📋 Final optimized operation order:")
            for i, op_name in enumerate(optimized_order, 1):
                logger.info(f"   {i}. {op_name}")

            # Show the difference
            if original_order != optimized_order:
                logger.info("🔄 Operation order has been optimized!")
                logger.info("📊 Changes:")
                for i, (orig, opt) in enumerate(zip(original_order, optimized_order)):
                    if orig != opt:
                        logger.info(f"   Position {i+1}: {orig} → {opt}")
            else:
                logger.info("ℹ️ No changes to operation order")

            logger.info("✅ Core optimizer applied successfully")
            return optimized_ops

        except Exception as e:
            logger.error(f"❌ Failed to apply core optimizer: {e}")
            logger.warning("⚠️ Continuing with original operation order")
            return ops

    def _create_ast_from_ops(self, ops: List[Any]) -> PipelineAST:
        """Create a PipelineAST from operations."""
        ast = PipelineAST()
        ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        current_node = ast.root
        for op in ops:
            # Determine operation type and name
            if hasattr(op, "_name"):
                op_name = op._name
            elif hasattr(op, "name"):
                op_name = op.name
            else:
                op_name = str(op)

            # Determine operation type based on name
            if "filter" in op_name.lower():
                op_type = OpType.FILTER
            elif "mapper" in op_name.lower():
                op_type = OpType.MAPPER
            else:
                op_type = OpType.MAPPER  # Default to mapper

            # Get operation config
            op_config = {}
            if hasattr(op, "config"):
                op_config = op.config
            elif hasattr(op, "__dict__"):
                op_config = {k: v for k, v in op.__dict__.items() if not k.startswith("_")}

            # Create operation node
            op_node = OpNode(name=op_name, op_type=op_type, config=op_config)

            # Add to AST
            current_node.children = [op_node]
            op_node.parent = current_node
            current_node = op_node

        return ast

    def _extract_ops_from_ast(self, ast: PipelineAST, original_ops: List[Any]) -> List[Any]:
        """Extract optimized operations from the AST."""
        try:
            logger.info(f"🔍 Extracting operations from AST with {len(original_ops)} original operations")

            # Get the optimized operation order from the AST
            optimized_order = self._get_operation_order_from_ast(ast)

            # Create a mapping from operation names to original operation objects
            op_map = {}
            for op in original_ops:
                if hasattr(op, "_name"):
                    op_name = op._name
                elif hasattr(op, "name"):
                    op_name = op.name
                else:
                    op_name = str(op)
                op_map[op_name] = op

            logger.info(f"🔍 Created operation map with {len(op_map)} operations")

            # Reorder operations according to the optimized AST
            optimized_ops = []
            for op_name in optimized_order:
                if op_name in op_map:
                    optimized_ops.append(op_map[op_name])
                else:
                    logger.warning(f"⚠️ Could not find operation '{op_name}' in original operations")

            # Add any operations that weren't in the AST (shouldn't happen, but safety check)
            for op in original_ops:
                op_name = getattr(op, "_name", getattr(op, "name", str(op)))
                if op_name not in optimized_order:
                    logger.warning(f"⚠️ Operation '{op_name}' not found in optimized order, adding at end")
                    optimized_ops.append(op)

            logger.info(f"📋 Reordered {len(optimized_ops)} operations based on optimization")
            return optimized_ops

        except Exception as e:
            logger.error(f"❌ Failed to extract optimized operations: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            logger.warning("⚠️ Returning original operations")
            return original_ops

    def _get_operation_order_from_ast(self, ast: PipelineAST) -> List[str]:
        """Get the operation order from the AST using depth-first traversal."""
        order = []

        logger.info(f"🔍 Starting AST traversal from root: {ast.root}")

        # Use depth-first traversal to get all operations
        self._traverse_ast_dfs(ast.root, order)

        logger.info(f"🔍 Extracted operation order from AST: {order}")
        return order

    def _traverse_ast_dfs(self, node: OpNode, order: List[str]):
        """Depth-first traversal of AST nodes."""
        if not node:
            return

        # Skip root node but process its children
        if node.name != "root":
            order.append(node.name)
            logger.info(f"🔍 Added to order: {node.name}")

        # Recursively traverse all children
        if node.children:
            for i, child in enumerate(node.children):
                logger.info(f"🔍 Processing child {i+1}/{len(node.children)}: {child.name}")
                self._traverse_ast_dfs(child, order)

    def is_optimization_enabled(self) -> bool:
        """Check if optimization is enabled."""
        return self.optimization_enabled

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled optimization strategies."""
        return self.optimization_strategies

    def _initialize_strategies(self) -> List[Any]:
        """Initialize strategy objects from strategy names using the registry."""
        strategy_objects = []

        for strategy_name in self.optimization_strategies:
            strategy_name = strategy_name.strip()  # Remove any whitespace

            # Use the registry to create strategy instances
            strategy_obj = StrategyRegistry.create_strategy(strategy_name)

            if strategy_obj is not None:
                strategy_objects.append(strategy_obj)
                logger.info(f"🔧 Initialized strategy: {strategy_name}")
            else:
                logger.warning(f"⚠️ Failed to initialize strategy: {strategy_name}")

        if not strategy_objects:
            logger.warning("⚠️ No valid strategies initialized, using default op_reorder strategy")
            default_strategy = StrategyRegistry.create_strategy("op_reorder")
            if default_strategy is not None:
                strategy_objects = [default_strategy]
            else:
                logger.error("❌ Failed to create default strategy")
                strategy_objects = []

        logger.info(f"🔧 Initialized {len(strategy_objects)} optimization strategies")
        return strategy_objects

    def _print_ast_structure(self, ast: PipelineAST, phase: str):
        """Print the AST structure for debugging purposes."""
        logger.info(f"🔍 {phase} optimization - AST structure:")

        if not ast or not ast.root:
            logger.info("   Empty AST")
            return

        # Print the AST tree structure
        self._print_ast_node(ast.root, 0, phase)

    def _print_ast_node(self, node: OpNode, depth: int, phase: str):
        """Recursively print AST node structure."""
        indent = "  " * depth

        if node.name == "root":
            logger.info(f"{indent}🌳 ROOT")
        else:
            # Get operation type emoji
            type_emoji = "🔧" if node.op_type == OpType.MAPPER else "🔍" if node.op_type == OpType.FILTER else "⚙️"
            logger.info(f"{indent}{type_emoji} {node.op_type.name}: {node.name}")

            # Print key config parameters if available
            if node.config:
                important_configs = {}
                for key, value in node.config.items():
                    if key in [
                        "text_key",
                        "image_key",
                        "audio_key",
                        "video_key",
                        "threshold",
                        "min_length",
                        "max_length",
                    ]:
                        important_configs[key] = value

                if important_configs:
                    config_str = ", ".join([f"{k}={v}" for k, v in important_configs.items()])
                    logger.info(f"{indent}    📝 Config: {config_str}")

        # Print children
        if node.children:
            for child in node.children:
                self._print_ast_node(child, depth + 1, phase)


# Global optimization manager instance
_optimization_manager = None


def get_optimization_manager(cfg=None) -> OptimizationManager:
    """
    Get the global optimization manager instance.

    Args:
        cfg: Configuration object (optional)

    Returns:
        OptimizationManager instance
    """
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = OptimizationManager(cfg)
    return _optimization_manager


def apply_optimizations(ops: List[Any], cfg=None) -> List[Any]:
    """
    Convenience function to apply optimizations to operations.

    Args:
        ops: List of operations to optimize
        cfg: Configuration object (optional)

    Returns:
        Optimized list of operations
    """
    manager = get_optimization_manager(cfg)
    return manager.apply_optimizations(ops)
