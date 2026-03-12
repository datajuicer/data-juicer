#!/usr/bin/env python3
"""
Optimization Manager for Data-Juicer Pipeline Optimization.

This module provides a centralized way to apply optimization strategies
to data processing pipelines across different executors.

Features:
- Strategy-based optimization (op_reorder, filter_fusion, etc.)
- Optional operation probing for empirical cost estimation
- Configurable via enable_optimizer and related options
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.optimizer.op_prober import OpProber, ProbeResults
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
        self.probe_results: Optional[ProbeResults] = None
        self._check_optimization_enabled()
        self._check_probing_config()

    def _check_optimization_enabled(self):
        """Check if optimization is enabled via config.

        Supports both new `enable_optimizer` config and legacy `op_fusion` config.
        Legacy `op_fusion` is automatically mapped to appropriate optimizer strategies:
        - op_fusion with fusion_strategy="greedy" -> filter_fusion strategy
        - op_fusion with fusion_strategy="probe" -> op_reorder + filter_fusion strategies
        """
        # Check for new-style enable_optimizer config
        if self.cfg and hasattr(self.cfg, "enable_optimizer") and self.cfg.enable_optimizer:
            self.optimization_enabled = True
            # Get strategies from config
            if hasattr(self.cfg, "optimizer_strategies"):
                self.optimization_strategies = self.cfg.optimizer_strategies
            else:
                self.optimization_strategies = ["op_reorder"]  # Default strategy

            # Ensure strategies is a list
            if isinstance(self.optimization_strategies, str):
                self.optimization_strategies = self.optimization_strategies.split(",")

            logger.info(f"Pipeline optimizer enabled with strategies: {self.optimization_strategies}")

        # Check for legacy op_fusion config (backward compatibility)
        elif self.cfg and hasattr(self.cfg, "op_fusion") and self.cfg.op_fusion:
            self.optimization_enabled = True
            fusion_strategy = getattr(self.cfg, "fusion_strategy", "greedy")

            # Map legacy fusion_strategy to optimizer strategies
            if fusion_strategy == "probe":
                # Probe strategy = reorder by speed + filter fusion
                self.optimization_strategies = ["op_reorder", "filter_fusion"]
                logger.info("Legacy op_fusion (probe) mapped to: op_reorder, filter_fusion")
            else:
                # Greedy strategy = filter fusion only (preserves original order)
                self.optimization_strategies = ["filter_fusion"]
                logger.info("Legacy op_fusion (greedy) mapped to: filter_fusion")

        else:
            self.optimization_enabled = False
            self.optimization_strategies = []

    def _check_probing_config(self):
        """Check probing configuration options.

        Probing runs each operation on a small sample to measure actual costs.
        This enables more accurate reordering decisions.

        Config options:
        - optimizer_probe_enabled: Whether to enable probing (default: True)
        - optimizer_probe_samples: Number of samples to probe (default: 100)
        """
        if not self.optimization_enabled:
            self.probing_enabled = False
            self.probe_samples = 0
            return

        # Check if probing is enabled (default: True when optimizer is enabled)
        self.probing_enabled = getattr(self.cfg, "optimizer_probe_enabled", True)
        self.probe_samples = getattr(self.cfg, "optimizer_probe_samples", 100)

        if self.probing_enabled:
            logger.info(f"Operation probing enabled with {self.probe_samples} samples")

    def probe_operations(self, ops: List[Any], dataset: Any) -> ProbeResults:
        """
        Probe operations to estimate their costs.

        This runs each operation on a small sample of the dataset and
        measures execution time and selectivity.

        Args:
            ops: List of operations to probe
            dataset: Dataset to sample from

        Returns:
            ProbeResults containing cost estimates
        """
        if not self.probing_enabled:
            return ProbeResults()

        prober = OpProber(sample_size=self.probe_samples)
        return prober.probe_operations(ops, dataset)

    def apply_optimizations(self, ops: List[Any], dataset: Any = None) -> List[Any]:
        """
        Apply optimization strategies to a list of operations.

        Args:
            ops: List of operations to optimize
            dataset: Optional dataset for probing operation costs

        Returns:
            Optimized list of operations
        """
        if not self.optimization_enabled:
            return ops

        try:
            # Store dataset for use by strategies (e.g., filter_fusion with probing)
            self._current_dataset = dataset

            # Probe operations if enabled and dataset is provided
            if self.probing_enabled and dataset is not None:
                self.probe_results = self.probe_operations(ops, dataset)
            else:
                self.probe_results = None

            # Create AST from operations
            ast = self._create_ast_from_ops(ops)

            # Get original operation order for comparison
            original_order = [getattr(op, "_name", getattr(op, "name", str(op))) for op in ops]

            # Apply core optimizer with properly initialized strategies
            strategy_objects = self._initialize_strategies()
            optimizer = PipelineOptimizer(strategy_objects)
            optimized_ast = optimizer.optimize(ast)

            # Extract optimized operations from the AST
            optimized_ops = self._extract_ops_from_ast(optimized_ast, ops)

            # Log before/after comparison
            optimized_order = [getattr(op, "_name", getattr(op, "name", str(op))) for op in optimized_ops]
            self._log_optimization_result(original_order, optimized_order)

            # If probing was enabled, reload fresh operations to avoid state issues
            # The probed operations may have modified internal state that causes issues
            if self.probing_enabled and dataset is not None:
                optimized_ops = self._reload_fresh_ops(optimized_ops)

            return optimized_ops

        except Exception as e:
            logger.error(f"Optimizer failed: {e}")
            logger.warning("Continuing with original operation order")
            return ops

    def _reload_fresh_ops(self, ops: List[Any]) -> List[Any]:
        """Reload fresh operation instances to avoid state issues from probing.

        After probing, operations may have internal state that conflicts with
        running on the full dataset. This method creates fresh instances.
        """
        try:
            from data_juicer.ops import load_ops

            # Build op configs from the existing ops in their optimized order
            op_configs = []
            for op in ops:
                op_name = getattr(op, "_name", getattr(op, "name", str(op)))

                # Get the original config from the op
                if hasattr(op, "_init_configs"):
                    # Use the initialization configs if available
                    op_config = {op_name: op._init_configs}
                else:
                    # Build config from op attributes
                    op_config = {op_name: {}}
                    if hasattr(op, "__dict__"):
                        for k, v in op.__dict__.items():
                            if not k.startswith("_") and not callable(v):
                                try:
                                    # Only include serializable values
                                    import json

                                    json.dumps(v)
                                    op_config[op_name][k] = v
                                except (TypeError, ValueError):
                                    pass

                op_configs.append(op_config)

            # Reload fresh operations
            fresh_ops = load_ops(op_configs)
            logger.debug(f"Reloaded {len(fresh_ops)} fresh operations after probing")
            return fresh_ops

        except Exception as e:
            logger.warning(f"Failed to reload fresh ops: {e}. Using probed ops.")
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
        """Extract optimized operations from the AST.

        This method handles both regular operations and fused operations:
        - Regular ops are looked up in the original ops map
        - Fused ops (fused_filter, fused_mapper) are constructed from their configs
        """
        try:
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

            # Extract operations from AST, handling fused operations
            optimized_ops = []
            used_ops = set()  # Track which original ops have been used

            self._extract_ops_recursive(ast.root, op_map, optimized_ops, used_ops)

            return optimized_ops

        except Exception as e:
            logger.error(f"Failed to extract optimized operations: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            return original_ops

    def _extract_ops_recursive(self, node: OpNode, op_map: Dict[str, Any], result: List[Any], used_ops: set):
        """Recursively extract operations from AST nodes."""
        if not node:
            return

        # Skip root node
        if node.name != "root":
            if node.name == "fused_filter":
                # Handle fused filter - create FusedFilter from component filters
                fused_op = self._create_fused_filter(node, op_map, used_ops)
                if fused_op:
                    result.append(fused_op)
            elif node.name == "fused_mapper":
                # Handle fused mapper - create FusedMapper from component mappers
                fused_op = self._create_fused_mapper(node, op_map, used_ops)
                if fused_op:
                    result.append(fused_op)
            elif node.name in op_map:
                # Regular operation - look up in map
                result.append(op_map[node.name])
                used_ops.add(node.name)
            else:
                logger.warning(f"Operation '{node.name}' not found in original operations")

        # Process children
        if node.children:
            for child in node.children:
                self._extract_ops_recursive(child, op_map, result, used_ops)

    def _create_fused_filter(self, node: OpNode, op_map: Dict[str, Any], used_ops: set) -> Any:
        """Create a FusedFilter from an AST node."""
        try:
            from data_juicer.core.optimizer.fused_op import FusedFilter

            # Get the fused op config
            fused_config = node.config.get("general_fused_op", {})
            fused_op_list = fused_config.get("fused_op_list", [])
            detailed_ops = fused_config.get("detailed_ops", [])

            if not fused_op_list:
                logger.warning("fused_filter has empty fused_op_list")
                return None

            # Collect the filter objects from original ops
            filters = []
            for op_config in fused_op_list:
                # op_config is like {"text_length_filter": {...}}
                op_name = list(op_config.keys())[0]
                if op_name in op_map:
                    filters.append(op_map[op_name])
                    used_ops.add(op_name)
                else:
                    logger.warning(f"Filter '{op_name}' not found for fusion")

            if not filters:
                logger.warning("No filters found for fused_filter")
                return None

            # Create the FusedFilter
            fused_filter = FusedFilter(
                name="fused_filter",
                fused_filters=filters,
            )

            logger.info(f"Fused {len(filters)} filters: {detailed_ops}")
            return fused_filter

        except Exception as e:
            logger.error(f"Failed to create FusedFilter: {e}")
            return None

    def _create_fused_mapper(self, node: OpNode, op_map: Dict[str, Any], used_ops: set) -> Any:
        """Create a FusedMapper from an AST node."""
        try:
            from data_juicer.core.optimizer.fused_op import FusedMapper

            # Get the fused op config
            fused_config = node.config.get("general_fused_op", {})
            fused_op_list = fused_config.get("fused_op_list", [])
            detailed_ops = fused_config.get("detailed_ops", [])

            if not fused_op_list:
                logger.warning("fused_mapper has empty fused_op_list")
                return None

            # Collect the mapper objects from original ops
            mappers = []
            for op_config in fused_op_list:
                op_name = list(op_config.keys())[0]
                if op_name in op_map:
                    mappers.append(op_map[op_name])
                    used_ops.add(op_name)
                else:
                    logger.warning(f"Mapper '{op_name}' not found for fusion")

            if not mappers:
                logger.warning("No mappers found for fused_mapper")
                return None

            # Create the FusedMapper
            fused_mapper = FusedMapper(
                name="fused_mapper",
                fused_mappers=mappers,
            )

            logger.info(f"Fused {len(mappers)} mappers: {detailed_ops}")
            return fused_mapper

        except Exception as e:
            logger.error(f"Failed to create FusedMapper: {e}")
            return None

    def _get_operation_order_from_ast(self, ast: PipelineAST) -> List[str]:
        """Get the operation order from the AST using depth-first traversal."""
        order = []
        self._traverse_ast_dfs(ast.root, order)
        return order

    def _traverse_ast_dfs(self, node: OpNode, order: List[str]):
        """Depth-first traversal of AST nodes."""
        if not node:
            return

        # Skip root node but process its children
        if node.name != "root":
            order.append(node.name)

        # Recursively traverse all children
        if node.children:
            for child in node.children:
                self._traverse_ast_dfs(child, order)

    def _log_optimization_result(self, original_order: List[str], optimized_order: List[str]):
        """Log the before/after optimization comparison."""
        if original_order == optimized_order:
            logger.info("Pipeline optimization: no changes applied")
            return

        # Log summary
        logger.info(f"Pipeline optimization: {len(original_order)} ops -> {len(optimized_order)} ops")

        # Log before/after
        logger.info("  BEFORE: " + " -> ".join(original_order))
        logger.info("  AFTER:  " + " -> ".join(optimized_order))

        # Identify specific changes
        if len(original_order) != len(optimized_order):
            # Fusion occurred
            fused_count = len(original_order) - len(optimized_order)
            logger.info(f"  FUSION: {fused_count} operations fused")
        else:
            # Reordering occurred - show what moved
            changes = []
            for i, (orig, opt) in enumerate(zip(original_order, optimized_order)):
                if orig != opt:
                    changes.append(f"pos {i+1}: {orig} -> {opt}")
            if changes:
                logger.info(f"  REORDER: {len(changes)} positions changed")
                for change in changes[:5]:  # Show first 5 changes
                    logger.info(f"    {change}")
                if len(changes) > 5:
                    logger.info(f"    ... and {len(changes) - 5} more")

    def is_optimization_enabled(self) -> bool:
        """Check if optimization is enabled."""
        return self.optimization_enabled

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled optimization strategies."""
        return self.optimization_strategies

    def _initialize_strategies(self) -> List[Any]:
        """Initialize strategy objects from strategy names using the registry.

        Passes probe results to strategies that support them (e.g., op_reorder).
        """
        strategy_objects = []

        for strategy_name in self.optimization_strategies:
            strategy_name = strategy_name.strip()  # Remove any whitespace

            # Create strategy with probe results if supported
            strategy_obj = self._create_strategy_with_probe_results(strategy_name)

            if strategy_obj is not None:
                strategy_objects.append(strategy_obj)
                logger.debug(f"Initialized strategy: {strategy_name}")
            else:
                logger.warning(f"Failed to initialize strategy: {strategy_name}")

        if not strategy_objects:
            logger.warning("No valid strategies initialized, using default op_reorder strategy")
            default_strategy = self._create_strategy_with_probe_results("op_reorder")
            if default_strategy is not None:
                strategy_objects = [default_strategy]
            else:
                logger.error("Failed to create default strategy")
                strategy_objects = []

        return strategy_objects

    def _create_strategy_with_probe_results(self, strategy_name: str) -> Any:
        """Create a strategy instance, passing probe results if supported.

        Args:
            strategy_name: Name of the strategy to create

        Returns:
            Strategy instance or None if creation fails
        """
        # Strategies that support probe results for reordering
        reorder_probe_strategies = {"op_reorder"}
        # Strategies that support probe-based fusion validation
        fusion_probe_strategies = {"filter_fusion"}

        if strategy_name in reorder_probe_strategies and self.probe_results:
            # Import and create strategy directly with probe results
            try:
                from data_juicer.core.optimizer.op_reorder_strategy import (
                    OpReorderStrategy,
                )

                if strategy_name == "op_reorder":
                    return OpReorderStrategy(probe_results=self.probe_results)
            except Exception as e:
                logger.warning(f"Failed to create {strategy_name} with probe results: {e}")

        if strategy_name in fusion_probe_strategies and self.probing_enabled:
            # Create filter fusion with probe-based validation
            try:
                from data_juicer.core.optimizer.filter_fusion_strategy import (
                    FilterFusionStrategy,
                )
                from data_juicer.core.optimizer.op_prober import OpProber

                prober = OpProber(sample_size=self.probe_samples)
                return FilterFusionStrategy(
                    prober=prober,
                    dataset=self._current_dataset,
                    probe_fusion=True,
                    fusion_speedup_threshold=1.1,
                )
            except Exception as e:
                logger.warning(f"Failed to create {strategy_name} with probing: {e}")

        # Fall back to registry creation
        return StrategyRegistry.create_strategy(strategy_name)


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


def apply_optimizations(ops: List[Any], cfg=None, dataset: Any = None) -> List[Any]:
    """
    Convenience function to apply optimizations to operations.

    Args:
        ops: List of operations to optimize
        cfg: Configuration object (optional)
        dataset: Dataset for probing operation costs (optional).
                 If provided and probing is enabled, operations will be
                 run on a small sample to measure actual costs.

    Returns:
        Optimized list of operations
    """
    # Always create a new manager with the provided config
    # to ensure enable_optimizer setting is respected
    manager = OptimizationManager(cfg)
    return manager.apply_optimizations(ops, dataset=dataset)
