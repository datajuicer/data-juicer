from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy
from data_juicer.core.optimizer.mapper_fusion_strategy import MapperFusionStrategy
from data_juicer.core.optimizer.op_reorder_strategy import OpReorderStrategy
from data_juicer.core.optimizer.strategy import OptimizationStrategy
from data_juicer.core.pipeline_ast import PipelineAST


class PipelineOptimizer:
    """Main optimizer class that manages multiple optimization strategies."""

    def __init__(
        self,
        strategies: Optional[List[OptimizationStrategy]] = None,
        analyzer_insights: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the optimizer with a list of strategies.

        Args:
            strategies: List of optimization strategies to apply. If None,
                       default strategies will be used.
            analyzer_insights: Optional dataset analysis insights for optimization
        """
        self.analyzer_insights = analyzer_insights or {}

        if strategies is None:
            # Create strategies with analyzer insights
            self.strategies = [
                OpReorderStrategy(),  # Apply reordering first
                MapperFusionStrategy(),
                FilterFusionStrategy(analyzer_insights=self.analyzer_insights),
            ]
        else:
            self.strategies = strategies

    def add_strategy(self, strategy: OptimizationStrategy) -> None:
        """Add a new optimization strategy.

        Args:
            strategy: The optimization strategy to add.
        """
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove an optimization strategy by name.

        Args:
            strategy_name: Name of the strategy to remove.
        """
        self.strategies = [s for s in self.strategies if s.name != strategy_name]

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply all optimization strategies to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST
        """
        logger.info(f"Starting pipeline optimization with {len(self.strategies)} strategies")

        if self.analyzer_insights:
            logger.info("Using analyzer insights for optimization:")
            dataset_size = self.analyzer_insights.get("dataset_size", 0)
            if dataset_size > 0:
                logger.info(f"  Dataset size: {dataset_size:,} samples")

            text_stats = self.analyzer_insights.get("text_length", {})
            if text_stats:
                mean_len = text_stats.get("mean", 0)
                std_len = text_stats.get("std", 0)
                if mean_len > 0:
                    cv = std_len / mean_len
                    logger.info(f"  Text length CV: {cv:.2f} (mean: {mean_len:.1f}, std: {std_len:.1f})")

        optimized_ast = ast
        for strategy in self.strategies:
            logger.info(f"Applying {strategy.name} strategy...")
            optimized_ast = strategy.optimize(optimized_ast)

        logger.info("Pipeline optimization completed")
        return optimized_ast

    def set_analyzer_insights(self, insights: Dict[str, Any]) -> None:
        """Set analyzer insights for optimization strategies.

        Args:
            insights: Dictionary containing dataset analysis insights
        """
        self.analyzer_insights = insights

        # Update existing strategies that support analyzer insights
        for strategy in self.strategies:
            if hasattr(strategy, "analyzer_insights"):
                strategy.analyzer_insights = insights
                logger.info(f"Updated {strategy.name} with analyzer insights")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization configuration.

        Returns:
            Dictionary containing optimization summary
        """
        summary = {
            "strategies": [s.name for s in self.strategies],
            "analyzer_insights_available": bool(self.analyzer_insights),
            "insights_keys": list(self.analyzer_insights.keys()) if self.analyzer_insights else [],
        }

        if self.analyzer_insights:
            dataset_size = self.analyzer_insights.get("dataset_size", 0)
            summary["dataset_size"] = dataset_size

            # Add data complexity indicators
            text_stats = self.analyzer_insights.get("text_length", {})
            if text_stats:
                mean_len = text_stats.get("mean", 0)
                std_len = text_stats.get("std", 0)
                if mean_len > 0:
                    summary["text_complexity"] = std_len / mean_len

            content_ratios = self.analyzer_insights.get("content_ratios", {})
            multimodal_count = sum(
                1
                for indicator in ["image_ratio", "audio_ratio", "video_ratio"]
                if content_ratios.get(indicator, 0) > 0.1
            )
            summary["multimodal_types"] = multimodal_count

        return summary

    def get_strategy(self, strategy_name: str) -> Optional[OptimizationStrategy]:
        """Get a strategy by name.

        Args:
            strategy_name: Name of the strategy to get.

        Returns:
            The strategy if found, None otherwise.
        """
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                return strategy
        return None

    def get_strategy_names(self) -> List[str]:
        """Get names of all registered strategies.

        Returns:
            List of strategy names.
        """
        return [strategy.name for strategy in self.strategies]

    def clear_strategies(self) -> None:
        """Remove all optimization strategies."""
        self.strategies = []
