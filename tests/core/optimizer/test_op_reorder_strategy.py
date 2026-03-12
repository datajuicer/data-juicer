#!/usr/bin/env python3
"""
Unit tests for OpReorderStrategy.
"""

import unittest

from data_juicer.core.optimizer.op_reorder_strategy import (
    CHEAP_FILTERS,
    EXPENSIVE_FILTERS,
    TEXT_MODIFYING_MAPPERS,
    OpReorderStrategy,
)
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST


class TestOpReorderStrategy(unittest.TestCase):
    """Test cases for OpReorderStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = OpReorderStrategy()

    def _create_ast_from_ops(self, ops_config):
        """Helper to create AST from list of operation configs.

        Args:
            ops_config: List of tuples (name, op_type, config)

        Returns:
            PipelineAST with the operations
        """
        ast = PipelineAST()
        ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        current = ast.root
        for name, op_type, config in ops_config:
            node = OpNode(name=name, op_type=op_type, config=config or {})
            current.children = [node]
            node.parent = current
            current = node

        return ast

    def _get_op_order(self, ast):
        """Get the order of operations from an AST."""
        ops = []
        current = ast.root
        while current.children:
            current = current.children[0]
            if current.op_type in [OpType.MAPPER, OpType.FILTER]:
                ops.append(current.name)
        return ops

    # ==================== Basic Tests ====================

    def test_empty_ast(self):
        """Test optimization of empty AST."""
        ast = PipelineAST()
        result = self.strategy.optimize(ast)
        self.assertIsNone(result.root)

    def test_single_operation(self):
        """Test AST with single operation (no reordering needed)."""
        ops = [("text_length_filter", OpType.FILTER, {})]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        self.assertEqual(order, ["text_length_filter"])

    def test_mappers_only(self):
        """Test AST with only mappers (no reordering)."""
        ops = [
            ("clean_html_mapper", OpType.MAPPER, {}),
            ("clean_links_mapper", OpType.MAPPER, {}),
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Mappers should maintain original order
        self.assertEqual(order, ["clean_html_mapper", "clean_links_mapper"])

    # ==================== Cost-Based Reordering Tests ====================

    def test_cheap_filter_before_expensive(self):
        """Test that cheap filters are moved before expensive filters."""
        ops = [
            ("perplexity_filter", OpType.FILTER, {}),  # Expensive
            ("text_length_filter", OpType.FILTER, {}),  # Cheap
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Cheap filter should come first
        self.assertEqual(order, ["text_length_filter", "perplexity_filter"])

    def test_multiple_cheap_before_multiple_expensive(self):
        """Test multiple cheap filters before multiple expensive filters."""
        ops = [
            ("perplexity_filter", OpType.FILTER, {}),  # Expensive
            ("language_id_score_filter", OpType.FILTER, {}),  # Expensive
            ("text_length_filter", OpType.FILTER, {}),  # Cheap
            ("words_num_filter", OpType.FILTER, {}),  # Cheap
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Cheap filters should come before expensive filters
        cheap_idx = [order.index(f) for f in ["text_length_filter", "words_num_filter"]]
        expensive_idx = [order.index(f) for f in ["perplexity_filter", "language_id_score_filter"]]

        self.assertTrue(max(cheap_idx) < min(expensive_idx))

    def test_already_optimal_order(self):
        """Test that already optimal order is preserved."""
        ops = [
            ("text_length_filter", OpType.FILTER, {}),  # Cheap
            ("words_num_filter", OpType.FILTER, {}),  # Cheap
            ("perplexity_filter", OpType.FILTER, {}),  # Expensive
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Order should remain the same (already optimal)
        self.assertEqual(order[0], "text_length_filter")
        self.assertEqual(order[1], "words_num_filter")
        self.assertEqual(order[2], "perplexity_filter")

    # ==================== Mapper-Filter Dependency Tests ====================

    def test_mappers_before_filters(self):
        """Test that mappers always come before filters."""
        ops = [
            ("text_length_filter", OpType.FILTER, {}),
            ("clean_html_mapper", OpType.MAPPER, {}),
            ("perplexity_filter", OpType.FILTER, {}),
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Mapper should come before all filters
        mapper_idx = order.index("clean_html_mapper")
        filter_indices = [order.index(f) for f in ["text_length_filter", "perplexity_filter"]]

        self.assertTrue(mapper_idx < min(filter_indices))

    def test_mapper_order_preserved(self):
        """Test that mapper order is preserved."""
        ops = [
            ("clean_html_mapper", OpType.MAPPER, {}),
            ("text_length_filter", OpType.FILTER, {}),
            ("fix_unicode_mapper", OpType.MAPPER, {}),
            ("perplexity_filter", OpType.FILTER, {}),
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Mappers should be in original relative order
        mapper_order = [op for op in order if "mapper" in op]
        self.assertEqual(mapper_order, ["clean_html_mapper", "fix_unicode_mapper"])

    # ==================== Filter Cost Detection Tests ====================

    def test_get_filter_cost_cheap(self):
        """Test that cheap filters are correctly identified."""
        for filter_name in CHEAP_FILTERS:
            node = OpNode(name=filter_name, op_type=OpType.FILTER, config={})
            cost = self.strategy._get_filter_cost(node)
            self.assertLess(cost, 10, f"{filter_name} should be cheap (cost < 10)")

    def test_get_filter_cost_expensive(self):
        """Test that expensive filters are correctly identified."""
        for filter_name in EXPENSIVE_FILTERS:
            node = OpNode(name=filter_name, op_type=OpType.FILTER, config={})
            cost = self.strategy._get_filter_cost(node)
            self.assertGreater(cost, 10, f"{filter_name} should be expensive (cost > 10)")

    def test_get_filter_cost_with_model(self):
        """Test that filters with model_key are expensive."""
        node = OpNode(
            name="custom_filter",
            op_type=OpType.FILTER,
            config={"model_key": "some_model"}
        )
        cost = self.strategy._get_filter_cost(node)
        self.assertGreaterEqual(cost, 20)

    def test_get_filter_cost_with_gpu(self):
        """Test that GPU filters are expensive."""
        node = OpNode(
            name="custom_filter",
            op_type=OpType.FILTER,
            config={"accelerator": "cuda"}
        )
        cost = self.strategy._get_filter_cost(node)
        self.assertGreaterEqual(cost, 20)

    # ==================== Analysis Tests ====================

    def test_get_reorder_analysis(self):
        """Test the reorder analysis function."""
        ops = [
            OpNode(name="perplexity_filter", op_type=OpType.FILTER, config={}),
            OpNode(name="text_length_filter", op_type=OpType.FILTER, config={}),
            OpNode(name="clean_html_mapper", op_type=OpType.MAPPER, config={}),
        ]

        analysis = self.strategy.get_reorder_analysis(ops)

        self.assertEqual(analysis["total_operations"], 3)
        self.assertEqual(analysis["filter_count"], 2)
        self.assertEqual(analysis["mapper_count"], 1)
        self.assertIn("text_length_filter", analysis["cheap_filters"])
        self.assertIn("perplexity_filter", analysis["expensive_filters"])
        self.assertTrue(analysis["potential_improvement"])

    def test_get_reorder_analysis_no_improvement(self):
        """Test analysis when order is already optimal."""
        ops = [
            OpNode(name="text_length_filter", op_type=OpType.FILTER, config={}),
            OpNode(name="perplexity_filter", op_type=OpType.FILTER, config={}),
        ]

        analysis = self.strategy.get_reorder_analysis(ops)

        self.assertFalse(analysis["potential_improvement"])
        self.assertEqual(analysis["recommendation"], "Order is already optimal")

    # ==================== Integration Tests ====================

    def test_full_pipeline_reorder(self):
        """Test reordering of a realistic pipeline."""
        ops = [
            ("clean_html_mapper", OpType.MAPPER, {}),
            ("clean_links_mapper", OpType.MAPPER, {}),
            ("perplexity_filter", OpType.FILTER, {}),
            ("language_id_score_filter", OpType.FILTER, {}),
            ("text_length_filter", OpType.FILTER, {"min_len": 100}),
            ("words_num_filter", OpType.FILTER, {"min_num": 10}),
        ]
        ast = self._create_ast_from_ops(ops)

        result = self.strategy.optimize(ast)
        order = self._get_op_order(result)

        # Expected: mappers first (in order), then cheap filters, then expensive
        self.assertEqual(order[0], "clean_html_mapper")
        self.assertEqual(order[1], "clean_links_mapper")

        # Cheap filters should be before expensive filters
        cheap = ["text_length_filter", "words_num_filter"]
        expensive = ["perplexity_filter", "language_id_score_filter"]

        cheap_max_idx = max(order.index(f) for f in cheap)
        expensive_min_idx = min(order.index(f) for f in expensive)

        self.assertLess(cheap_max_idx, expensive_min_idx)


class TestCheapExpensiveFilterLists(unittest.TestCase):
    """Test the filter categorization lists."""

    def test_cheap_filters_exist(self):
        """Test that cheap filters list is not empty."""
        self.assertGreater(len(CHEAP_FILTERS), 0)

    def test_expensive_filters_exist(self):
        """Test that expensive filters list is not empty."""
        self.assertGreater(len(EXPENSIVE_FILTERS), 0)

    def test_no_overlap(self):
        """Test that no filter is in both cheap and expensive lists."""
        overlap = CHEAP_FILTERS & EXPENSIVE_FILTERS
        self.assertEqual(len(overlap), 0, f"Filters in both lists: {overlap}")

    def test_text_modifying_mappers_exist(self):
        """Test that text modifying mappers list is not empty."""
        self.assertGreater(len(TEXT_MODIFYING_MAPPERS), 0)


if __name__ == "__main__":
    unittest.main()
