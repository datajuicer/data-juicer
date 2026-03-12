import unittest

from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST


class TestFilterFusionStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FilterFusionStrategy()

    def _create_ast_with_ops(self, op_specs):
        """Helper to create an AST with operations.

        Args:
            op_specs: List of (name, op_type, config) tuples

        Returns:
            PipelineAST with the specified operations
        """
        ast = PipelineAST()
        ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        current = ast.root
        for name, op_type, config in op_specs:
            node = OpNode(name=name, op_type=op_type, config=config or {})
            current.add_child(node)
            current = node

        return ast

    def _get_op_chain(self, ast):
        """Get the chain of operations from an AST."""
        chain = []
        current = ast.root
        while current:
            if current.name != "root":
                chain.append(current)
            if current.children:
                current = current.children[0]
            else:
                break
        return chain

    def test_optimize_single_filter(self):
        """Test optimization with a single filter - should remain unchanged."""
        ast = self._create_ast_with_ops(
            [("text_length_filter", OpType.FILTER, {"min_len": 10})]
        )

        optimized_ast = self.strategy.optimize(ast)
        chain = self._get_op_chain(optimized_ast)

        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].name, "text_length_filter")

    def test_optimize_multiple_filters_no_shared_vars(self):
        """Test that filters not sharing intermediate vars are not fused."""
        # These filters don't share intermediate variables
        ast = self._create_ast_with_ops(
            [
                ("text_length_filter", OpType.FILTER, {}),
                ("character_repetition_filter", OpType.FILTER, {}),
            ]
        )

        optimized_ast = self.strategy.optimize(ast)
        chain = self._get_op_chain(optimized_ast)

        # Should remain separate (no fusion)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0].name, "text_length_filter")
        self.assertEqual(chain[1].name, "character_repetition_filter")

    def test_optimize_filters_sharing_intermediate_vars(self):
        """Test that filters sharing intermediate vars (INTER_WORDS) are fused."""
        # These filters share __dj__words intermediate variable
        ast = self._create_ast_with_ops(
            [
                ("words_num_filter", OpType.FILTER, {}),
                ("word_repetition_filter", OpType.FILTER, {}),
            ]
        )

        optimized_ast = self.strategy.optimize(ast)
        chain = self._get_op_chain(optimized_ast)

        # Should be fused into 1 operation
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].name, "fused_filter")

        # Check fused config
        fused_config = chain[0].config.get("general_fused_op", {})
        self.assertIn("fused_op_list", fused_config)
        self.assertIn("detailed_ops", fused_config)
        self.assertEqual(len(fused_config["detailed_ops"]), 2)

    def test_optimize_mixed_ops(self):
        """Test that mappers break filter fusion groups."""
        ast = self._create_ast_with_ops(
            [
                ("words_num_filter", OpType.FILTER, {}),
                ("clean_copyright_mapper", OpType.MAPPER, {}),
                ("word_repetition_filter", OpType.FILTER, {}),
            ]
        )

        optimized_ast = self.strategy.optimize(ast)
        chain = self._get_op_chain(optimized_ast)

        # Should have 3 ops (mapper breaks fusion)
        self.assertEqual(len(chain), 3)
        self.assertEqual(chain[0].name, "words_num_filter")
        self.assertEqual(chain[1].name, "clean_copyright_mapper")
        self.assertEqual(chain[2].name, "word_repetition_filter")

    def test_optimize_empty_pipeline(self):
        """Test optimization with an empty pipeline."""
        ast = PipelineAST()  # No root
        optimized_ast = self.strategy.optimize(ast)
        self.assertIsNone(optimized_ast.root)

    def test_optimize_with_probe_results(self):
        """Test optimization with probe results (for future integration)."""
        probe_results = {
            "words_num_filter": {"speed": 0.5},
            "word_repetition_filter": {"speed": 0.3},
        }
        strategy = FilterFusionStrategy(probe_results=probe_results)

        ast = self._create_ast_with_ops(
            [
                ("words_num_filter", OpType.FILTER, {}),
                ("word_repetition_filter", OpType.FILTER, {}),
            ]
        )

        optimized_ast = strategy.optimize(ast)
        chain = self._get_op_chain(optimized_ast)

        # Should still fuse (probe_results used for other purposes)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].name, "fused_filter")

    def test_filters_with_line_vars(self):
        """Test that filters sharing INTER_LINES are fused."""
        # These filters share __dj__lines intermediate variable
        ast = self._create_ast_with_ops(
            [
                ("average_line_length_filter", OpType.FILTER, {}),
                ("maximum_line_length_filter", OpType.FILTER, {}),
            ]
        )

        optimized_ast = self.strategy.optimize(ast)
        chain = self._get_op_chain(optimized_ast)

        # Should be fused
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].name, "fused_filter")


if __name__ == "__main__":
    unittest.main()
