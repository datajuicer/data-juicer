import unittest

from data_juicer.core.pipeline_ast import PipelineAST, OpNode, OpType
from data_juicer.core.optimizer.op_pruning_strategy import (
    OpPruningStrategy,
    NO_OP_DEFAULTS,
    PRUNABLE_FILTERS,
)


class TestOpPruningStrategy(unittest.TestCase):
    """Test cases for OpPruningStrategy."""

    def setUp(self):
        self.strategy = OpPruningStrategy()

    def _create_ast_with_ops(self, ops_config):
        """Helper to create AST with given operations."""
        ast = PipelineAST()
        ast.root = OpNode(name='root', op_type=OpType.ROOT, config={})

        current = ast.root
        for name, op_type, config in ops_config:
            node = OpNode(name=name, op_type=op_type, config=config)
            current.add_child(node)
            current = node

        return ast

    def _count_ops(self, ast):
        """Count non-root operations in AST."""
        count = 0
        def traverse(node):
            nonlocal count
            if node.name != 'root':
                count += 1
            for child in node.children:
                traverse(child)
        if ast.root:
            traverse(ast.root)
        return count

    # ===================
    # No-op filter tests
    # ===================

    def test_prune_noop_text_length_filter(self):
        """Test pruning text_length_filter with pass-through conditions."""
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 0, 'max_len': 10**9}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)
        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 1)
        self.assertIn('passes everything', pruned[0])

    def test_keep_text_length_filter_with_constraints(self):
        """Test keeping text_length_filter with actual constraints."""
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 50000}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 1)
        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 0)

    def test_prune_noop_words_num_filter(self):
        """Test pruning words_num_filter with pass-through conditions."""
        ast = self._create_ast_with_ops([
            ('words_num_filter', OpType.FILTER, {'min_num': 0, 'max_num': 10**9}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)
        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 1)

    def test_prune_noop_alphanumeric_filter(self):
        """Test pruning alphanumeric_filter with full range."""
        ast = self._create_ast_with_ops([
            ('alphanumeric_filter', OpType.FILTER, {'min_ratio': 0.0, 'max_ratio': 1.0}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_prune_noop_special_characters_filter(self):
        """Test pruning special_characters_filter with full range."""
        ast = self._create_ast_with_ops([
            ('special_characters_filter', OpType.FILTER, {'min_ratio': 0.0, 'max_ratio': 1.0}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_prune_noop_character_repetition_filter(self):
        """Test pruning character_repetition_filter with max_ratio=1."""
        ast = self._create_ast_with_ops([
            ('character_repetition_filter', OpType.FILTER, {'rep_len': 10, 'max_ratio': 1.0}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_prune_noop_word_repetition_filter(self):
        """Test pruning word_repetition_filter with max_ratio=1."""
        ast = self._create_ast_with_ops([
            ('word_repetition_filter', OpType.FILTER, {'rep_len': 10, 'max_ratio': 1.0}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_prune_noop_stopwords_filter(self):
        """Test pruning stopwords_filter with min_ratio=0."""
        ast = self._create_ast_with_ops([
            ('stopwords_filter', OpType.FILTER, {'min_ratio': 0.0}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_prune_noop_flagged_words_filter(self):
        """Test pruning flagged_words_filter with max_ratio=1."""
        ast = self._create_ast_with_ops([
            ('flagged_words_filter', OpType.FILTER, {'max_ratio': 1.0}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_prune_noop_suffix_filter_empty(self):
        """Test pruning suffix_filter with empty suffixes."""
        ast = self._create_ast_with_ops([
            ('suffix_filter', OpType.FILTER, {'suffixes': []}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_keep_suffix_filter_with_suffixes(self):
        """Test keeping suffix_filter with actual suffixes."""
        ast = self._create_ast_with_ops([
            ('suffix_filter', OpType.FILTER, {'suffixes': ['.txt', '.md']}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 1)

    # ===================
    # Duplicate detection tests
    # ===================

    def test_prune_duplicate_consecutive_filter(self):
        """Test pruning consecutive duplicate filters."""
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}),
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 1)
        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 1)
        self.assertIn('duplicate', pruned[0])

    def test_keep_non_duplicate_filters(self):
        """Test keeping filters with different configs."""
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}),
            ('text_length_filter', OpType.FILTER, {'min_len': 200, 'max_len': 5000}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 2)

    def test_duplicate_detection_ignores_internal_keys(self):
        """Test that duplicate detection ignores internal attributes."""
        # Two filters with same semantic config but different internal attributes
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {
                'min_len': 100, 'max_len': 5000,
                'accelerator': None, 'batch_size': 32, 'num_proc': 4
            }),
            ('text_length_filter', OpType.FILTER, {
                'min_len': 100, 'max_len': 5000,
                'accelerator': 'cuda', 'batch_size': 64, 'num_proc': 8
            }),
        ])

        optimized = self.strategy.optimize(ast)

        # Should be pruned as duplicate since semantic config is the same
        self.assertEqual(self._count_ops(optimized), 1)

    def test_no_duplicate_for_non_consecutive(self):
        """Test that non-consecutive identical ops are not pruned as duplicates."""
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}),
            ('words_num_filter', OpType.FILTER, {'min_num': 10, 'max_num': 1000}),
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}),
        ])

        optimized = self.strategy.optimize(ast)

        # First and third are same but not consecutive - should keep all 3
        self.assertEqual(self._count_ops(optimized), 3)

    # ===================
    # No-op mapper tests
    # ===================

    def test_prune_noop_remove_specific_chars_mapper(self):
        """Test pruning remove_specific_chars_mapper with empty chars."""
        ast = self._create_ast_with_ops([
            ('remove_specific_chars_mapper', OpType.MAPPER, {'chars_to_remove': ''}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    def test_keep_remove_specific_chars_mapper_with_chars(self):
        """Test keeping remove_specific_chars_mapper with actual chars."""
        ast = self._create_ast_with_ops([
            ('remove_specific_chars_mapper', OpType.MAPPER, {'chars_to_remove': '@#$'}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 1)

    def test_prune_noop_replace_content_mapper(self):
        """Test pruning replace_content_mapper with empty pattern."""
        ast = self._create_ast_with_ops([
            ('replace_content_mapper', OpType.MAPPER, {'pattern': ''}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)

    # ===================
    # Mixed operation tests
    # ===================

    def test_mixed_pipeline_pruning(self):
        """Test pruning a mixed pipeline with various redundant operations."""
        ast = self._create_ast_with_ops([
            ('clean_html_mapper', OpType.MAPPER, {}),                              # Keep
            ('text_length_filter', OpType.FILTER, {'min_len': 0, 'max_len': 10**9}),  # Prune (no-op)
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}), # Keep
            ('words_num_filter', OpType.FILTER, {'min_num': 0, 'max_num': 10**9}),   # Prune (no-op)
            ('words_num_filter', OpType.FILTER, {'min_num': 20, 'max_num': 1000}),   # Keep
            ('special_characters_filter', OpType.FILTER, {'min_ratio': 0.0, 'max_ratio': 0.3}), # Keep
            ('special_characters_filter', OpType.FILTER, {'min_ratio': 0.0, 'max_ratio': 0.3}), # Prune (duplicate)
        ])

        optimized = self.strategy.optimize(ast)

        # Should keep: clean_html_mapper, text_length_filter (100-5000),
        #              words_num_filter (20-1000), special_characters_filter (0-0.3)
        self.assertEqual(self._count_ops(optimized), 4)

        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 3)  # 2 no-ops + 1 duplicate

    def test_empty_ast(self):
        """Test optimization of empty AST."""
        ast = PipelineAST()

        optimized = self.strategy.optimize(ast)

        self.assertIsNone(optimized.root)

    def test_all_ops_pruned(self):
        """Test when all operations are pruned."""
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 0, 'max_len': 10**9}),
            ('words_num_filter', OpType.FILTER, {'min_num': 0, 'max_num': 10**9}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 0)
        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 2)

    def test_no_ops_pruned(self):
        """Test when no operations are pruned."""
        ast = self._create_ast_with_ops([
            ('clean_html_mapper', OpType.MAPPER, {}),
            ('text_length_filter', OpType.FILTER, {'min_len': 100, 'max_len': 5000}),
            ('words_num_filter', OpType.FILTER, {'min_num': 20, 'max_num': 1000}),
        ])

        optimized = self.strategy.optimize(ast)

        self.assertEqual(self._count_ops(optimized), 3)
        pruned = self.strategy.get_pruned_operations()
        self.assertEqual(len(pruned), 0)

    # ===================
    # Edge cases
    # ===================

    def test_large_threshold_value(self):
        """Test that large values close to threshold are handled correctly."""
        # 999999999 is just under 10**9 but should still be caught
        ast = self._create_ast_with_ops([
            ('text_length_filter', OpType.FILTER, {'min_len': 0, 'max_len': 999999999}),
        ])

        optimized = self.strategy.optimize(ast)

        # 999999999 >= 10**8 (LARGE_THRESHOLD), so should be pruned
        self.assertEqual(self._count_ops(optimized), 0)

    def test_unknown_filter_not_pruned(self):
        """Test that unknown filter types are not pruned."""
        ast = self._create_ast_with_ops([
            ('custom_unknown_filter', OpType.FILTER, {'min_len': 0, 'max_len': 10**9}),
        ])

        optimized = self.strategy.optimize(ast)

        # Unknown filter type should be kept
        self.assertEqual(self._count_ops(optimized), 1)

    def test_preserves_operation_order(self):
        """Test that remaining operations preserve original order."""
        ast = self._create_ast_with_ops([
            ('clean_html_mapper', OpType.MAPPER, {}),
            ('text_length_filter', OpType.FILTER, {'min_len': 0, 'max_len': 10**9}),  # Pruned
            ('words_num_filter', OpType.FILTER, {'min_num': 20, 'max_num': 1000}),
            ('alphanumeric_filter', OpType.FILTER, {'min_ratio': 0.5, 'max_ratio': 1.0}),
        ])

        optimized = self.strategy.optimize(ast)

        # Collect remaining op names
        names = []
        def traverse(node):
            if node.name != 'root':
                names.append(node.name)
            for child in node.children:
                traverse(child)
        traverse(optimized.root)

        self.assertEqual(names, ['clean_html_mapper', 'words_num_filter', 'alphanumeric_filter'])


class TestConfigsEqual(unittest.TestCase):
    """Test cases for _configs_equal helper method."""

    def setUp(self):
        self.strategy = OpPruningStrategy()

    def test_both_none(self):
        """Test comparing two None configs."""
        self.assertTrue(self.strategy._configs_equal(None, None))

    def test_one_none(self):
        """Test comparing None with non-None config."""
        self.assertFalse(self.strategy._configs_equal(None, {}))
        self.assertFalse(self.strategy._configs_equal({}, None))

    def test_equal_configs(self):
        """Test comparing equal configs."""
        config = {'min_len': 100, 'max_len': 5000}
        self.assertTrue(self.strategy._configs_equal(config, config.copy()))

    def test_different_values(self):
        """Test comparing configs with different values."""
        config1 = {'min_len': 100, 'max_len': 5000}
        config2 = {'min_len': 100, 'max_len': 6000}
        self.assertFalse(self.strategy._configs_equal(config1, config2))

    def test_different_keys(self):
        """Test comparing configs with different keys."""
        config1 = {'min_len': 100}
        config2 = {'min_num': 100}
        self.assertFalse(self.strategy._configs_equal(config1, config2))

    def test_ignores_internal_keys(self):
        """Test that internal keys are ignored in comparison."""
        config1 = {'min_len': 100, 'accelerator': None, 'batch_size': 32}
        config2 = {'min_len': 100, 'accelerator': 'cuda', 'batch_size': 64}
        self.assertTrue(self.strategy._configs_equal(config1, config2))


if __name__ == '__main__':
    unittest.main()
