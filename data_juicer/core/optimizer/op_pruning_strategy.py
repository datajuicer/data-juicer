#!/usr/bin/env python3
"""
Operation Pruning Strategy for the core optimizer.

This strategy identifies and removes no-op or redundant operations from the pipeline:
1. No-op filters: Filters with pass-through conditions (e.g., min=0, max=inf)
2. Duplicate operations: Consecutive identical operations
3. Redundant operations: Operations that have no effect given their config

Key principle: Remove operations that don't change the output to save processing time.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..pipeline_ast import OpNode, OpType, PipelineAST
from .strategy import OptimizationStrategy, register_strategy

# Default values that indicate a filter parameter is effectively disabled
# (filter_name, param_name) -> pass-through value
NO_OP_DEFAULTS = {
    # text_length_filter: min_len=0 means no minimum, max_len>=10^9 means no maximum
    ("text_length_filter", "min_len"): 0,
    ("text_length_filter", "max_len"): 10**9,
    # words_num_filter: min_num=0 means no minimum, max_num>=10^9 means no maximum
    ("words_num_filter", "min_num"): 0,
    ("words_num_filter", "max_num"): 10**9,
    # alphanumeric_filter: min_ratio=0 and max_ratio=1 means pass everything
    ("alphanumeric_filter", "min_ratio"): 0.0,
    ("alphanumeric_filter", "max_ratio"): 1.0,
    # special_characters_filter: min_ratio=0 and max_ratio=1 means pass everything
    ("special_characters_filter", "min_ratio"): 0.0,
    ("special_characters_filter", "max_ratio"): 1.0,
    # character_repetition_filter: max_ratio=1 means allow any repetition
    ("character_repetition_filter", "max_ratio"): 1.0,
    # word_repetition_filter: max_ratio=1 means allow any repetition
    ("word_repetition_filter", "max_ratio"): 1.0,
    # average_line_length_filter: extreme values mean pass everything
    ("average_line_length_filter", "min_len"): 0,
    ("average_line_length_filter", "max_len"): 10**9,
    # maximum_line_length_filter: extreme values mean pass everything
    ("maximum_line_length_filter", "min_len"): 0,
    ("maximum_line_length_filter", "max_len"): 10**9,
    # token_num_filter: min_num=0 and max_num>=10^9 means pass everything
    ("token_num_filter", "min_num"): 0,
    ("token_num_filter", "max_num"): 10**9,
    # suffix_filter: empty suffixes means pass everything
    ("suffix_filter", "suffixes"): [],
    # stopwords_filter: min_ratio=0 means pass everything
    ("stopwords_filter", "min_ratio"): 0.0,
    # flagged_words_filter: max_ratio=1 means pass everything
    ("flagged_words_filter", "max_ratio"): 1.0,
    # perplexity_filter: extreme values mean pass everything
    ("perplexity_filter", "min_ppl"): 0.0,
    ("perplexity_filter", "max_ppl"): float("inf"),
    # language_id_score_filter: min_score=0 means pass everything
    ("language_id_score_filter", "min_score"): 0.0,
}

# Filters that can be detected as no-ops based on their config
PRUNABLE_FILTERS = {
    "text_length_filter",
    "words_num_filter",
    "alphanumeric_filter",
    "special_characters_filter",
    "character_repetition_filter",
    "word_repetition_filter",
    "average_line_length_filter",
    "maximum_line_length_filter",
    "token_num_filter",
    "suffix_filter",
    "stopwords_filter",
    "flagged_words_filter",
    "perplexity_filter",
    "language_id_score_filter",
}


@register_strategy("op_pruning")
class OpPruningStrategy(OptimizationStrategy):
    """
    Strategy that identifies and removes no-op or redundant operations.

    Operations are pruned if:
    1. They are filters with pass-through conditions (no-op filters)
    2. They are duplicates of the immediately preceding operation
    3. They have empty or disabled configurations
    """

    def __init__(self):
        """Initialize the operation pruning strategy."""
        super().__init__(name="op_pruning")
        self._pruned_ops: List[str] = []

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """
        Apply operation pruning to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST with no-op operations removed
        """
        if not ast.root:
            return ast

        # Get the chain of operations
        operations = self._get_operation_chain(ast.root)
        if not operations:
            return ast

        # Reset tracking
        self._pruned_ops = []

        # Identify operations to prune
        ops_to_keep = []
        prev_op: Optional[OpNode] = None

        for op in operations:
            should_prune, reason = self._should_prune(op, prev_op)

            if should_prune:
                self._pruned_ops.append(f"{op.name} ({reason})")
                logger.info(f"Pruning operation: {op.name} - {reason}")
            else:
                ops_to_keep.append(op)
                prev_op = op

        # Log summary
        if self._pruned_ops:
            logger.info(f"Op pruning: removed {len(self._pruned_ops)} operations")
            for pruned in self._pruned_ops:
                logger.info(f"  - {pruned}")
        else:
            logger.debug("Op pruning: no operations to prune")

        # Build new AST with remaining operations
        return self._build_pruned_ast(ast, ops_to_keep)

    def _should_prune(self, op: OpNode, prev_op: Optional[OpNode]) -> Tuple[bool, str]:
        """
        Determine if an operation should be pruned.

        Args:
            op: The operation to check
            prev_op: The previous operation (for duplicate detection)

        Returns:
            Tuple of (should_prune, reason)
        """
        # Check for duplicate consecutive operation
        if prev_op is not None and self._is_duplicate(op, prev_op):
            return True, "duplicate of previous operation"

        # Check for no-op filter
        if op.op_type == OpType.FILTER:
            is_noop, reason = self._is_noop_filter(op)
            if is_noop:
                return True, reason

        # Check for empty mapper (no actual transformation)
        if op.op_type == OpType.MAPPER:
            is_noop, reason = self._is_noop_mapper(op)
            if is_noop:
                return True, reason

        return False, ""

    def _is_duplicate(self, op: OpNode, prev_op: OpNode) -> bool:
        """
        Check if an operation is a duplicate of the previous one.

        Args:
            op: Current operation
            prev_op: Previous operation

        Returns:
            True if operations are duplicates
        """
        if op.name != prev_op.name:
            return False

        # Compare configs (ignoring order)
        return self._configs_equal(op.config, prev_op.config)

    def _configs_equal(self, config1: Dict, config2: Dict) -> bool:
        """Compare two configs for equality.

        Only compares filter-specific parameters, ignoring internal attributes
        like accelerator, batch_size, num_proc, etc.
        """
        if config1 is None and config2 is None:
            return True
        if config1 is None or config2 is None:
            return False

        # Get the actual op config (may be nested)
        cfg1 = self._extract_op_config(config1)
        cfg2 = self._extract_op_config(config2)

        # Keys to ignore (internal attributes common to all ops)
        IGNORED_KEYS = {
            "accelerator",
            "batch_size",
            "num_proc",
            "cpu_required",
            "mem_required",
            "gpu_required",
            "text_key",
            "image_key",
            "audio_key",
            "video_key",
            "work_dir",
            "skip_op_error",
            "turbo",
            "auto_op_parallelism",
            "num_cpus",
            "num_gpus",
            "memory",
            "ray_execution_mode",
            "runtime_env",
            "history_key",
            "query_key",
            "response_key",
            "prompt_key",
            "system_key",
            "instruction_key",
            "image_bytes_key",
            "index_key",
            "batch_mode",
        }

        # Get relevant keys only
        keys1 = {k for k in cfg1.keys() if not k.startswith("_") and k not in IGNORED_KEYS}
        keys2 = {k for k in cfg2.keys() if not k.startswith("_") and k not in IGNORED_KEYS}

        # Keys must match
        if keys1 != keys2:
            return False

        # Compare values for relevant keys
        for key in keys1:
            if cfg1.get(key) != cfg2.get(key):
                return False

        return True

    def _extract_op_config(self, config: Dict) -> Dict:
        """Extract the actual operation config from potentially nested structure."""
        if not config:
            return {}

        # If config has a single key that's the op name, unwrap it
        if len(config) == 1:
            key = list(config.keys())[0]
            if isinstance(config[key], dict):
                return config[key]

        return config

    def _is_noop_filter(self, op: OpNode) -> Tuple[bool, str]:
        """
        Check if a filter is a no-op based on its configuration.

        Args:
            op: The filter operation to check

        Returns:
            Tuple of (is_noop, reason)
        """
        op_name = op.name
        config = self._extract_op_config(op.config) or {}

        if op_name not in PRUNABLE_FILTERS:
            return False, ""

        # Check specific filter types
        # Use 10**8 as threshold for "effectively infinite" to catch common large values
        LARGE_THRESHOLD = 10**8

        if op_name == "text_length_filter":
            min_len = config.get("min_len", 0)
            max_len = config.get("max_len", 10**9)
            if min_len <= 0 and max_len >= LARGE_THRESHOLD:
                return True, "min_len=0 and max_len=inf (passes everything)"

        elif op_name == "words_num_filter":
            min_num = config.get("min_num", 0)
            max_num = config.get("max_num", 10**9)
            if min_num <= 0 and max_num >= LARGE_THRESHOLD:
                return True, "min_num=0 and max_num=inf (passes everything)"

        elif op_name == "alphanumeric_filter":
            min_ratio = config.get("min_ratio", 0.0)
            max_ratio = config.get("max_ratio", 1.0)
            if min_ratio <= 0.0 and max_ratio >= 1.0:
                return True, "min_ratio=0 and max_ratio=1 (passes everything)"

        elif op_name == "special_characters_filter":
            min_ratio = config.get("min_ratio", 0.0)
            max_ratio = config.get("max_ratio", 1.0)
            if min_ratio <= 0.0 and max_ratio >= 1.0:
                return True, "min_ratio=0 and max_ratio=1 (passes everything)"

        elif op_name in ("character_repetition_filter", "word_repetition_filter"):
            max_ratio = config.get("max_ratio", 1.0)
            if max_ratio >= 1.0:
                return True, "max_ratio=1 (allows any repetition)"

        elif op_name in ("average_line_length_filter", "maximum_line_length_filter"):
            min_len = config.get("min_len", 0)
            max_len = config.get("max_len", 10**9)
            if min_len <= 0 and max_len >= LARGE_THRESHOLD:
                return True, "min_len=0 and max_len=inf (passes everything)"

        elif op_name == "token_num_filter":
            min_num = config.get("min_num", 0)
            max_num = config.get("max_num", 10**9)
            if min_num <= 0 and max_num >= LARGE_THRESHOLD:
                return True, "min_num=0 and max_num=inf (passes everything)"

        elif op_name == "suffix_filter":
            suffixes = config.get("suffixes", [])
            if not suffixes:
                return True, "empty suffixes list (passes everything)"

        elif op_name == "stopwords_filter":
            min_ratio = config.get("min_ratio", 0.0)
            if min_ratio <= 0.0:
                return True, "min_ratio=0 (passes everything)"

        elif op_name == "flagged_words_filter":
            max_ratio = config.get("max_ratio", 1.0)
            if max_ratio >= 1.0:
                return True, "max_ratio=1 (passes everything)"

        elif op_name == "perplexity_filter":
            min_ppl = config.get("min_ppl", 0.0)
            max_ppl = config.get("max_ppl", float("inf"))
            if min_ppl <= 0.0 and max_ppl >= LARGE_THRESHOLD:
                return True, "min_ppl=0 and max_ppl=inf (passes everything)"

        elif op_name == "language_id_score_filter":
            min_score = config.get("min_score", 0.0)
            if min_score <= 0.0:
                return True, "min_score=0 (passes everything)"

        return False, ""

    def _is_noop_mapper(self, op: OpNode) -> Tuple[bool, str]:
        """
        Check if a mapper is a no-op based on its configuration.

        Args:
            op: The mapper operation to check

        Returns:
            Tuple of (is_noop, reason)
        """
        op_name = op.name
        config = self._extract_op_config(op.config) or {}

        # Check for specific no-op mapper conditions
        if op_name == "remove_specific_chars_mapper":
            chars = config.get("chars_to_remove", "")
            if not chars:
                return True, "no characters specified to remove"

        elif op_name == "replace_content_mapper":
            pattern = config.get("pattern", "")
            if not pattern:
                return True, "no pattern specified to replace"

        return False, ""

    def _build_pruned_ast(self, original_ast: PipelineAST, operations: List[OpNode]) -> PipelineAST:
        """
        Build a new AST with only the kept operations.

        Args:
            original_ast: Original AST
            operations: List of operations to keep

        Returns:
            New AST with pruned operations removed
        """
        new_ast = PipelineAST()
        new_ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        if not operations:
            return new_ast

        # Build the chain
        current = new_ast.root
        for op in operations:
            new_node = OpNode(
                name=op.name,
                op_type=op.op_type,
                config=op.config.copy() if op.config else {},
            )
            current.add_child(new_node)
            current = new_node

        return new_ast

    def get_pruned_operations(self) -> List[str]:
        """Get the list of pruned operations from the last optimization."""
        return self._pruned_ops.copy()
