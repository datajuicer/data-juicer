from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.optimizer.strategy import OptimizationStrategy, register_strategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST
from data_juicer.utils.constant import InterVars, StatsKeys
from data_juicer.utils.registry import Registry

# Type of intermediate vars
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)
LOADED_IMAGES = Registry(InterVars.loaded_images)
LOADED_AUDIOS = Registry(InterVars.loaded_audios)
LOADED_VIDEOS = Registry(InterVars.loaded_videos)
INTER_SAMPLED_FRAMES = Registry(InterVars.sampled_frames)

ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_AUDIOS, LOADED_IMAGES, LOADED_VIDEOS, INTER_SAMPLED_FRAMES]


@register_strategy("filter_fusion")
class FilterFusionStrategy(OptimizationStrategy):
    """Strategy for fusing filter operations in the pipeline."""

    def __init__(
        self, probe_results: Optional[Dict[str, Any]] = None, analyzer_insights: Optional[Dict[str, Any]] = None
    ):
        """Initialize the filter fusion strategy.

        Args:
            probe_results: Optional dictionary containing operation speeds
            analyzer_insights: Optional dictionary containing dataset analysis insights
        """
        super().__init__(name="filter_fusion")
        self.probe_results = probe_results or {}
        self.analyzer_insights = analyzer_insights or {}

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply filter fusion to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST
        """
        if not ast.root:
            return ast

        # Create a new AST
        new_ast = PipelineAST()
        new_ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        # Get all unique operation chains
        op_chains = self._get_unique_op_chains(ast.root)

        # Process each chain
        current = new_ast.root
        for chain in op_chains:
            # Group filter operations with analyzer insights
            filter_groups = self._group_filters_with_insights(chain)

            for group in filter_groups:
                if len(group) > 1:
                    # Create fused operation with clean naming
                    fused_name = "fused_filter"
                    detailed_ops = [n.name for n in group]
                    logger.info(f"Fusing filter operations into {fused_name}: {detailed_ops}")

                    # Create operation configs
                    op_configs = []
                    for op in group:
                        op_config = {op.name: op.config or {}}
                        op_configs.append(op_config)

                    # Create fused node
                    fused_node = OpNode(
                        name=fused_name,
                        op_type=OpType.FILTER,
                        config={
                            "general_fused_op": {
                                "fused_op_list": op_configs,
                                "detailed_ops": detailed_ops,  # For display purposes
                            }
                        },
                    )
                    current.add_child(fused_node)
                    current = fused_node
                else:
                    # Keep single operations as is
                    new_node = OpNode(name=group[0].name, op_type=group[0].op_type, config=group[0].config or {})
                    current.add_child(new_node)
                    current = new_node

        return new_ast

    def _get_unique_op_chains(self, node: OpNode) -> List[List[OpNode]]:
        """Get unique chains of operations from the tree.

        Args:
            node: Root node of the tree

        Returns:
            List of unique operation chains
        """
        chains = []
        seen_chains = set()

        def traverse(current: OpNode, chain: List[OpNode]):
            if not current.children:
                # End of chain, check if we've seen this sequence before
                chain_key = tuple(n.name for n in chain)
                if chain_key not in seen_chains:
                    chains.append(chain.copy())
                    seen_chains.add(chain_key)
                return

            for child in current.children:
                chain.append(child)
                traverse(child, chain)
                chain.pop()

        traverse(node, [])
        return chains

    def _group_filters_with_insights(self, chain: List[OpNode]) -> List[List[OpNode]]:
        """Group filter operations using analyzer insights for better decisions.

        Args:
            chain: List of operations in the pipeline

        Returns:
            List of filter operation groups
        """
        groups = []
        current_group = []

        for node in chain:
            if not PipelineAST.is_filter_op(node):
                # If we encounter a non-filter, finalize current group
                if current_group:
                    groups.append(current_group)
                    current_group = []
                # Add the non-filter node as a separate group
                groups.append([node])
            else:
                # This is a filter node
                if not current_group:
                    # Start a new group
                    current_group = [node]
                else:
                    # Check if current filter can be fused with the group using insights
                    if self._can_fuse_with_group_insights(node, current_group):
                        current_group.append(node)
                    else:
                        # Finalize current group and start a new one
                        groups.append(current_group)
                        current_group = [node]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _can_fuse_with_group_insights(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if a filter can be fused with a group using analyzer insights.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if the operation can be fused with the group
        """
        # Basic dependency check
        for op in group:
            if self._has_dependency(node, op) or self._has_dependency(op, node):
                return False

        # Use smart complex filter fusion for better performance
        if not self._smart_complex_filter_fusion(node, group):
            return False

        # Use analyzer insights for advanced decisions
        if self.analyzer_insights:
            return self._analyzer_based_fusion_decision(node, group)

        return True

    def _analyzer_based_fusion_decision(self, node: OpNode, group: List[OpNode]) -> bool:
        """Make fusion decisions based on analyzer insights.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if fusion is recommended based on data characteristics
        """
        # Get dataset characteristics from analyzer
        dataset_size = self.analyzer_insights.get("dataset_size", 0)
        text_length_stats = self.analyzer_insights.get("text_length", {})
        content_ratios = self.analyzer_insights.get("content_ratios", {})

        # Decision 1: Large datasets benefit more from fusion
        if dataset_size > 100000:
            logger.debug(f"Large dataset ({dataset_size:,} samples) - favoring fusion")
            return True

        # Decision 2: High variance in text length suggests complex processing
        if text_length_stats:
            mean_length = text_length_stats.get("mean", 0)
            std_length = text_length_stats.get("std", 0)
            if mean_length > 0 and std_length / mean_length > 1.5:
                logger.debug("High text length variance - favoring fusion for complex data")
                return True

        # Decision 3: Mixed content types suggest complex processing
        multimodal_indicators = ["image_ratio", "audio_ratio", "video_ratio"]
        multimodal_count = sum(1 for indicator in multimodal_indicators if content_ratios.get(indicator, 0) > 0.1)

        if multimodal_count > 1:
            logger.debug(f"Multimodal content detected ({multimodal_count} types) - favoring fusion")
            return True

        # Decision 4: Check if operations are computationally similar
        return self._check_computational_similarity(node, group)

    def _check_computational_similarity(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if operations have similar computational characteristics.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if operations are computationally similar
        """
        node_complexity = self._get_operation_complexity(node.name)
        group_complexities = [self._get_operation_complexity(op.name) for op in group]

        # Prefer grouping operations of similar complexity
        if node_complexity in group_complexities:
            return True

        # Allow mixing simple and medium operations
        if node_complexity == "simple" and all(c in ["simple", "medium"] for c in group_complexities):
            return True
        if node_complexity == "medium" and all(c in ["simple", "medium"] for c in group_complexities):
            return True

        return False

    def _get_operation_complexity(self, op_name: str) -> str:
        """Get the computational complexity of an operation using dynamic analysis.

        Args:
            op_name: Name of the operation

        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        # First, try to get complexity from operation metadata if available
        complexity = self._get_complexity_from_metadata(op_name)
        if complexity:
            return complexity

        # Fallback to pattern-based analysis
        return self._analyze_complexity_by_pattern(op_name)

    def _get_complexity_from_metadata(self, op_name: str) -> Optional[str]:
        """Get complexity from operation metadata or runtime analysis."""
        # Try to load the actual operation to analyze its characteristics
        try:
            from data_juicer.ops import load_ops

            # Create a minimal config to load the operation
            op_config = {op_name: {}}
            loaded_ops = load_ops([op_config])

            if loaded_ops:
                op = loaded_ops[0]
                return self._analyze_operation_complexity(op)
        except Exception:
            pass

        return None

    def _analyze_operation_complexity(self, op) -> str:
        """Analyze operation complexity based on its actual characteristics."""
        complexity_indicators = {"simple": 0, "medium": 0, "complex": 0}

        # Check for model dependencies (indicates complexity)
        if hasattr(op, "config") and op.config:
            config = op.config
            if any(key in config for key in ["model_key", "sp_model_key", "kl_model_key"]):
                complexity_indicators["complex"] += 2
            if "lang" in config:
                complexity_indicators["medium"] += 1

        # Check for external dependencies
        if hasattr(op, "_name"):
            op_name = op._name.lower()

            # Language model dependencies
            if any(keyword in op_name for keyword in ["perplexity", "language", "spacy", "nlp"]):
                complexity_indicators["complex"] += 2

            # Statistical analysis dependencies
            if any(keyword in op_name for keyword in ["repetition", "ratio", "statistics"]):
                complexity_indicators["medium"] += 1

            # Simple text processing
            if any(keyword in op_name for keyword in ["length", "words", "characters"]):
                complexity_indicators["simple"] += 1

        # Check method complexity
        if hasattr(op, "compute_stats_batched"):
            # Analyze the method signature and docstring for complexity hints
            method = op.compute_stats_batched
            if hasattr(method, "__doc__") and method.__doc__:
                doc = method.__doc__.lower()
                if any(keyword in doc for keyword in ["model", "spacy", "nlp", "language"]):
                    complexity_indicators["complex"] += 1
                elif any(keyword in doc for keyword in ["statistics", "ratio", "analysis"]):
                    complexity_indicators["medium"] += 1

        # Determine final complexity
        max_complexity = max(complexity_indicators.items(), key=lambda x: x[1])
        if max_complexity[1] == 0:
            return "medium"  # Default
        return max_complexity[0]

    def _analyze_complexity_by_pattern(self, op_name: str) -> str:
        """Analyze complexity based on operation name patterns."""
        op_name_lower = op_name.lower()

        # Simple operations (basic text processing)
        simple_patterns = [
            "text_length",
            "words_num",
            "character_repetition",
            "average_line_length",
            "maximum_line_length",
        ]

        # Medium complexity operations (statistical analysis)
        medium_patterns = ["word_repetition", "special_characters", "alphanumeric", "stopwords", "flagged_words"]

        # Complex operations (language models, NLP)
        complex_patterns = [
            "perplexity",
            "language_id",
            "text_entity",
            "text_action",
            "spacy",
            "nlp",
            "dependency",
            "pos_tag",
        ]

        # Check patterns
        for pattern in simple_patterns:
            if pattern in op_name_lower:
                return "simple"

        for pattern in complex_patterns:
            if pattern in op_name_lower:
                return "complex"

        for pattern in medium_patterns:
            if pattern in op_name_lower:
                return "medium"

        # Default to medium if no patterns match
        return "medium"

    def _get_adaptive_complexity(self, op_name: str, performance_data: Optional[Dict] = None) -> str:
        """Get adaptive complexity based on performance data if available."""
        if performance_data and op_name in performance_data:
            # Use performance data to adjust complexity
            avg_time = performance_data[op_name].get("avg_time", 0)

            if avg_time < 0.001:  # Very fast
                return "simple"
            elif avg_time < 0.01:  # Fast
                return "medium"
            else:  # Slow
                return "complex"

        # Fall back to static analysis
        return self._get_operation_complexity(op_name)

    def _has_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if op1 depends on op2.

        Args:
            op1: First operation
            op2: Second operation

        Returns:
            True if op1 depends on op2
        """
        # Get operation configurations
        config1 = op1.config or {}
        config2 = op2.config or {}

        # 1. Check intermediate variables
        op1_vars = set(config1.get("inter_vars", []))
        op2_vars = set(config2.get("inter_vars", []))
        if op1_vars & op2_vars:
            return True

        # 2. Check stats dependencies
        if self._check_stats_dependencies(op1, op2):
            return True

        # 3. Check model dependencies
        if self._check_model_dependencies(op1, op2):
            return True

        # 4. Check data field dependencies
        if self._check_field_dependencies(op1, op2):
            return True

        # 5. Check operation-specific dependencies
        if self._check_operation_specific_dependencies(op1, op2):
            return True

        return False

    def _check_stats_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations depend on the same stats."""
        # Get stats keys that each operation produces/consumes
        op1_stats = self._get_stats_keys(op1)
        op2_stats = self._get_stats_keys(op2)

        # If they share any stats keys, they have a dependency
        return bool(op1_stats & op2_stats)

    def _get_stats_keys(self, op: OpNode) -> set:
        """Get stats keys that an operation produces or consumes."""
        # Map operation names to their stats keys
        stats_mapping = {
            "words_num_filter": {StatsKeys.num_words},
            "text_length_filter": {StatsKeys.text_len},
            "character_repetition_filter": {StatsKeys.char_rep_ratio},
            "word_repetition_filter": {StatsKeys.word_rep_ratio},
            "average_line_length_filter": {StatsKeys.avg_line_length},
            "maximum_line_length_filter": {StatsKeys.max_line_length},
            "alphanumeric_filter": {StatsKeys.alnum_ratio, StatsKeys.alpha_token_ratio},
            "special_characters_filter": {StatsKeys.special_char_ratio},
            "perplexity_filter": {StatsKeys.perplexity},
            "stopwords_filter": {StatsKeys.stopwords_ratio},
            "flagged_words_filter": {StatsKeys.flagged_words_ratio},
            "text_entity_dependency_filter": {StatsKeys.num_dependency_edges},
            "general_field_filter": {StatsKeys.general_field_filter_condition},
        }

        return stats_mapping.get(op.name, set())

    def _check_model_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations use the same models."""
        config1 = op1.config or {}
        config2 = op2.config or {}

        # Get model keys
        op1_models = set()
        op2_models = set()

        # Check for model keys in config
        for key in ["model_key", "sp_model_key", "kl_model_key"]:
            if key in config1:
                op1_models.add(config1[key])
            if key in config2:
                op2_models.add(config2[key])

        # If they share any models, they have a dependency
        return bool(op1_models & op2_models)

    def _check_field_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations process the same data fields."""
        config1 = op1.config or {}
        config2 = op2.config or {}

        # Get field keys
        op1_fields = set()
        op2_fields = set()

        # Check for field keys in config
        for key in ["text_key", "image_key", "audio_key", "video_key"]:
            if key in config1:
                op1_fields.add(config1[key])
            if key in config2:
                op2_fields.add(config2[key])

        # If they share any fields, they might have a dependency
        # (This is a conservative check - some operations can share fields safely)
        shared_fields = op1_fields & op2_fields

        # Only consider it a dependency if both operations are text processors
        # and they share text_key (indicating they process the same text)
        if shared_fields and "text_key" in shared_fields:
            return True

        return False

    def _check_operation_specific_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check operation-specific dependencies."""
        # Some operations have specific dependencies that can't be generalized

        # Example: Operations that modify the same data structure
        # This is a placeholder for future operation-specific checks
        return False

    def _smart_complex_filter_fusion(self, node: OpNode, group: List[OpNode]) -> bool:
        """Smart fusion decision for complex filters that may cause slowdown.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if fusion is recommended, False if it would cause slowdown
        """
        # Get complexity of current operation and group
        node_complexity = self._get_operation_complexity(node.name)
        group_complexities = [self._get_operation_complexity(op.name) for op in group]

        # Rule 1: Never fuse complex operations together (causes slowdown)
        if node_complexity == "complex" and any(c == "complex" for c in group_complexities):
            logger.debug(f"Rejecting fusion: complex operation {node.name} with complex group")
            return False

        # Rule 2: Limit group size for complex operations (max 2 complex filters per group)
        complex_count_in_group = sum(1 for c in group_complexities if c == "complex")
        if node_complexity == "complex" and complex_count_in_group >= 2:
            logger.debug(
                f"Rejecting fusion: complex operation {node.name} would exceed max 2 complex filters per group"
            )
            return False

        # Rule 3: Check for model conflicts
        if self._has_model_conflicts(node, group):
            logger.debug(f"Rejecting fusion: model conflicts detected for {node.name}")
            return False

        # Rule 4: Check memory requirements
        if self._would_exceed_memory_limit(node, group):
            logger.debug(f"Rejecting fusion: would exceed memory limit for {node.name}")
            return False

        # Rule 5: Check for sequential dependencies
        if self._has_sequential_dependencies(node, group):
            logger.debug(f"Rejecting fusion: sequential dependencies for {node.name}")
            return False

        return True

    def _has_model_conflicts(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if operations have conflicting model requirements."""
        # Get model requirements for current operation
        node_models = self._get_model_requirements(node)

        # Check against group models
        for op in group:
            group_models = self._get_model_requirements(op)
            # If both operations require different models of the same type
            for model_type in node_models:
                if model_type in group_models and node_models[model_type] != group_models[model_type]:
                    return True
        return False

    def _get_model_requirements(self, node: OpNode) -> Dict[str, str]:
        """Get model requirements for an operation."""
        models = {}
        config = node.config or {}

        # Check for common model keys
        for key in ["model_key", "sp_model_key", "kl_model_key"]:
            if key in config:
                models[key] = config[key]

        # Check for language-specific models
        if "lang" in config:
            models["lang"] = config["lang"]

        return models

    def _would_exceed_memory_limit(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if fusion would exceed memory limits."""
        # Estimate memory usage for current operation
        node_memory = self._estimate_operation_memory(node)

        # Estimate memory for group
        group_memory = sum(self._estimate_operation_memory(op) for op in group)

        # Total estimated memory
        total_memory = node_memory + group_memory

        # Conservative memory limit (2GB)
        memory_limit = 2 * 1024 * 1024 * 1024  # 2GB in bytes

        return total_memory > memory_limit

    def _estimate_operation_memory(self, node: OpNode) -> int:
        """Estimate memory usage for an operation in bytes."""
        complexity = self._get_operation_complexity(node.name)

        # Rough memory estimates based on complexity
        if complexity == "simple":
            return 50 * 1024 * 1024  # 50MB
        elif complexity == "medium":
            return 200 * 1024 * 1024  # 200MB
        else:  # complex
            return 500 * 1024 * 1024  # 500MB

    def _has_sequential_dependencies(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if operations must be executed sequentially."""
        # Check for data flow dependencies
        for op in group:
            if self._has_dependency(node, op) or self._has_dependency(op, node):
                return True

        # Check for operation-specific sequential requirements
        node_name = node.name.lower()
        group_names = [op.name.lower() for op in group]

        # Some operations must be sequential
        sequential_patterns = [
            ("perplexity", "language_id"),  # Language detection before perplexity
            ("text_entity", "text_action"),  # Entity detection before action analysis
        ]

        for pattern1, pattern2 in sequential_patterns:
            if pattern1 in node_name and any(pattern2 in name for name in group_names):
                return True
            if pattern2 in node_name and any(pattern1 in name for name in group_names):
                return True

        return False
