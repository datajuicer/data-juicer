from typing import List

from loguru import logger

from data_juicer.core.optimizer.strategy import OptimizationStrategy, register_strategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST


@register_strategy("mapper_fusion")
class MapperFusionStrategy(OptimizationStrategy):
    """Strategy for fusing mapper operations in the pipeline."""

    def __init__(self):
        """Initialize the mapper fusion strategy."""
        super().__init__(name="mapper_fusion")

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply mapper fusion to the pipeline AST.

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
            # Group mapper operations
            mapper_groups = self._group_mappers(chain)

            for group in mapper_groups:
                if len(group) > 1:
                    # Create fused operation with clean naming
                    fused_name = "fused_mapper"
                    detailed_ops = [n.name for n in group]
                    logger.info(f"Fusing mapper operations into {fused_name}: {detailed_ops}")

                    # Create fused node using FusedMapper
                    fused_node = OpNode(
                        name=fused_name,
                        op_type=OpType.MAPPER,
                        config={
                            "fused_mapper": {
                                "name": fused_name,
                                "fused_mappers": detailed_ops,
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

    def _group_mappers(self, chain: List[OpNode]) -> List[List[OpNode]]:
        """Group mapper operations that can be fused together.

        Args:
            chain: List of operations in the pipeline

        Returns:
            List of mapper operation groups
        """
        groups = []
        current_group = []

        logger.info(f"Grouping mappers from chain: {[n.name for n in chain]}")

        for node in chain:
            if not PipelineAST.is_mapper_op(node):
                # If we encounter a non-mapper, finalize current group
                if current_group:
                    logger.info(f"Finalizing mapper group: {[n.name for n in current_group]}")
                    groups.append(current_group)
                    current_group = []
                # Add the non-mapper node as a separate group
                groups.append([node])
            else:
                # This is a mapper node
                if not current_group:
                    # Start a new group
                    current_group = [node]
                    logger.info(f"Starting new mapper group with: {node.name}")
                else:
                    # Check if current mapper can be fused with the group
                    if self._can_fuse_with_group(node, current_group):
                        current_group.append(node)
                        logger.info(f"Added {node.name} to current group: {[n.name for n in current_group]}")
                    else:
                        # Finalize current group and start a new one
                        logger.info(f"Finalizing mapper group due to dependency: {[n.name for n in current_group]}")
                        groups.append(current_group)
                        current_group = [node]
                        logger.info(f"Starting new mapper group with: {node.name}")

        # Don't forget the last group
        if current_group:
            logger.info(f"Finalizing final mapper group: {[n.name for n in current_group]}")
            groups.append(current_group)

        logger.info(f"Final mapper groups: {[[n.name for n in group] for group in groups]}")
        return groups

    def _can_fuse_with_group(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if a mapper can be fused with a group.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if the operation can be fused with the group
        """
        # Check dependencies
        for op in group:
            if self._has_dependency(node, op) or self._has_dependency(op, node):
                logger.info(f"Cannot fuse {node.name} with group {[n.name for n in group]} due to dependency")
                return False

        logger.info(f"Can fuse {node.name} with group {[n.name for n in group]}")
        return True

    def _has_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if op1 depends on op2.

        Args:
            op1: First operation
            op2: Second operation

        Returns:
            True if op1 depends on op2
        """
        # 1. Check intermediate variables (for mappers that produce/consume inter_vars)
        op1_vars = set(op1.config.get("inter_vars", []))
        op2_vars = set(op2.config.get("inter_vars", []))
        if op1_vars & op2_vars:
            logger.info(f"Dependency found via inter_vars: {op1.name} <-> {op2.name}")
            return True

        # 2. Check field dependencies (mappers that modify the same fields)
        if self._check_field_dependencies(op1, op2):
            logger.info(f"Dependency found via field dependencies: {op1.name} <-> {op2.name}")
            return True

        # 3. Check operation-specific dependencies
        if self._check_operation_specific_dependencies(op1, op2):
            logger.info(f"Dependency found via operation-specific dependencies: {op1.name} <-> {op2.name}")
            return True

        return False

    def _check_field_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations modify the same fields."""
        # For mappers, we allow fusion even if they modify the same fields
        # since they can be executed sequentially
        # Only prevent fusion for specific logical dependencies
        return False

    def _get_modified_fields(self, op: OpNode) -> set:
        """Get the fields that an operation modifies."""
        # This is a simplified mapping - in practice, you'd want to analyze the actual operation logic
        field_mapping = {
            "clean_email_mapper": {"text"},
            "clean_links_mapper": {"text"},
            "fix_unicode_mapper": {"text"},
            "punctuation_normalization_mapper": {"text"},
            "whitespace_normalization_mapper": {"text"},
            "text_lowercase_mapper": {"text"},
            "text_uppercase_mapper": {"text"},
            "remove_words_mapper": {"text"},
            "remove_characters_mapper": {"text"},
            "replace_words_mapper": {"text"},
            "replace_characters_mapper": {"text"},
            "split_text_mapper": {"text"},
            "join_text_mapper": {"text"},
            "text_length_mapper": {"text"},
            "text_quality_mapper": {"text"},
        }

        return field_mapping.get(op.name, set())

    def _check_operation_specific_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check operation-specific dependencies."""
        # Define specific dependencies that prevent fusion

        # Unicode fixing should come before punctuation normalization
        if op1.name == "punctuation_normalization_mapper" and op2.name == "fix_unicode_mapper":
            return True

        if op1.name == "fix_unicode_mapper" and op2.name == "punctuation_normalization_mapper":
            return True

        # Email/links cleaning should come before punctuation normalization
        if op1.name == "punctuation_normalization_mapper" and op2.name in ["clean_email_mapper", "clean_links_mapper"]:
            return True

        if op1.name in ["clean_email_mapper", "clean_links_mapper"] and op2.name == "punctuation_normalization_mapper":
            return True

        # Whitespace normalization should come after most other text operations
        if op1.name == "whitespace_normalization_mapper" and op2.name in [
            "clean_email_mapper",
            "clean_links_mapper",
            "fix_unicode_mapper",
            "punctuation_normalization_mapper",
        ]:
            return True

        if (
            op1.name
            in ["clean_email_mapper", "clean_links_mapper", "fix_unicode_mapper", "punctuation_normalization_mapper"]
            and op2.name == "whitespace_normalization_mapper"
        ):
            return True

        return False
