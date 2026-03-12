#!/usr/bin/env python3
"""
Mapper Fusion Strategy for the core optimizer.

This strategy fuses consecutive mapper operations to reduce dataset iteration overhead.
"""

from typing import List

from loguru import logger

from data_juicer.core.optimizer.strategy import OptimizationStrategy, register_strategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST
from data_juicer.utils.ray_utils import is_ray_mode


@register_strategy("mapper_fusion")
class MapperFusionStrategy(OptimizationStrategy):
    """Strategy for fusing mapper operations in the pipeline.

    Benefits of mapper fusion:
    - Reduced dataset iteration overhead (1 pass instead of N passes)
    - Better CPU cache locality (data stays hot between mappers)
    - Reduced multiprocessing setup overhead
    - For GPU mappers: data stays on GPU between operations
    """

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

        # Skip mapper fusion in Ray mode - Ray already parallelizes mappers efficiently
        if is_ray_mode():
            logger.info("Skipping mapper fusion in Ray mode - Ray already parallelizes mappers efficiently")
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
                if len(group) > 1 and all(PipelineAST.is_mapper_op(n) for n in group):
                    # Create fused operation
                    fused_name = "fused_mapper"
                    detailed_ops = [n.name for n in group]
                    logger.info(f"Fusing mapper operations: {detailed_ops}")

                    # Create operation configs (preserve original configs)
                    op_configs = []
                    for op in group:
                        op_config = {op.name: op.config or {}}
                        op_configs.append(op_config)

                    # Create fused node with config that optimization_manager can use
                    fused_node = OpNode(
                        name=fused_name,
                        op_type=OpType.MAPPER,
                        config={
                            "general_fused_op": {
                                "fused_op_list": op_configs,
                                "detailed_ops": detailed_ops,
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
        """Get unique chains of operations from the tree."""
        chains = []
        seen_chains = set()

        def traverse(current: OpNode, chain: List[OpNode]):
            if not current.children:
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
        """Group consecutive mapper operations that can be fused.

        Args:
            chain: List of operations in the pipeline

        Returns:
            List of operation groups (mappers grouped, others as singles)
        """
        groups = []
        current_group = []

        for node in chain:
            if not PipelineAST.is_mapper_op(node):
                # Non-mapper: finalize current mapper group
                if current_group:
                    groups.append(current_group)
                    current_group = []
                # Add non-mapper as a separate group
                groups.append([node])
            else:
                # Mapper: try to add to current group
                if not current_group:
                    current_group = [node]
                elif self._can_fuse_with_group(node, current_group):
                    current_group.append(node)
                else:
                    # Can't fuse, start new group
                    groups.append(current_group)
                    current_group = [node]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _can_fuse_with_group(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if a mapper can be fused with a group.

        Mappers can be fused if:
        1. No inter-variable dependencies
        2. No operation-specific ordering requirements

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if the operation can be fused
        """
        for op in group:
            if self._has_dependency(node, op):
                return False
        return True

    def _has_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if op1 depends on op2's output.

        Args:
            op1: First operation
            op2: Second operation

        Returns:
            True if there's a dependency
        """
        config1 = op1.config or {}
        config2 = op2.config or {}

        # Check intermediate variables
        op1_vars = set(config1.get("inter_vars", []))
        op2_vars = set(config2.get("inter_vars", []))
        if op1_vars & op2_vars:
            return True

        # Check operation-specific dependencies
        return self._has_operation_specific_dependency(op1, op2)

    def _has_operation_specific_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check operation-specific dependencies that prevent fusion."""
        op1_name = op1.name.lower()
        op2_name = op2.name.lower()

        # Unicode fixing should come before other text processing
        if "fix_unicode" in op2_name and any(p in op1_name for p in ["punctuation", "whitespace", "clean"]):
            return True

        # Whitespace normalization should come after content cleaning
        if "whitespace" in op1_name and any(p in op2_name for p in ["clean_email", "clean_links", "clean_html"]):
            return True

        return False
