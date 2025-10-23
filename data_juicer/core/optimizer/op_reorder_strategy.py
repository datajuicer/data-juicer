#!/usr/bin/env python3
"""
Operation Reordering Strategy for the core optimizer.

This strategy analyzes dependencies between operations and reorders them
for optimal performance, prioritizing filters over mappers when possible.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Set

from loguru import logger

from ..pipeline_ast import OpNode, OpType, PipelineAST
from .strategy import OptimizationStrategy, register_strategy


@register_strategy("op_reorder")
class OpReorderStrategy(OptimizationStrategy):
    """
    Strategy that reorders operations based on dependency analysis and performance optimization.

    Key features:
    1. Analyzes dependencies between operations
    2. Performs topological sorting
    3. Prioritizes filters over mappers when possible
    4. Optimizes for early filtering to reduce data volume
    """

    def __init__(self):
        """Initialize the operation reordering strategy."""
        super().__init__(name="op_reorder")

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """
        Apply operation reordering to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST with reordered operations
        """
        if not ast.root or not ast.root.children:
            return ast

        logger.info("🔄 Applying operation reordering optimization...")

        # Get all operations from the AST
        operations = self._extract_operations(ast.root)

        # Log original order
        logger.info("📋 Original operation order:")
        for i, op in enumerate(operations, 1):
            op_type = "🔧 MAPPER" if op.op_type == OpType.MAPPER else "🔍 FILTER"
            logger.info(f"  {i}. {op_type}: {op.name}")

        # Analyze dependencies
        dependencies = self._analyze_dependencies(operations)
        logger.info(f"🔗 Found {len(dependencies)} dependencies between operations")

        # Log specific dependencies
        if dependencies:
            logger.info("🔗 Operation dependencies:")
            for op_name, deps in dependencies.items():
                if deps:
                    logger.info(f"  {op_name} depends on: {', '.join(deps)}")
        else:
            logger.info("🔗 No dependencies found - operations can be freely reordered")

        # Perform topological sort
        optimal_order = self._topological_sort(operations, dependencies)

        # Create a mapping from operation names to operation nodes
        op_map = {op.name: op for op in operations}

        # Log optimized order
        logger.info("⚡ Optimized operation order:")
        for i, op_name in enumerate(optimal_order, 1):
            op_node = op_map.get(op_name)
            if op_node:
                op_type = "🔧 MAPPER" if op_node.op_type == OpType.MAPPER else "🔍 FILTER"
                logger.info(f"  {i}. {op_type}: {op_name}")

        # Reorder operations in the AST
        new_ast = self._reorder_ast(ast, optimal_order)

        logger.info(f"✅ Reordered {len(operations)} operations for optimal performance")
        return new_ast

    def _extract_operations(self, root: OpNode) -> List[OpNode]:
        """Extract all operations from the AST."""
        operations = []

        def collect_ops(node: OpNode):
            if node.op_type in [OpType.MAPPER, OpType.FILTER]:
                operations.append(node)
            for child in node.children:
                collect_ops(child)

        collect_ops(root)
        return operations

    def _analyze_dependencies(self, operations: List[OpNode]) -> Dict[str, Set[str]]:
        """
        Analyze dependencies between operations.

        Args:
            operations: List of operation nodes

        Returns:
            Dictionary mapping operation names to their dependencies
        """
        dependency_graph = defaultdict(set)

        for i, op1 in enumerate(operations):
            op1_name = op1.name
            op1_vars = set(op1.config.get("inter_vars", []))
            op1_fields = set(op1.config.get("fields", []))

            for j, op2 in enumerate(operations):
                if i == j:
                    continue

                op2_name = op2.name
                op2_vars = set(op2.config.get("inter_vars", []))
                op2_fields = set(op2.config.get("fields", []))

                # Check for variable dependencies
                if op1_vars & op2_vars:
                    dependency_graph[op2_name].add(op1_name)

                # Check for field dependencies
                if op1_fields & op2_fields:
                    dependency_graph[op2_name].add(op1_name)

                # Check for operation-specific dependencies
                if self._has_operation_dependency(op1, op2):
                    dependency_graph[op2_name].add(op1_name)

        return dict(dependency_graph)

    def _has_operation_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if op2 depends on op1 based on operation types."""
        op1_name = op1.name.lower()
        op2_name = op2.name.lower()

        # Language detection before perplexity
        if "language_id" in op1_name and "perplexity" in op2_name:
            return True
        if "perplexity" in op1_name and "language_id" in op2_name:
            return True

        # Entity detection before action analysis
        if "text_entity" in op1_name and "text_action" in op2_name:
            return True
        if "text_action" in op1_name and "text_entity" in op2_name:
            return True

        # Image preprocessing before analysis
        if "image_resize" in op1_name and "image_quality" in op2_name:
            return True
        if "image_quality" in op1_name and "image_resize" in op2_name:
            return True

        return False

    def _topological_sort(self, operations: List[OpNode], dependencies: Dict[str, Set[str]]) -> List[str]:
        """
        Perform topological sorting of operations.

        Args:
            operations: List of operation nodes
            dependencies: Dependency graph

        Returns:
            List of operation names in optimal order
        """
        # Build in-degree count
        in_degree = defaultdict(int)
        all_ops = {op.name: op for op in operations}

        for op_name in all_ops:
            in_degree[op_name] = 0

        for op_name, deps in dependencies.items():
            for dep in deps:
                in_degree[op_name] += 1

        # Initialize queue with operations that have no dependencies
        queue = deque([op for op in all_ops if in_degree[op] == 0])
        result = []

        while queue:
            # Sort queue to prioritize filters over mappers
            queue = deque(sorted(queue, key=lambda op: self._get_operation_priority(all_ops[op])))

            current = queue.popleft()
            result.append(current)

            # Update in-degrees of dependent operations
            for op_name, deps in dependencies.items():
                if current in deps:
                    in_degree[op_name] -= 1
                    if in_degree[op_name] == 0:
                        queue.append(op_name)

        return result

    def _get_operation_priority(self, operation: OpNode) -> int:
        """Get priority for operation ordering (lower = higher priority)."""
        op_name = operation.name.lower()
        op_type = operation.op_type

        # Filters have highest priority (lowest number)
        if op_type == OpType.FILTER:
            return 1

        # Text filters before other filters
        if "text" in op_name and op_type == OpType.FILTER:
            return 2

        # Image filters
        if "image" in op_name and op_type == OpType.FILTER:
            return 3

        # Audio filters
        if "audio" in op_name and op_type == OpType.FILTER:
            return 4

        # Video filters
        if "video" in op_name and op_type == OpType.FILTER:
            return 5

        # Mappers have lower priority
        if op_type == OpType.MAPPER:
            return 10

        # Other operations
        return 20

    def _reorder_ast(self, ast: PipelineAST, optimal_order: List[str]) -> PipelineAST:
        """
        Reorder operations in the AST according to optimal order.

        Args:
            ast: Original pipeline AST
            optimal_order: List of operation names in optimal order

        Returns:
            New AST with reordered operations
        """
        # Create new AST
        new_ast = PipelineAST()
        new_ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        # Get all operations in optimal order
        operations = self._extract_operations(ast.root)
        op_dict = {op.name: op for op in operations}

        # Build new chain in optimal order
        current = new_ast.root
        for op_name in optimal_order:
            if op_name in op_dict:
                # Create a copy of the operation
                op = op_dict[op_name]
                new_op = OpNode(name=op.name, op_type=op.op_type, config=op.config.copy())
                current.children = [new_op]
                new_op.parent = current
                current = new_op

        return new_ast

    def get_reorder_benefits(self, operations: List[OpNode]) -> Dict[str, Any]:
        """
        Analyze the potential benefits of reordering operations.

        Args:
            operations: List of operation nodes

        Returns:
            Dictionary with reordering benefits analysis
        """
        # Count filters vs mappers
        filter_count = sum(1 for op in operations if op.op_type == OpType.FILTER)
        mapper_count = sum(1 for op in operations if op.op_type == OpType.MAPPER)

        # Analyze potential early filtering
        early_filter_ops = []
        for i, op in enumerate(operations):
            if op.op_type == OpType.FILTER and i > 0:
                early_filter_ops.append(op.name)

        return {
            "total_operations": len(operations),
            "filter_count": filter_count,
            "mapper_count": mapper_count,
            "early_filter_opportunities": len(early_filter_ops),
            "early_filter_ops": early_filter_ops,
            "potential_speedup": "High" if len(early_filter_ops) > 0 else "Medium",
            "memory_reduction": "High" if filter_count > mapper_count else "Medium",
        }
