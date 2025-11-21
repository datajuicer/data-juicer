# standard library imports
import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# third party imports
import yaml


class OpType(Enum):
    """Types of operations in the pipeline."""

    ROOT = "root"
    MAPPER = "mapper"
    FILTER = "filter"
    DEDUPLICATOR = "deduplicator"
    SELECTOR = "selector"
    GROUPER = "grouper"
    AGGREGATOR = "aggregator"


# Operation dependencies and optimization rules
# This defines which operation types can depend on which other operation types
OP_TYPE_DEPENDENCIES = {
    OpType.FILTER: {OpType.MAPPER},  # Filters can depend on mappers
    OpType.DEDUPLICATOR: {OpType.MAPPER, OpType.FILTER},  # Deduplicators can depend on mappers and filters
    OpType.SELECTOR: {
        OpType.MAPPER,
        OpType.FILTER,
        OpType.DEDUPLICATOR,
    },  # Selectors can depend on all previous ops
    OpType.GROUPER: {
        OpType.MAPPER,
        OpType.FILTER,
        OpType.DEDUPLICATOR,
        OpType.SELECTOR,
    },  # Groupers can depend on all previous ops
    OpType.AGGREGATOR: {OpType.GROUPER},  # Aggregators can only depend on groupers
}


@dataclass
class OpNode:
    """Node in the pipeline AST representing an operation."""

    name: str
    op_type: OpType
    config: Dict[str, Any]
    children: List["OpNode"] = None
    parent: Optional["OpNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: "OpNode"):
        """Add a child node to this operation."""
        child.parent = self
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation."""
        return {
            "name": self.name,
            "type": self.op_type.value,
            "config": self.config,
            "children": [child.to_dict() for child in self.children],
        }


class PipelineAST:
    """Abstract Syntax Tree for a Data-Juicer pipeline."""

    def __init__(self):
        self.root = None
        self._op_type_map = {
            "mapper": OpType.MAPPER,
            "filter": OpType.FILTER,
            "deduplicator": OpType.DEDUPLICATOR,
            "selector": OpType.SELECTOR,
            "grouper": OpType.GROUPER,
            "aggregator": OpType.AGGREGATOR,
        }

        # Operation dependencies and optimization rules
        # Use the module-level constant to avoid duplication
        self._op_dependencies = OP_TYPE_DEPENDENCIES

    def _get_op_type(self, op_name: str) -> OpType:
        """Determine the operation type from its name."""
        for suffix, op_type in self._op_type_map.items():
            if op_name.endswith(f"_{suffix}"):
                return op_type
        return OpType.MAPPER  # Default to mapper if type cannot be determined

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the AST from a configuration dictionary."""
        if "process" not in config:
            raise ValueError("Configuration must contain a 'process' field")

        process_list = config["process"]
        if not process_list:
            return

        # Create root node
        self.root = OpNode(name="root", op_type=OpType.ROOT, config={})  # Root is a special type

        # Build tree following the order in process_list
        current_node = self.root
        for op_config in process_list:
            op_name, op_args = list(op_config.items())[0]
            op_type = self._get_op_type(op_name)

            new_node = OpNode(name=op_name, op_type=op_type, config=op_args)
            current_node.add_child(new_node)
            current_node = new_node

    def build_from_yaml(self, yaml_path: str) -> None:
        """Build the AST from a YAML configuration file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        self.build_from_config(config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the AST to a dictionary representation."""
        if not self.root:
            return {}
        return self.root.to_dict()

    def visualize(self) -> str:
        """Generate a string representation of the AST for visualization."""
        if not self.root:
            return "Empty pipeline"

        def _visualize_node(node: OpNode, level: int = 0, is_last: bool = True) -> str:
            indent = "  " * level
            prefix = "└── " if is_last else "├── "

            # Check if this is a fused operation and get detailed ops
            detailed_ops = None
            if node.name == "fused_mapper" and "fused_mapper" in node.config:
                detailed_ops = node.config["fused_mapper"].get("detailed_ops", [])
            elif node.name == "fused_filter" and "general_fused_op" in node.config:
                detailed_ops = node.config["general_fused_op"].get("detailed_ops", [])

            # Format the node name with detailed operations if available
            if detailed_ops:
                ops_str = ", ".join(detailed_ops)
                result = f"{indent}{prefix}{node.name} ({node.op_type.value}) [{ops_str}]\n"
            else:
                result = f"{indent}{prefix}{node.name} ({node.op_type.value})\n"

            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                result += _visualize_node(child, level + 1, is_last_child)
            return result

        return "Pipeline:\n" + _visualize_node(self.root, 0, True)

    @staticmethod
    def is_mapper_op(node_or_type) -> bool:
        """Check if node or op_type is a mapper operation using value comparison."""
        if hasattr(node_or_type, "op_type"):
            return getattr(node_or_type, "op_type").value == "mapper"
        return node_or_type.value == "mapper"

    @staticmethod
    def is_filter_op(node_or_type) -> bool:
        """Check if node or op_type is a filter operation using value comparison."""
        if hasattr(node_or_type, "op_type"):
            return getattr(node_or_type, "op_type").value == "filter"
        return node_or_type.value == "filter"

    @staticmethod
    def op_type_equals(a, b) -> bool:
        """Compare OpType values safely to handle module import issues."""
        return getattr(a, "value", a) == getattr(b, "value", b)


if __name__ == "__main__":
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Build and visualize pipeline AST from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_juicer_recipes/pile-philpaper-refine.yaml",
        help="Path to the pipeline configuration file (YAML)",
    )
    parser.add_argument(
        "--probe-results", type=str, help="Path to probe results file (YAML) containing operation speeds"
    )
    parser.add_argument("--optimize", action="store_true", help="Apply optimization strategies to the pipeline")

    args = parser.parse_args()

    # Get absolute path to config file
    config_path = os.path.abspath(args.config)
    print(f"Using config file: {config_path}")

    # Load and process config
    config = yaml.safe_load(open(config_path, "r"))

    # Build initial AST
    ast = PipelineAST()
    ast.build_from_config(config)
    print("\nOriginal Pipeline:")
    print(ast.visualize())
