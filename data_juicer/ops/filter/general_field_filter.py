import ast
from typing import Any, Dict

from ...utils.constant import Fields, StatsKeys
from ..base_op import OPERATORS, Filter

OP_NAME = "general_field_filter"


@OPERATORS.register_module(OP_NAME)
class GeneralFieldFilter(Filter):
    """Filter to keep samples based on a general field filter condition.

    The filter condition is a string that can include logical operators (and/or) and chain
    comparisons. For example: "10 < num <= 30 and text != 'nothing here' and __dj__meta__.a
    == 3". The condition is evaluated for each sample, and only samples that meet the
    condition are kept. The result of the filter condition is stored in the sample's stats
    under the key 'general_field_filter_condition'. If the filter condition is empty or
    already computed, the sample is not re-evaluated."""

    def __init__(self, filter_condition: str = "", *args, **kwargs):
        """
        Initialization method.
        :param filter_condition: The filter condition as a string.
            It can include logical operators (and/or) and chain comparisons.
            For example: "10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3".
        """
        super().__init__(*args, **kwargs)
        self.filter_condition = filter_condition.strip()
        self.ast_tree: ast.Expression = None
        if filter_condition:
            try:
                self.ast_tree = ast.parse(filter_condition, mode="eval")
            except SyntaxError as e:
                raise ValueError(f"Invalid filter condition: {filter_condition}") from e

    def compute_stats_single(self, sample, context=False):
        if (
            not self.filter_condition
            or self.filter_condition == ""
            or StatsKeys.general_field_filter_condition in sample[Fields.stats]
        ):
            return sample

        transformer = ExpressionTransformer(sample)
        result = transformer.transform(self.ast_tree)

        sample[Fields.stats][StatsKeys.general_field_filter_condition] = result
        return sample

    def process_single(self, sample: Dict) -> bool:
        return sample.get(Fields.stats, {}).get(StatsKeys.general_field_filter_condition, True)


class ExpressionTransformer(ast.NodeVisitor):
    _COMPARE_OPERATORS = {
        ast.Gt: lambda left_operand, right_operand: left_operand > right_operand,
        ast.Lt: lambda left_operand, right_operand: left_operand < right_operand,
        ast.Eq: lambda left_operand, right_operand: left_operand == right_operand,
        ast.NotEq: lambda left_operand, right_operand: left_operand != right_operand,
        ast.GtE: lambda left_operand, right_operand: left_operand >= right_operand,
        ast.LtE: lambda left_operand, right_operand: left_operand <= right_operand,
    }

    def __init__(self, sample: Dict):
        self.sample = sample

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        values = (self.visit(child) for child in node.values)
        if isinstance(node.op, ast.And):
            return all(values)
        elif isinstance(node.op, ast.Or):
            return any(values)
        raise ValueError(f"Unsupported logical operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        comparators = [self.visit(c) for c in node.comparators]
        ops = node.ops

        result = True
        for i in range(len(ops)):
            op = ops[i]
            right = comparators[i]
            if left is None or right is None:
                return False
            if not self._apply_op(op, left, right):
                result = False
                break
            left = right
        return result

    def _apply_op(self, op: ast.AST, left: Any, right: Any) -> bool:
        op_type = type(op)
        if op_type in self._COMPARE_OPERATORS:
            return self._COMPARE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")

    def visit_Name(self, node: ast.Name) -> Any:
        if "." in node.id:
            prefix, key = node.id.split(".", 1)
            if prefix == Fields.stats:
                return self.sample.get(Fields.stats, {}).get(key)
            if prefix == Fields.meta:
                return self.sample.get(Fields.meta, {}).get(key)
            raise ValueError(f"Unsupported prefix: {prefix}")
        return self.sample.get(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        if isinstance(value, dict):
            return value.get(node.attr)
        elif hasattr(value, node.attr):
            return getattr(value, node.attr)
        return None

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"Unsupported node type: {type(node).__name__}, details: {ast.dump(node)}")

    def transform(self, ast_tree: ast.Expression) -> bool:
        return self.visit(ast_tree.body)
