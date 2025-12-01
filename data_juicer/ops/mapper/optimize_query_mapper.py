from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.optimize_qa_mapper import OptimizeQAMapper

OP_NAME = "optimize_query_mapper"


@OPERATORS.register_module(OP_NAME)
class OptimizeQueryMapper(OptimizeQAMapper):
    """Optimize queries in question-answer pairs to make them more specific and detailed.

    This mapper refines the questions in a QA pair, making them more specific and detailed
    while ensuring that the original answer can still address the optimized question. It
    uses a predefined system prompt for the optimization process. The optimized query is
    extracted from the raw output by stripping any leading or trailing whitespace. The
    mapper utilizes a CUDA accelerator for faster processing."""

    DEFAULT_SYSTEM_PROMPT = "优化问答对中的【问题】，将其更加详细具体，但仍可以由原答案回答。只输出优化后的【问题】，不要输出多余内容。"  # noqa: E501

    _accelerator = "cuda"

    def parse_output(self, raw_output):
        return raw_output.strip(), None
