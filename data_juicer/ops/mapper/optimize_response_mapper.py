from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.optimize_qa_mapper import OptimizeQAMapper

OP_NAME = "optimize_response_mapper"


@OPERATORS.register_module(OP_NAME)
class OptimizeResponseMapper(OptimizeQAMapper):
    """Optimize response in question-answer pairs to be more detailed and specific.

    This operator enhances the responses in question-answer pairs, making them more detailed
    and specific while ensuring they still address the original question. It uses a
    predefined system prompt for optimization. The optimized response is stripped of any
    leading or trailing whitespace before being returned. This mapper leverages a Hugging
    Face model for the optimization process, which is accelerated using CUDA."""

    DEFAULT_SYSTEM_PROMPT = (
        "请优化问答对中的回答，将其更加详细具体，但仍可以回答原问题。只输出优化后的回答，不要输出多余内容。"
    )

    _accelerator = "cuda"

    def parse_output(self, raw_output):
        return None, raw_output.strip()
