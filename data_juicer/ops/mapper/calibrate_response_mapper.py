from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.calibrate_qa_mapper import CalibrateQAMapper

OP_NAME = "calibrate_response_mapper"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class CalibrateResponseMapper(CalibrateQAMapper):
    """Calibrate response in question-answer pairs based on reference text.

    This mapper calibrates the 'response' part of a question-answer pair by using a
    reference text. It aims to make the response more detailed and accurate while ensuring
    it still answers the original question. The calibration process uses a default system
    prompt, which can be customized. The output is stripped of any leading or trailing
    whitespace."""

    # noqa: E501

    DEFAULT_SYSTEM_PROMPT = "请根据提供的【参考信息】对问答对中的【回答】进行校准，\
        使其更加详细、准确，且仍可以回答原问题。只输出校准后的回答，不要输出多余内容。"

    def parse_output(self, raw_output):
        return None, raw_output.strip()
