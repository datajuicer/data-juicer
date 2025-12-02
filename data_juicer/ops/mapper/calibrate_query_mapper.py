from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.mapper.calibrate_qa_mapper import CalibrateQAMapper

OP_NAME = "calibrate_query_mapper"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class CalibrateQueryMapper(CalibrateQAMapper):
    """Calibrate query in question-answer pairs based on reference text.

    This operator adjusts the query (question) in a question-answer pair to be more detailed
    and accurate, while ensuring it can still be answered by the original answer. It uses a
    reference text to inform the calibration process. The calibration is guided by a system
    prompt, which instructs the model to refine the question without adding extraneous
    information. The output is parsed to extract the calibrated query, with any additional
    content removed."""

    DEFAULT_SYSTEM_PROMPT = "请根据提供的【参考信息】对问答对中的【问题】进行校准，\
        使其更加详细、准确，且仍可以由原答案回答。只输出校准后的问题，不要输出多余内容。"

    def parse_output(self, raw_output):
        return raw_output.strip(), None
