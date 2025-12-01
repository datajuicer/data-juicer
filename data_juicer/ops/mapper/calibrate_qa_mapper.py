import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "calibrate_qa_mapper"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class CalibrateQAMapper(Mapper):
    """Calibrates question-answer pairs based on reference text using an API model.

    This operator uses a specified API model to calibrate question-answer pairs, making them
    more detailed and accurate. It constructs the input prompt by combining the reference
    text and the question-answer pair, then sends it to the API for calibration. The output
    is parsed to extract the calibrated question and answer. The operator retries the API
    call and parsing up to a specified number of times in case of errors. The default system
    prompt, input templates, and output pattern can be customized. The operator supports
    additional parameters for model initialization and sampling."""

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = (
        "请根据提供的【参考信息】对【问题】和【回答】进行校准，使其更加详细、准确。\n"
        "按照以下格式输出：\n"
        "【问题】\n"
        "校准后的问题\n"
        "【回答】\n"
        "校准后的回答"
    )
    DEFAULT_INPUT_TEMPLATE = "{reference}\n{qa_pair}"
    DEFAULT_REFERENCE_TEMPLATE = "【参考信息】\n{}"
    DEFAULT_QA_PAIR_TEMPLATE = "【问题】\n{}\n【回答】\n{}"
    DEFAULT_OUTPUT_PATTERN = r"【问题】\s*(.*?)\s*【回答】\s*(.*)"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        reference_template: Optional[str] = None,
        qa_pair_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.

        :param api_model: API model name.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the calibration task.
        :param input_template: Template for building the model input.
        :param reference_template: Template for formatting the reference text.
        :param qa_pair_template: Template for formatting question-answer pairs.
        :param output_pattern: Regular expression for parsing model output.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.reference_template = reference_template or self.DEFAULT_REFERENCE_TEMPLATE
        self.qa_pair_template = qa_pair_template or self.DEFAULT_QA_PAIR_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.sampling_params = sampling_params

        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num

    def build_input(self, sample):
        reference = self.reference_template.format(sample[self.text_key])
        qa_pair = self.qa_pair_template.format(sample[self.query_key], sample[self.response_key])
        input_prompt = self.input_template.format(reference=reference, qa_pair=qa_pair)
        return input_prompt

    def parse_output(self, raw_output):
        match = re.match(self.output_pattern, raw_output)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        else:
            return None, None

    def process_single(self, sample, rank=None):
        client = get_model(self.model_key, rank=rank)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_input(sample)},
        ]
        parsed_q, parsed_a = None, None
        for _ in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                parsed_q, parsed_a = self.parse_output(output)
                if parsed_q or parsed_a:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")
        if parsed_q:
            sample[self.query_key] = parsed_q
        if parsed_a:
            sample[self.response_key] = parsed_a

        return sample
