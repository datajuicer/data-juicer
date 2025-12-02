import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "pair_preference_mapper"


# TODO: Extend LLM-based OPs into API-based implementation.
@OPERATORS.register_module(OP_NAME)
class PairPreferenceMapper(Mapper):
    """Mapper to construct paired preference samples by generating a rejected response and its
    reason.

    This operator uses an API model to generate a new response that is opposite in style,
    factuality, or stance to the original response. The generated response and the reason
    for its generation are stored in the sample. The default system prompt and input
    template are provided, but can be customized. The output is parsed using a regular
    expression to extract the new response and the reason. If parsing fails, the operator
    retries up to a specified number of times. The generated response and reason are stored
    in the sample under the keys 'rejected_response' and 'reason', respectively."""

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = (
        "你的任务是根据参考信息修改问答对中的回答，在语言风格、事实性、人物身份、立场等任一方面与原回答相反。"
        "必须按照以下标记格式输出，不要输出其他多余内容。\n"
        "【回答】\n"
        "生成的新回答\n"
        "【原因】\n"
        "生成该回答的原因"
    )
    DEFAULT_INPUT_TEMPLATE = (
        "【参考信息】\n" "{reference}\n" "\n" "以下是原始问答对：\n" "【问题】\n" "{query}\n" "【回答】\n" "{response}"
    )
    DEFAULT_OUTPUT_PATTERN = r".*?【回答】\s*(.*?)\s*【原因】\s*(.*)"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        rejected_key: str = "rejected_response",
        reason_key: str = "reason",
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
        :param system_prompt: System prompt for guiding the generation task.
        :param input_template: Template for building the model input. It must
            contain placeholders '{query}' and '{response}', and can optionally
            include '{reference}'.
        :param output_pattern: Regular expression for parsing model output.
        :param rejected_key: The field name in the sample to store the
            generated rejected response. Defaults to 'rejected_response'.
        :param reason_key: The field name in the sample to store the reason for
            generating the response. Defaults to 'reason'.
        :param try_num: The number of retries for the API call in case of
            response parsing failure. Defaults to 3.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.rejected_key = rejected_key
        self.reason_key = reason_key

        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )
        self.try_num = try_num
        self.sampling_params = sampling_params

    def build_input(self, sample):
        mapping = {
            "query": sample[self.query_key],
            "response": sample[self.response_key],
            "reference": sample.get(self.text_key, ""),
        }
        return self.input_template.format_map(mapping)

    def parse_output(self, raw_output):
        logger.debug(raw_output)
        match = re.match(self.output_pattern, raw_output, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        else:
            return ("", "")

    def process_single(self, sample, rank=None):
        client = get_model(self.model_key, rank=rank)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_input(sample)},
        ]

        parsed_rejected, parsed_reason = "", ""
        for _ in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                parsed_rejected, parsed_reason = self.parse_output(output)
                if parsed_rejected and parsed_reason:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")
        sample[self.rejected_key] = parsed_rejected
        sample[self.reason_key] = parsed_reason

        return sample
