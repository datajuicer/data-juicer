import copy
import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..common import split_text_by_punctuation

OP_NAME = "extract_event_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractEventMapper(Mapper):
    """Extracts events and relevant characters from the text.

    This operator uses an API model to summarize the text into multiple events and extract
    the relevant characters for each event. The summary and character extraction follow a
    predefined format. The operator retries the API call up to a specified number of times
    if there is an error. The extracted events and characters are stored in the meta field
    of the samples. If no events are found, the original samples are returned. The operator
    can optionally drop the original text after processing."""

    _batched_op = True

    DEFAULT_SYSTEM_PROMPT = (
        "给定一段文本，对文本的情节进行分点总结，并抽取与情节相关的人物。\n"
        "要求：\n"
        "- 尽量不要遗漏内容，不要添加文本中没有的情节，符合原文事实\n"
        "- 联系上下文说明前因后果，但仍然需要符合事实\n"
        "- 不要包含主观看法\n"
        "- 注意要尽可能保留文本的专有名词\n"
        "- 注意相关人物需要在对应情节中出现\n"
        "- 只抽取情节中的主要人物，不要遗漏情节的主要人物\n"
        "- 总结格式如下：\n"
        "### 情节1：\n"
        "- **情节描述**： ...\n"
        "- **相关人物**：人物1，人物2，人物3，...\n"
        "### 情节2：\n"
        "- **情节描述**： ...\n"
        "- **相关人物**：人物1，人物2，...\n"
        "### 情节3：\n"
        "- **情节描述**： ...\n"
        "- **相关人物**：人物1，...\n"
        "...\n"
    )
    DEFAULT_INPUT_TEMPLATE = "# 文本\n```\n{text}\n```\n"
    DEFAULT_OUTPUT_PATTERN = r"""
        \#\#\#\s*情节(\d+)：\s*
        -\s*\*\*情节描述\*\*\s*：\s*(.*?)\s*
        -\s*\*\*相关人物\*\*\s*：\s*(.*?)(?=\#\#\#|\Z)
    """

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        event_desc_key: str = MetaKeys.event_description,
        relevant_char_key: str = MetaKeys.relevant_characters,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        drop_text: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param event_desc_key: The key name to store the event descriptions
            in the meta field. It's "event_description" in default.
        :param relevant_char_key: The field name to store the relevant
            characters to the events in the meta field. It's
            "relevant_characters" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param input_template: Template for building the model input.
        :param output_pattern: Regular expression for parsing model output.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.event_desc_key = event_desc_key
        self.relevant_char_key = relevant_char_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output):
        pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)

        event_list, character_list = [], []

        for match in matches:
            _, desc, chars = match
            chars = split_text_by_punctuation(chars)
            if len(chars) > 0:
                event_list.append(desc)
                character_list.append(chars)

        return event_list, character_list

    def _process_single_sample(self, text="", rank=None):
        client = get_model(self.model_key, rank=rank)

        input_prompt = self.input_template.format(text=text)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]

        event_list, character_list = [], []
        for _ in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                event_list, character_list = self.parse_output(output)
                if len(event_list) > 0:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        return event_list, character_list

    def process_batched(self, samples, rank=None):
        # check if it's generated already
        if self.event_desc_key in samples[Fields.meta][0] and self.relevant_char_key in samples[Fields.meta][0]:
            return samples

        events, characters = [], []
        for text in samples[self.text_key]:
            cur_events, cur_characters = self._process_single_sample(text, rank=rank)
            events.append(cur_events)
            characters.append(cur_characters)

        if self.drop_text:
            samples.pop(self.text_key)

        new_samples = []
        for i in range(len(events)):
            for event, character in zip(events[i], characters[i]):
                cur_sample = {key: copy.deepcopy(samples[key][i]) for key in samples}
                cur_sample[Fields.meta][self.event_desc_key] = event
                cur_sample[Fields.meta][self.relevant_char_key] = character
                new_samples.append(cur_sample)

        if len(new_samples) == 0:
            logger.warning("Extract Not event in the batch of samples!")
            return samples

        res_samples = {}
        keys = new_samples[0].keys()
        for key in keys:
            res_samples[key] = [s[key] for s in new_samples]

        return res_samples
