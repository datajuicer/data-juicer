import re
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "extract_nickname_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractNicknameMapper(Mapper):
    """Extracts nickname relationships in the text using a language model.

    This operator uses a language model to identify and extract nickname relationships from
    the input text. It follows specific instructions to ensure accurate extraction, such as
    identifying the speaker, the person being addressed, and the nickname used. The
    extracted relationships are stored in the meta field under the specified key. The
    operator uses a default system prompt, input template, and output pattern, but these can
    be customized. The results are parsed and validated to ensure they meet the required
    format. If the text already contains the nickname information, it is not processed
    again. The operator retries the API call a specified number of times if an error occurs."""

    DEFAULT_SYSTEM_PROMPT = (
        "给定你一段文本，你的任务是将人物之间的称呼方式（昵称）提取出来。\n"
        "要求：\n"
        "- 需要给出说话人对被称呼人的称呼，不要搞反了。\n"
        "- 相同的说话人和被称呼人最多给出一个最常用的称呼。\n"
        "- 请不要输出互相没有昵称的称呼方式。\n"
        "- 输出格式如下：\n"
        "```\n"
        "### 称呼方式1\n"
        "- **说话人**：...\n"
        "- **被称呼人**：...\n"
        "- **...对...的昵称**：...\n"
        "### 称呼方式2\n"
        "- **说话人**：...\n"
        "- **被称呼人**：...\n"
        "- **...对...的昵称**：...\n"
        "### 称呼方式3\n"
        "- **说话人**：...\n"
        "- **被称呼人**：...\n"
        "- **...对...的昵称**：...\n"
        "...\n"
        "```\n"
    )
    DEFAULT_INPUT_TEMPLATE = "# 文本\n```\n{text}\n```\n"
    DEFAULT_OUTPUT_PATTERN = r"""
        \#\#\#\s*称呼方式(\d+)\s*
        -\s*\*\*说话人\*\*\s*：\s*(.*?)\s*
        -\s*\*\*被称呼人\*\*\s*：\s*(.*?)\s*
        -\s*\*\*(.*?)对(.*?)的昵称\*\*\s*：\s*(.*?)(?=\#\#\#|\Z) # for double check
    """

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        nickname_key: str = MetaKeys.nickname,
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
        :param nickname_key: The key name to store the nickname
            relationship in the meta field. It's "nickname" in default.
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

        self.nickname_key = nickname_key

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

        nickname_relations = []

        for match in matches:
            _, role1, role2, role1_tmp, role2_tmp, nickname = match
            # for double check
            if role1.strip() != role1_tmp.strip() or role2.strip() != role2_tmp.strip():
                continue
            role1 = role1.strip()
            role2 = role2.strip()
            nickname = nickname.strip()
            # is name but not nickname
            if role2 == nickname:
                continue
            if role1 and role2 and nickname:
                nickname_relations.append((role1, role2, nickname))
        nickname_relations = list(set(nickname_relations))

        nickname_relations = [
            {
                MetaKeys.source_entity: nr[0],
                MetaKeys.target_entity: nr[1],
                MetaKeys.relation_description: nr[2],
                MetaKeys.relation_keywords: ["nickname"],
                MetaKeys.relation_strength: None,
            }
            for nr in nickname_relations
        ]

        return nickname_relations

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.nickname_key in sample[Fields.meta]:
            return sample

        client = get_model(self.model_key, rank=rank)

        input_prompt = self.input_template.format(text=sample[self.text_key])
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]
        nickname_relations = [
            {
                MetaKeys.source_entity: "",
                MetaKeys.target_entity: "",
                MetaKeys.relation_description: "",
                MetaKeys.relation_keywords: np.array([], dtype=str),
                MetaKeys.relation_strength: None,
            }
        ]
        for _ in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                results = self.parse_output(output)
                if len(results) > 0:
                    nickname_relations = results
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        sample[Fields.meta][self.nickname_key] = nickname_relations
        if self.drop_text:
            sample.pop(self.text_key)

        return sample
