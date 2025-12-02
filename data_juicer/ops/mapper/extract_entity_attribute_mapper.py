import re
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "extract_entity_attribute_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractEntityAttributeMapper(Mapper):
    """Extracts attributes for given entities from the text and stores them in the sample's
    metadata.

    This operator uses an API model to extract specified attributes for given entities from
    the input text. It constructs prompts based on provided templates and parses the model's
    output to extract attribute descriptions and supporting text. The extracted data is
    stored in the sample's metadata under the specified keys. If the required metadata
    fields already exist, the operator skips processing for that sample. The operator
    retries the API call and parsing up to a specified number of times in case of errors.
    The default system prompt, input template, and parsing patterns are used if not
    provided."""

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "给定一段文本，从文本中总结{entity}的{attribute}，并且从原文摘录最能说明该{attribute}的代表性示例。\n"
        "要求：\n"
        "- 摘录的示例应该简短。\n"
        "- 遵循如下的回复格式：\n"
        "# {entity}\n"
        "## {attribute}：\n"
        "...\n"
        "### 代表性示例摘录1：\n"
        "```\n"
        "...\n"
        "```\n"
        "### 代表性示例摘录2：\n"
        "```\n"
        "...\n"
        "```\n"
        "...\n"
    )

    DEFAULT_INPUT_TEMPLATE = "# 文本\n```\n{text}\n```\n"
    DEFAULT_ATTR_PATTERN_TEMPLATE = r"\#\#\s*{attribute}：\s*(.*?)(?=\#\#\#|\Z)"
    DEFAULT_DEMON_PATTERN = r"\#\#\#\s*代表性示例摘录(\d+)：\s*```\s*(.*?)```\s*(?=\#\#\#|\Z)"  # noqa: E501

    def __init__(
        self,
        api_model: str = "gpt-4o",
        query_entities: List[str] = [],
        query_attributes: List[str] = [],
        *,
        entity_key: str = MetaKeys.main_entities,
        attribute_key: str = MetaKeys.attributes,
        attribute_desc_key: str = MetaKeys.attribute_descriptions,
        support_text_key: str = MetaKeys.attribute_support_texts,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        input_template: Optional[str] = None,
        attr_pattern_template: Optional[str] = None,
        demo_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        drop_text: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.

        :param api_model: API model name.
        :param query_entities: Entity list to be queried.
        :param query_attributes: Attribute list to be queried.
        :param entity_key: The key name in the meta field to store the
            given main entity for attribute extraction. It's "entity" in
            default.
        :param attribute_key: The key name in the meta field to
            store the given attribute to be extracted. It's "attribute"
            in default.
        :param attribute_desc_key: The key name in the meta field to store
            the extracted attribute description. It's
            "attribute_description" in default.
        :param support_text_key: The key name in the meta field to store
            the attribute support text extracted from the raw text.
            It's "support_text" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt_template: System prompt template for the
            task. Need to be specified by given entity and attribute.
        :param input_template: Template for building the model input.
        :param attr_pattern_template: Pattern for parsing the attribute from
            output. Need to be specified by given attribute.
        :param demo_pattern: Pattern for parsing the demonstration from
            output to support the attribute.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.query_entities = query_entities
        self.query_attributes = query_attributes

        self.entity_key = entity_key
        self.attribute_key = attribute_key
        self.attribute_desc_key = attribute_desc_key
        self.support_text_key = support_text_key

        self.system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.attr_pattern_template = attr_pattern_template or self.DEFAULT_ATTR_PATTERN_TEMPLATE
        self.demo_pattern = demo_pattern or self.DEFAULT_DEMON_PATTERN

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output, attribute_name):
        attribute_pattern = self.attr_pattern_template.format(attribute=attribute_name)
        pattern = re.compile(attribute_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)
        if matches:
            attribute = matches[0].strip()
        else:
            attribute = ""

        pattern = re.compile(self.demo_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)
        demos = [demo.strip() for _, demo in matches if demo.strip()]

        return attribute, demos

    def _process_single_text(self, text="", rank=None):
        client = get_model(self.model_key, rank=rank)

        entities, attributes, descs, demo_lists = [], [], [], []
        for entity in self.query_entities:
            for attribute in self.query_attributes:
                system_prompt = self.system_prompt_template.format(entity=entity, attribute=attribute)
                input_prompt = self.input_template.format(text=text)
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}]

                desc, demos = "", np.array([], dtype=str)
                for _ in range(self.try_num):
                    try:
                        output = client(messages, **self.sampling_params)
                        cur_desc, cur_demos = self.parse_output(output, attribute)
                        if cur_desc and len(cur_demos) > 0:
                            desc = cur_desc
                            demos = cur_demos
                            break
                    except Exception as e:
                        logger.warning(f"Exception: {e}")
                entities.append(entity)
                attributes.append(attribute)
                descs.append(desc)
                demo_lists.append(demos)

        return entities, attributes, descs, demo_lists

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if set([self.entity_key, self.attribute_key, self.attribute_desc_key, self.support_text_key]) <= set(
            sample[Fields.meta].keys()
        ):
            return sample

        res = self._process_single_text(sample[self.text_key], rank=rank)
        entities, attributes, descs, demo_lists = res

        if self.drop_text:
            sample.pop(self.text_key)

        sample[Fields.meta][self.entity_key] = entities
        sample[Fields.meta][self.attribute_key] = attributes
        sample[Fields.meta][self.attribute_desc_key] = descs
        sample[Fields.meta][self.support_text_key] = demo_lists

        return sample
