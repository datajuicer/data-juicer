import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "relation_identity_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class RelationIdentityMapper(Mapper):
    """Identify the relation between two entities in a given text.

    This operator uses an API model to analyze the relationship between two specified
    entities in the text. It constructs a prompt with the provided system and input
    templates, then sends it to the API model for analysis. The output is parsed using a
    regular expression to extract the relationship. If the two entities are the same, the
    relationship is identified as "another identity." The result is stored in the meta field
    under the key 'role_relation' by default. The operator retries the API call up to a
    specified number of times in case of errors. If `drop_text` is set to True, the original
    text is removed from the sample after processing."""

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "给定关于{entity1}和{entity2}的文本信息。"
        "判断{entity1}和{entity2}之间的关系。\n"
        "要求：\n"
        "- 关系用一个或多个词语表示，必要时可以加一个形容词来描述这段关系\n"
        "- 输出关系时不要参杂任何标点符号\n"
        "- 需要你进行合理的推理才能得出结论\n"
        "- 如果两个人物身份是同一个人，输出关系为：另一个身份\n"
        "- 输出格式为：\n"
        "分析推理：...\n"
        "所以{entity2}是{entity1}的：...\n"
        "- 注意输出的是{entity2}是{entity1}的什么关系，而不是{entity1}是{entity2}的什么关系"
    )
    DEFAULT_INPUT_TEMPLATE = "关于{entity1}和{entity2}的文本信息：\n```\n{text}\n```\n"
    DEFAULT_OUTPUT_PATTERN_TEMPLATE = r"""
        \s*分析推理：\s*(.*?)\s*
        \s*所以{entity2}是{entity1}的：\s*(.*?)\Z
    """

    def __init__(
        self,
        api_model: str = "gpt-4o",
        source_entity: str = None,
        target_entity: str = None,
        *,
        output_key: str = MetaKeys.role_relation,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        input_template: Optional[str] = None,
        output_pattern_template: Optional[str] = None,
        try_num: PositiveInt = 3,
        drop_text: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param source_entity: The source entity of the relation to be
            identified.
        :param target_entity: The target entity of the relation to be
            identified.
        :param output_key: The output key in the meta field in the
            samples. It is 'role_relation' in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt_template: System prompt template for the task.
        :param input_template: Template for building the model input.
        :param output_pattern_template: Regular expression template for
            parsing model output.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        if source_entity is None or target_entity is None:
            logger.warning("source_entity and target_entity cannot be None")

        self.source_entity = source_entity
        self.target_entity = target_entity

        self.output_key = output_key

        system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
        self.system_prompt = system_prompt_template.format(entity1=source_entity, entity2=target_entity)
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        output_pattern_template = output_pattern_template or self.DEFAULT_OUTPUT_PATTERN_TEMPLATE
        self.output_pattern = output_pattern_template.format(entity1=source_entity, entity2=target_entity)

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output):
        pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)

        relation = ""

        for match in matches:
            _, relation = match
            relation = relation.strip()

        return relation

    def process_single(self, sample, rank=None):
        meta = sample[Fields.meta]
        if self.output_key in meta:
            return sample

        client = get_model(self.model_key, rank=rank)

        text = sample[self.text_key]
        input_prompt = self.input_template.format(entity1=self.source_entity, entity2=self.target_entity, text=text)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]
        relation = ""
        for i in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                relation = self.parse_output(output)
                if len(relation) > 0:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        meta[self.output_key] = relation

        if self.drop_text:
            sample.pop(self.text_key)

        return sample
