import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Aggregator
from data_juicer.utils.common_utils import (
    avg_split_string_list_under_limit,
    is_string_list,
)
from data_juicer.utils.constant import BatchMetaKeys, Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from .nested_aggregator import NestedAggregator

OP_NAME = "entity_attribute_aggregator"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class EntityAttributeAggregator(Aggregator):
    """Summarizes a given attribute of an entity from a set of documents.

    The operator extracts and summarizes the specified attribute of a given entity from the
    provided documents. It uses a system prompt, example prompt, and input template to
    generate the summary. The output is formatted as a markdown-style summary with the
    entity and attribute clearly labeled. The summary is limited to a specified number of
    words (default is 100). The operator uses a Hugging Face tokenizer to handle token
    limits and splits documents if necessary. If the input key or required fields are
    missing, the operator logs a warning and returns the sample unchanged. The summary is
    stored in the batch metadata under the specified output key. The system prompt, input
    template, example prompt, and output pattern can be customized."""

    DEFAULT_SYSTEM_TEMPLATE = (
        "给定与`{entity}`相关的一些文档，总结`{entity}`的`{attribute}`。\n"
        "要求：\n"
        "- 尽量使用原文专有名词\n"
        "- 联系上下文，自动忽略上下文不一致的细节错误\n"
        "- 只对文档中与`{entity}`的`{attribute}`有关的内容进行总结\n"
        "- 字数限制在**{word_limit}字以内**\n"
        "- 要求输出格式如下：\n"
        "# {entity}\n"
        "## {attribute}\n"
        "...\n"
        "{example}"
    )

    DEFAULT_EXAMPLE_PROMPT = (
        "- 例如，根据相关文档总结`孙悟空`的`出身背景`，**100字**以内的样例如下：\n"
        "`孙悟空`的`出身背景`总结：\n"
        "# 孙悟空\n"
        "## 出身背景\n"
        "号称齐天大圣，花果山水帘洞的美猴王、西行取经队伍中的大师兄。"
        "师父是唐僧玄奘，曾拜菩提祖师学艺。"
        "亲生父母未知，自石头中孕育而生。自认斗战胜佛，最怕观世音菩萨和紧箍咒。\n"
    )

    DEFAULT_INPUT_TEMPLATE = "`{entity}`的相关文档：\n" "{sub_docs}\n\n" "`{entity}`的`{attribute}`总结：\n"

    DEFAULT_OUTPUT_PATTERN_TEMPLATE = r"\#\s*{entity}\s*\#\#\s*{attribute}\s*(.*?)\Z"  # noqa: E501

    def __init__(
        self,
        api_model: str = "gpt-4o",
        entity: str = None,
        attribute: str = None,
        input_key: str = MetaKeys.event_description,
        output_key: str = BatchMetaKeys.entity_attribute,
        word_limit: PositiveInt = 100,
        max_token_num: Optional[PositiveInt] = None,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        example_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        output_pattern_template: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param entity: The given entity.
        :param attribute: The given attribute.
        :param input_key: The input key in the meta field of the samples.
            It is "event_description" in default.
        :param output_key: The output key in the aggregation field of the
            samples. It is "entity_attribute" in default.
        :param word_limit: Prompt the output length.
        :param max_token_num: The max token num of the total tokens of the
            sub documents. Without limitation if it is None.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt_template: The system prompt template.
        :param example_prompt: The example part in the system prompt.
        :param input_template: The input template.
        :param output_pattern_template: The output template.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        if entity is None or attribute is None:
            raise ValueError("The entity and attribute cannot be None!")

        self.entity = entity
        self.attribute = attribute
        self.input_key = input_key
        self.output_key = output_key
        self.word_limit = word_limit
        self.max_token_num = max_token_num

        system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_TEMPLATE
        self.example_prompt = example_prompt or self.DEFAULT_EXAMPLE_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        output_pattern_template = output_pattern_template or self.DEFAULT_OUTPUT_PATTERN_TEMPLATE
        self.system_prompt = system_prompt_template.format(
            entity=self.entity, attribute=self.attribute, word_limit=self.word_limit, example=self.example_prompt
        )
        self.output_pattern = output_pattern_template.format(entity=entity, attribute=attribute)

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            return_processor=True,
            **model_params,
        )

        self.try_num = try_num
        self.nested_sum = NestedAggregator(
            api_model=api_model,
            max_token_num=max_token_num,
            api_endpoint=api_endpoint,
            response_path=response_path,
            try_num=try_num,
            model_params=model_params,
            sampling_params=sampling_params,
        )

    def parse_output(self, response):
        pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(response)
        if matches:
            result = matches[0].strip()
        else:
            result = ""

        return result

    def attribute_summary(self, sub_docs, rank=None):
        if not sub_docs:
            return ""

        model, tokenizer = get_model(self.model_key, rank, self.use_cuda())
        token_nums = [len(tokenizer.encode(sub_doc)) for sub_doc in sub_docs]
        group_docs = avg_split_string_list_under_limit(sub_docs, token_nums, self.max_token_num)
        results = []
        for docs in group_docs:
            doc_str = "\n\n".join(docs)
            input_prompt = self.input_template.format(entity=self.entity, attribute=self.attribute, sub_docs=doc_str)
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]
            result = ""
            for i in range(self.try_num):
                try:
                    response = model(messages, **self.sampling_params)
                    result = self.parse_output(response)
                    if len(result) > 0:
                        break
                except Exception as e:
                    logger.warning(f"Exception: {e}")
            results.append(result)

        return self.nested_sum.recursive_summary(results)

    def process_single(self, sample=None, rank=None):
        if self.output_key in sample[Fields.batch_meta]:
            return sample

        if Fields.meta not in sample or self.input_key not in sample[Fields.meta][0]:
            logger.warning("The input key does not exist in the sample!")
            return sample

        sub_docs = [d[self.input_key] for d in sample[Fields.meta]]
        # if not batched sample
        if not is_string_list(sub_docs):
            logger.warning("Require string meta as input!")
            return sample

        sample[Fields.batch_meta][self.output_key] = self.attribute_summary(sub_docs, rank=rank)

        return sample
