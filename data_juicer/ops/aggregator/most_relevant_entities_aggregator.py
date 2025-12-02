import re
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Aggregator
from data_juicer.utils.common_utils import is_string_list
from data_juicer.utils.constant import BatchMetaKeys, Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..common import split_text_by_punctuation

OP_NAME = "most_relevant_entities_aggregator"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class MostRelevantEntitiesAggregator(Aggregator):
    """Extracts and ranks entities closely related to a given entity from provided texts.

    The operator uses a language model API to identify and rank entities, filtering out
    entities of the same type as the given entity. The ranked list is sorted in descending
    order of importance. Input texts are aggregated and passed to the model, with an
    optional token limit. The output is parsed using a regular expression to extract the
    relevant entities. Results are stored in the batch metadata under the key
    'most_relevant_entities'. The operator retries the API call up to a specified number of
    times in case of errors. The system prompt, input template, and output pattern can be
    customized."""

    DEFAULT_SYSTEM_TEMPLATE = (
        "给定与`{entity}`相关的一些文档，"
        "总结一些与`{entity}`最为相关的`{entity_type}`。\n"
        "要求：\n"
        "- 不用包含与{entity}为同一{entity_type}的{entity_type}。\n"
        "- 请按照人物的重要性进行排序，**越重要人物在列表越前面**。\n"
        "- 你的返回格式如下：\n"
        "## 分析\n"
        "你对各个{entity_type}与{entity}关联度的分析\n"
        "## 列表\n"
        "人物1, 人物2, 人物3, ..."
    )

    DEFAULT_INPUT_TEMPLATE = "`{entity}`的相关文档：\n" "{sub_docs}\n\n" "与`{entity}`最相关的一些`{entity_type}`：\n"

    DEFAULT_OUTPUT_PATTERN = r"\#\#\s*列表\s*(.*?)\Z"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        entity: str = None,
        query_entity_type: str = None,
        input_key: str = MetaKeys.event_description,
        output_key: str = BatchMetaKeys.most_relevant_entities,
        max_token_num: Optional[PositiveInt] = None,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        input_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param entity: The given entity.
        :param query_entity_type: The type of queried relevant entities.
        :param input_key: The input key in the meta field of the samples.
            It is "event_description" in default.
        :param output_key: The output key in the aggregation field of the
            samples. It is "most_relevant_entities" in default.
        :param max_token_num: The max token num of the total tokens of the
            sub documents. Without limitation if it is None.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt_template: The system prompt template.
        :param input_template: The input template.
        :param output_pattern: The output pattern.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        if entity is None or query_entity_type is None:
            raise ValueError("The entity and query_entity_type cannot be None!")

        self.entity = entity
        self.query_entity_type = query_entity_type
        self.input_key = input_key
        self.output_key = output_key
        self.max_token_num = max_token_num

        system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_TEMPLATE
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN
        self.system_prompt = system_prompt_template.format(entity=entity, entity_type=query_entity_type)

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

    def parse_output(self, response):
        pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(response)
        if matches:
            result = matches[0].strip()
        else:
            result = ""
        result = split_text_by_punctuation(result)

        return result

    def query_most_relevant_entities(self, sub_docs, rank=None):
        if not sub_docs:
            return ""

        model, tokenizer = get_model(self.model_key, rank, self.use_cuda())
        token_nums = [len(tokenizer.encode(sub_doc)) for sub_doc in sub_docs]
        if self.max_token_num is None:
            final_docs = sub_docs
        else:
            final_docs = []
            total_num = 0
            for token_num, doc in zip(token_nums, sub_docs):
                total_num += token_num
                if total_num > self.max_token_num:
                    break
                final_docs.append(doc)

        doc_str = "\n\n".join(final_docs)
        input_prompt = self.input_template.format(
            entity=self.entity, entity_type=self.query_entity_type, sub_docs=doc_str
        )

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]
        result = np.array([], dtype=str)
        for i in range(self.try_num):
            try:
                response = model(messages, **self.sampling_params)
                cur_result = self.parse_output(response)
                if len(cur_result) > 0:
                    result = cur_result
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        return result

    def process_single(self, sample=None, rank=None):
        if self.output_key in sample[Fields.batch_meta]:
            return sample

        if Fields.meta not in sample or self.input_key not in sample[Fields.meta][0]:
            logger.warning("The input key does not exist in the sample!")
            return sample

        sub_docs = [d[self.input_key] for d in sample[Fields.meta]]

        # if not batched sample
        if not is_string_list(sub_docs):
            return sample

        sample[Fields.batch_meta][self.output_key] = self.query_most_relevant_entities(sub_docs, rank=rank)

        return sample
