from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Aggregator
from data_juicer.utils.common_utils import (
    avg_split_string_list_under_limit,
    is_string_list,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "nested_aggregator"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class NestedAggregator(Aggregator):
    """Aggregates nested content from multiple samples into a single summary.

    This operator uses a recursive summarization approach to aggregate content from multiple
    samples. It processes the input text, which is split into sub-documents, and generates a
    summary that maintains the average length of the original documents. The aggregation is
    performed using an API model, guided by system prompts and templates. The operator
    supports retrying the API call in case of errors and allows for customization of the
    summarization process through various parameters. The default system prompt and
    templates are provided in Chinese, but they can be customized. The operator uses a
    Hugging Face tokenizer to handle tokenization."""

    DEFAULT_SYSTEM_PROMPT = (
        "给定一些文档碎片，将这些文档整合成一个文档总结。\n"
        "要求：\n"
        "- 总结的长度与文档碎片的平均长度基本一致\n"
        "- 不要包含主观看法\n"
        "- 注意要尽可能保留文本的专有名词\n"
        "- 只输出文档总结不要输出其他内容\n"
        "- 参考如下样例：\n"
        "文档碎片：\n"
        "唐僧师徒四人行至白虎岭，遇上了变化多端的白骨精。\n\n"
        "文档碎片：\n"
        "白骨精首次变身少女送斋，被孙悟空识破打死，唐僧责怪悟空。\n\n"
        "文档碎片：\n"
        "妖怪再变老妇寻女，又被悟空击毙，师傅更加不满，念紧箍咒惩罚。\n\n"
        "文档碎片：\n"
        "不甘心的白骨精第三次化作老公公来诱骗，依旧逃不过金睛火眼。\n\n"
        "文档碎片：\n"
        "最终，在观音菩萨的帮助下，真相大白，唐僧明白了自己的误解。\n\n"
        "\n"
        "文档总结：\n"
        "唐僧师徒在白虎岭三遇白骨精变化诱惑，悟空屡次识破击毙妖怪却遭误解，最终观音相助真相大白。"
    )

    DEFAULT_INPUT_TEMPLATE = "{sub_docs}\n\n" "文档总结：\n"

    DEFAULT_SUB_DOC_TEMPLATE = "文档碎片：\n{text}\n"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        input_key: str = MetaKeys.event_description,
        output_key: str = None,
        max_token_num: Optional[PositiveInt] = None,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        sub_doc_template: Optional[str] = None,
        input_template: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param input_key: The input key in the meta field of the samples.
            It is "event_description" in default.
        :param output_key: The output key in the aggregation field in the
            samples. It is same as the input_key in default.
        :param max_token_num: The max token num of the total tokens of the
            sub documents. Without limitation if it is None.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: The system prompt.
        :param sub_doc_template: The template for input text in each sample.
        :param input_template: The input template.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.input_key = input_key or self.text_key
        self.output_key = output_key or self.input_key
        self.max_token_num = max_token_num

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.sub_doc_template = sub_doc_template or self.DEFAULT_SUB_DOC_TEMPLATE
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE

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
        def if_match(text):
            quotes = [("'", "'"), ('"', '"'), ("“", "”"), ("‘", "’"), ("`", "`")]
            if len(text) < 2:
                return False
            if (text[0], text[-1]) in quotes:
                return True
            else:
                return False

        text = response.strip()
        while if_match(text):
            text = text[1:-1].strip()
        return text

    def recursive_summary(self, sub_docs, rank=None):
        if not sub_docs:
            return ""
        if len(sub_docs) == 1:
            return sub_docs[0]
        model, tokenizer = get_model(self.model_key, rank, self.use_cuda())
        token_nums = [len(tokenizer.encode(sub_doc)) for sub_doc in sub_docs]
        group_docs = avg_split_string_list_under_limit(sub_docs, token_nums, self.max_token_num)
        # merge every two if every single sub doc is a group
        group_num = len(group_docs)
        if group_num == len(sub_docs):
            group_docs = [
                group_docs[i] + group_docs[i + 1] if i + 1 < group_num else group_docs[i]
                for i in range(0, group_num, 2)
            ]
        results = []
        for docs in group_docs:
            doc_strs = [self.sub_doc_template.format(text=d) for d in docs]
            input_prompt = self.input_template.format(sub_docs="\n".join(doc_strs))
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
        return self.recursive_summary(results)

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

        sample[Fields.batch_meta][self.output_key] = self.recursive_summary(sub_docs, rank=rank)

        return sample
