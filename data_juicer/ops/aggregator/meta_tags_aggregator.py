import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Aggregator
from data_juicer.utils.common_utils import is_string_list
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "meta_tags_aggregator"


# TODO: LLM-based inference.
@OPERATORS.register_module(OP_NAME)
class MetaTagsAggregator(Aggregator):
    """Merge similar meta tags into a single, unified tag.

    This operator aggregates and consolidates similar meta tags from the input data. It can
    handle two scenarios:
    - When a set of target tags is provided, it maps the original tags to these predefined
      categories. If a "miscellaneous" or "other" category is included, any tags that do not
      fit into the specified categories are grouped under this label.
    - When no target tags are provided, it generates reasonable categories based on the
      similarity and frequency of the input tags.

    The operator uses a language model (default: gpt-4o) to analyze and merge the tags. The
    system prompt, input template, and output pattern can be customized. The aggregated tags
    are then updated in the input sample's metadata."""

    DEFAULT_SYSTEM_PROMPT = (
        "给定一些标签以及这些标签出现的频次，合并意思相近的标签。\n"
        "要求：\n"
        "- 任务分为两种情况，一种是给定合并后的标签，需要将合并前的标签映射到"
        "这些标签。如果给定的合并后的标签中有类似“其他”这种标签，将无法归类的"
        "标签合并到“其他”。以下是这种情况的一个样例：\n"
        "合并后的标签应限定在[科技, 健康, 其他]中。\n"
        "| 合并前标签 | 频次 |\n"
        "| ------ | ------ |\n"
        "| 医疗 | 20 |\n"
        "| 信息技术 | 16 |\n"
        "| 学习 | 19 |\n"
        "| 气候变化 | 22 |\n"
        "| 人工智能 | 11 |\n"
        "| 养生 | 17 |\n"
        "| 科学创新 | 10 |\n"
        "\n"
        "## 分析：“信息技术”、“人工智能”、“科学创新”都属于“科技”类别，“医疗"
        "”和“养生”跟“健康”有关联，“学习”、“气候变化”和“科技”还有“健康”关"
        "联不强，应该被归为“其他”。\n"
        "## 标签合并：\n"
        "** 医疗归类为健康 **\n"
        "** 信息技术归类为科技 **\n"
        "** 学习归类为其他 **\n"
        "** 气候变化归类为其他 **\n"
        "** 人工智能归类为科技 **\n"
        "** 养生归类为健康 **\n"
        "** 科学创新归类为科技 **\n"
        "- 另外一种情况没有事先给定合并后的标签，需要生成合理的标签类别："
        "| 合并前标签 | 频次 |\n"
        "| ------ | ------ |\n"
        "| 医疗 | 20 |\n"
        "| 信息技术 | 16 |\n"
        "| 学习 | 2 |\n"
        "| 气候变化 | 1 |\n"
        "| 人工智能 | 11 |\n"
        "| 养生 | 17 |\n"
        "| 科学创新 | 10 |\n"
        "\n"
        "## 分析：“信息技术”、“人工智能”、“科学创新”这三个标签比较相近，归为"
        "同一类，都属于“科技”类别，“医疗”和“养生”都跟“健康”有关系，可以归"
        "类为“健康”，“学习”和“气候变化”跟其他标签关联度不强，且频次较低，"
        "统一归类为“其他”。\n"
        "## 标签合并：\n"
        "** 医疗归类为健康 **\n"
        "** 信息技术归类为科技 **\n"
        "** 学习归类为其他 **\n"
        "** 气候变化归类为其他 **\n"
        "** 人工智能归类为科技 **\n"
        "** 养生归类为健康 **\n"
        "** 科学创新归类为科技 **\n"
    )

    DEFAULT_INPUT_TEMPLATE = "{target_tag_str}" "| 合并前标签 | 频次 |\n" "| ------ | ------ |\n" "{tag_strs}"
    DEFAULT_TARGET_TAG_TEMPLATE = "合并后的标签应限定在[{target_tags}]中。\n"
    DEFAULT_TAG_TEMPLATE = "| {tag} | {cnt} |"

    DEFAULT_OUTPUT_PATTERN = r"\*\*\s*(\w+)归类为(\w+)\s*\*\*"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        meta_tag_key: str = MetaKeys.dialog_sentiment_labels,
        target_tags: Optional[List[str]] = None,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        target_tag_template: Optional[str] = None,
        tag_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param meta_tag_key: The key of the meta tag to be mapped.
        :param target_tags: The tags that is supposed to be mapped to.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: The system prompt.
        :param input_template: The input template.
        :param target_tag_template: The tap template for target tags.
        :param tag_template: The tap template for each tag and its
            frequency.
        :param output_pattern: The output pattern.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.meta_tag_key = meta_tag_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        target_tag_template = target_tag_template or self.DEFAULT_TARGET_TAG_TEMPLATE
        self.tag_template = tag_template or self.DEFAULT_TAG_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.target_tag_str = ""
        if target_tags:
            self.target_tag_str = target_tag_template.format(target_tags=", ".join(target_tags))

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
        tag_map = {tag1: tag2 for tag1, tag2 in matches}
        return tag_map

    def meta_map(self, meta_cnts, rank=None):
        model, _ = get_model(self.model_key, rank, self.use_cuda())

        tag_strs = [self.tag_template.format(tag=k, cnt=meta_cnts[k]) for k in meta_cnts]
        input_prompt = self.input_template.format(target_tag_str=self.target_tag_str, tag_strs="\n".join(tag_strs))

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]
        tag_map = {}
        for i in range(self.try_num):
            try:
                response = model(messages, **self.sampling_params)
                tag_map = self.parse_output(response)
                if len(tag_map) > 0:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        return tag_map

    def process_single(self, sample=None, rank=None):
        if Fields.meta not in sample:
            logger.warning("Not any meta in the sample!")
            return sample

        metas = sample[Fields.meta]
        # if not batched sample
        if not isinstance(metas, list):
            logger.warning("Not a batched sample!")
            return sample

        meta_cnts = {}

        def update_dict(key):
            if key in meta_cnts:
                meta_cnts[key] += 1
            else:
                meta_cnts[key] = 1

        for meta in metas:
            tag = meta[self.meta_tag_key]
            if isinstance(tag, str):
                update_dict(tag)
            elif is_string_list(tag):
                for t in tag:
                    update_dict(t)
            else:
                logger.warning("Meta tag must be string or list of string!")
                return sample

        tag_map = self.meta_map(meta_cnts, rank=rank)
        for i in range(len(metas)):
            tag = metas[i][self.meta_tag_key]
            if isinstance(tag, str) and tag in tag_map:
                metas[i][self.meta_tag_key] = tag_map[tag]
            elif is_string_list(tag):
                metas[i][self.meta_tag_key] = [tag_map[t] if t in tag_map else t for t in tag]

        return sample
