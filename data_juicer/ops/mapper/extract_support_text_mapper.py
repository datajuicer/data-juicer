from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "extract_support_text_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractSupportTextMapper(Mapper):
    """Extracts a supporting sub-text from the original text based on a given summary.

    This operator uses an API model to identify and extract a segment of the original text
    that best matches the provided summary. It leverages a system prompt and input template
    to guide the extraction process. The extracted support text is stored in the specified
    meta field key. If the extraction fails or returns an empty string, the original summary
    is used as a fallback. The operator retries the extraction up to a specified number of
    times in case of errors."""

    DEFAULT_SYSTEM_PROMPT = (
        "你将扮演一个文本摘录助手的角色。你的主要任务是基于给定"
        "的文章（称为“原文”）以及对原文某个部分的简短描述或总结"
        "（称为“总结”），准确地识别并提取出与该总结相对应的原文"
        "片段。\n"
        "要求：\n"
        "- 你需要尽可能精确地匹配到最符合总结内容的那部分内容\n"
        "- 如果存在多个可能的答案，请选择最贴近总结意思的那个\n"
        "- 下面是一个例子帮助理解这一过程：\n"
        "### 原文：\n"
        "《红楼梦》是中国古典小说四大名著之一，由清代作家曹雪芹创"
        "作。它讲述了贾宝玉、林黛玉等人的爱情故事及四大家族的兴衰"
        "历程。书中通过复杂的人物关系展现了封建社会的各种矛盾冲突"
        "。其中关于贾府内部斗争的部分尤其精彩，特别是王熙凤与尤二"
        "姐之间的争斗，生动描绘了权力争夺下的女性形象。此外，《红"
        "楼梦》还以其精美的诗词闻名，这些诗词不仅增添了文学色彩，"
        "也深刻反映了人物的性格特点和命运走向。\n\n"
        "### 总结：\n"
        "描述了书中的两个女性角色之间围绕权力展开的竞争。\n\n"
        "### 原文摘录：\n"
        "其中关于贾府内部斗争的部分尤其精彩，特别是王熙凤与尤二姐"
        "之间的争斗，生动描绘了权力争夺下的女性形象。"
    )
    DEFAULT_INPUT_TEMPLATE = "### 原文：\n{text}\n\n" "### 总结：\n{summary}\n\n" "### 原文摘录：\n"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        summary_key: str = MetaKeys.event_description,
        support_text_key: str = MetaKeys.support_text,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        try_num: PositiveInt = 3,
        drop_text: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param summary_key: The key name to store the input summary in the
            meta field. It's "event_description" in default.
        :param support_text_key: The key name to store the output
            support text for the summary in the meta field. It's
            "support_text" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param input_template: Template for building the model input.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.summary_key = summary_key
        self.support_text_key = support_text_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num
        self.drop_text = drop_text

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.support_text_key in sample[Fields.meta]:
            return sample

        client = get_model(self.model_key, rank=rank)

        if self.summary_key not in sample[Fields.meta]:
            logger.warning(f"{self.summary_key} does not exist in the meta field!")
            return sample
        summary = sample[Fields.meta][self.summary_key]
        if not isinstance(summary, str):
            logger.warning("Invalid input summary!")
            return sample

        input_prompt = self.input_template.format(text=sample[self.text_key], summary=summary)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]

        support_text = ""
        for i in range(self.try_num):
            try:
                response = client(messages, **self.sampling_params)
                support_text = response.strip()
                if len(support_text) > 0:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")
        # default to summary if return None
        if not support_text:
            support_text = summary

        sample[Fields.meta][self.support_text_key] = support_text
        return sample
