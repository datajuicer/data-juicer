import re
from typing import Dict, Optional

from loguru import logger
from pydantic import NonNegativeInt, PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "dialog_sentiment_intensity_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogSentimentIntensityMapper(Mapper):
    """Mapper to predict user's sentiment intensity in a dialog, ranging from -5 to 5.

    This operator analyzes the sentiment of user queries in a dialog and outputs a list of
    sentiment intensities and corresponding analyses. The sentiment intensity ranges from -5
    (extremely negative) to 5 (extremely positive), with 0 indicating a neutral sentiment.
    The analysis is based on the provided history, query, and response keys. The default
    system prompt and templates guide the sentiment analysis process. The results are stored
    in the meta field under 'dialog_sentiment_intensity' for intensities and
    'dialog_sentiment_intensity_analysis' for analyses. The operator uses an API model to
    generate the sentiment analysis, with configurable retry attempts and sampling
    parameters."""

    DEFAULT_SYSTEM_PROMPT = (
        "请判断用户和LLM多轮对话中用户的情绪变化。\n"
        "要求：\n"
        "- 用户情绪值是-5到5之间到整数，-5表示极度负面，5表示极度正面，"
        "-5到5之间数值表示情绪从负面逐渐到正面的变化过程，0代表情呈绪中性。\n"
        "- 只输出当轮对话的分析，不要继续构造对话。\n"
        "- 需要先进行分析，然后确定用户的情绪值，下面是一个样例，请模仿样例格式输出。\n"
        "用户：你好，我对可持续发展的定义有点模糊，帮我解释一下？\n"
        "情绪分析：刚开始，还没得到LLM回复，用户情绪呈中性。\n"
        "情绪值：0\n"
        "LLM：当然可以！可持续发展是指在满足当代人的需求的同时，不损害子孙后代满足其自"
        "身需求的能力的发展模式。它包括经济发展、社会发展和环境保护三个主要方面。通过合"
        "理利用资源和保护环境，我们可以确保未来的世代也能享有健全的生态系统和经济制度。\n"
        "用户：谢谢你的解释！那你能告诉我一些普通人可以采取的可持续生活方式吗？\n"
        "情绪分析：对回答感到满意，情绪正面。\n"
        "情绪值：1\n"
        "LLM：当然可以，普通人可以通过减少一次性产品的使用、选择公共交通或拼车、节约用"
        "水、以及支持本地和可持续发展的企业等方式来践行可持续生活。此外，关注垃圾分类和"
        "多用电子账单也是不错的选择。\n"
        "用户：你提到支持本地企业，这一点我很感兴趣。能详细说说为什么这对可持续发展有促"
        "进作用吗？\n"
        "情绪分析：觉得回答实用且具体，情绪进一步转好。\n"
        "情绪值：2\n"
        "LLM：呃，我最近发现了一部新电影，讲述了一个关于外星人和地球土著合作保护环境的"
        "故事。虽然它是科幻片，但很有启发性，推荐你去看看。\n"
        "用户：什么吗，根本是答非所问。\n"
        "情绪分析：LLM没有回应问题而是提到无关内容，导致用户情绪直线下降。\n"
        "情绪值：-2\n"
        "LLM：抱歉刚才的偏题！支持本地企业有助于减少长途运输产生的碳足迹，使供应链更加"
        "环保。此外，本地企业也更有可能采用可持续的生产方式，同时促进社区经济的繁荣。\n"
        "用户：还行吧，算你能够掰回来。\n"
        "情绪分析：问题得到解答，问题偏题得到纠正，情绪稍有好转。\n"
        "情绪值：-1\n"
    )
    DEFAULT_QUERY_TEMPLATE = "用户：{query}\n"
    DEFAULT_RESPONSE_TEMPLATE = "LLM：{response}\n"
    DEFAULT_ANALYSIS_TEMPLATE = "情绪分析：{analysis}\n"
    DEFAULT_INTENSITY_TEMPLATE = "情绪值：{intensity}\n"
    DEFAULT_ANALYSIS_PATTERN = "情绪分析：(.*?)\n"
    DEFAULT_INTENSITY_PATTERN = "情绪值：(.*?)($|\n)"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        max_round: NonNegativeInt = 10,
        *,
        intensities_key: str = MetaKeys.dialog_sentiment_intensity,
        analysis_key: str = MetaKeys.dialog_sentiment_intensity_analysis,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        query_template: Optional[str] = None,
        response_template: Optional[str] = None,
        analysis_template: Optional[str] = None,
        intensity_template: Optional[str] = None,
        analysis_pattern: Optional[str] = None,
        intensity_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.

        :param api_model: API model name.
        :param max_round: The max num of round in the dialog to build the
            prompt.
        :param intensities_key: The key name in the meta field to store
            the output sentiment intensities. It is
            'dialog_sentiment_intensity' in default.
        :param analysis_key: The key name in the meta field to store the
            corresponding analysis. It is
            'dialog_sentiment_intensity_analysis' in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param query_template: Template for query part to build the input
            prompt.
        :param response_template: Template for response part to build the
            input prompt.
        :param analysis_template: Template for analysis part to build the
            input prompt.
        :param intensity_template: Template for intensity part to build the
            input prompt.
        :param analysis_pattern: Pattern to parse the return sentiment
            analysis.
        :param intensity_pattern: Pattern to parse the return sentiment
            intensity.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.max_round = max_round
        self.intensities_key = intensities_key
        self.analysis_key = analysis_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.query_template = query_template or self.DEFAULT_QUERY_TEMPLATE
        self.response_template = response_template or self.DEFAULT_RESPONSE_TEMPLATE
        self.analysis_template = analysis_template or self.DEFAULT_ANALYSIS_TEMPLATE
        self.intensity_template = intensity_template or self.DEFAULT_INTENSITY_TEMPLATE
        self.analysis_pattern = analysis_pattern or self.DEFAULT_ANALYSIS_PATTERN
        self.intensity_pattern = intensity_pattern or self.DEFAULT_INTENSITY_PATTERN

        self.sampling_params = sampling_params

        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num

    def build_input(self, history, query):
        if self.max_round > 0:
            input_prompt = "".join(history[-self.max_round * 4 :])
        else:
            input_prompt = ""
        input_prompt += self.query_template.format(query=query[0])

        return input_prompt

    def parse_output(self, response):
        analysis = ""
        intensity = 0

        match = re.search(self.analysis_pattern, response)
        if match:
            analysis = match.group(1)

        match = re.search(self.intensity_pattern, response)
        if match:
            intensity = int(match.group(1))

        return analysis, intensity

    def process_single(self, sample, rank=None):
        meta = sample[Fields.meta]
        if self.intensities_key in meta and self.analysis_key in meta:
            return sample

        client = get_model(self.model_key, rank=rank)

        analysis_list = []
        intensities = []
        history = []

        dialog = sample[self.history_key]
        if self.query_key in sample and sample[self.query_key]:
            if self.response_key in sample and sample[self.response_key]:
                dialog.append((sample[self.query_key], sample[self.response_key]))
            else:
                dialog.append((sample[self.query_key], ""))

        for qa in dialog:
            input_prompt = self.build_input(history, qa)
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": input_prompt,
                },
            ]

            for _ in range(self.try_num):
                try:
                    response = client(messages, **self.sampling_params)
                    analysis, intensity = self.parse_output(response)
                    if len(analysis) > 0:
                        break
                except Exception as e:
                    logger.warning(f"Exception: {e}")

            analysis_list.append(analysis)
            intensities.append(intensity)

            history.append(self.query_template.format(query=qa[0]))
            history.append(self.analysis_template.format(analysis=analysis))
            history.append(self.intensity_template.format(intensity=intensity))
            history.append(self.response_template.format(response=qa[1]))

        meta[self.intensities_key] = intensities
        meta[self.analysis_key] = analysis_list

        return sample
