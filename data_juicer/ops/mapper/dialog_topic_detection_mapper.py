import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import NonNegativeInt, PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "dialog_topic_detection_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogTopicDetectionMapper(Mapper):
    """Generates user's topic labels and analysis in a dialog.

    This operator processes a dialog to detect and label the topics discussed by the user.
    It takes input from `history_key`, `query_key`, and `response_key` and outputs lists of
    labels and analysis for each query in the dialog. The operator uses a predefined system
    prompt and templates to build the input prompt for the API call. It supports customizing
    the system prompt, templates, and patterns for parsing the API response. The results are
    stored in the `meta` field under the keys specified by `labels_key` and `analysis_key`.
    If these keys already exist in the `meta` field, the operator skips processing. The
    operator retries the API call up to `try_num` times in case of errors."""

    DEFAULT_SYSTEM_PROMPT = (
        "请判断用户和LLM多轮对话中用户所讨论的话题。\n"
        "要求：\n"
        "- 针对用户的每个query，需要先进行分析，然后列出用户正在讨论的话题，下面是"
        "一个样例，请模仿样例格式输出。\n"
        "用户：你好，今天我们来聊聊秦始皇吧。\n"
        "话题分析：用户提到秦始皇，这是中国历史上第一位皇帝。\n"
        "话题类别：历史\n"
        "LLM：当然可以，秦始皇是中国历史上第一个统一全国的皇帝，他在公元前221年建"
        "立了秦朝，并采取了一系列重要的改革措施，如统一文字、度量衡和货币等。\n"
        "用户：秦始皇修建的长城和现在的长城有什么区别？\n"
        "话题分析：用户提到秦始皇修建的长城，并将其与现代长城进行比较，涉及建筑历史"
        "和地理位置。\n"
        "话题类别：历史"
        "LLM：秦始皇时期修建的长城主要是为了抵御北方游牧民族的入侵，它的规模和修建"
        "技术相对较为简陋。现代人所看到的长城大部分是明朝时期修建和扩建的，明长城不"
        "仅规模更大、结构更坚固，而且保存得比较完好。\n"
        "用户：有意思，那么长城的具体位置在哪些省份呢？\n"
        "话题分析：用户询问长城的具体位置，涉及到地理知识。\n"
        "话题类别：地理\n"
        "LLM：长城横跨中国北方多个省份，主要包括河北、山西、内蒙古、宁夏、陕西、甘"
        "肃和北京等。每一段长城都建在关键的战略位置，以便最大限度地发挥其防御作用"
        "。\n"
    )
    DEFAULT_QUERY_TEMPLATE = "用户：{query}\n"
    DEFAULT_RESPONSE_TEMPLATE = "LLM：{response}\n"
    DEFAULT_CANDIDATES_TEMPLATE = "备选话题类别：[{candidate_str}]"
    DEFAULT_ANALYSIS_TEMPLATE = "话题分析：{analysis}\n"
    DEFAULT_LABELS_TEMPLATE = "话题类别：{labels}\n"
    DEFAULT_ANALYSIS_PATTERN = "话题分析：(.*?)\n"
    DEFAULT_LABELS_PATTERN = "话题类别：(.*?)($|\n)"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        topic_candidates: Optional[List[str]] = None,
        max_round: NonNegativeInt = 10,
        *,
        labels_key: str = MetaKeys.dialog_topic_labels,
        analysis_key: str = MetaKeys.dialog_topic_labels_analysis,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        query_template: Optional[str] = None,
        response_template: Optional[str] = None,
        candidate_template: Optional[str] = None,
        analysis_template: Optional[str] = None,
        labels_template: Optional[str] = None,
        analysis_pattern: Optional[str] = None,
        labels_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.

        :param api_model: API model name.
        :param topic_candidates: The output topic candidates. Use
            open-domain topic labels if it is None.
        :param max_round: The max num of round in the dialog to build the
            prompt.
        :param labels_key: The key name in the meta field to store the
            output labels. It is 'dialog_topic_labels' in default.
        :param analysis_key: The key name in the meta field to store the
            corresponding analysis. It is 'dialog_topic_labels_analysis'
            in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param query_template: Template for query part to build the input
            prompt.
        :param response_template: Template for response part to build the
            input prompt.
        :param candidate_template: Template for topic candidates to
            build the input prompt.
        :param analysis_template: Template for analysis part to build the
            input prompt.
        :param labels_template: Template for labels part to build the
            input prompt.
        :param analysis_pattern: Pattern to parse the return topic
            analysis.
        :param labels_pattern: Pattern to parse the return topic
            labels.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.topic_candidates = topic_candidates
        self.max_round = max_round
        self.labels_key = labels_key
        self.analysis_key = analysis_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.query_template = query_template or self.DEFAULT_QUERY_TEMPLATE
        self.response_template = response_template or self.DEFAULT_RESPONSE_TEMPLATE
        self.candidate_template = candidate_template or self.DEFAULT_CANDIDATES_TEMPLATE
        self.analysis_template = analysis_template or self.DEFAULT_ANALYSIS_TEMPLATE
        self.labels_template = labels_template or self.DEFAULT_LABELS_TEMPLATE
        self.analysis_pattern = analysis_pattern or self.DEFAULT_ANALYSIS_PATTERN
        self.labels_pattern = labels_pattern or self.DEFAULT_LABELS_PATTERN

        self.sampling_params = sampling_params

        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num

    def build_input(self, history, query):
        if self.topic_candidates:
            input_prompt = self.candidate_template.format(candidate_str=",".join(self.topic_candidates))
        else:
            input_prompt = ""

        if self.max_round > 0:
            input_prompt += "".join(history[-self.max_round * 4 :])

        input_prompt += self.query_template.format(query=query[0])

        return input_prompt

    def parse_output(self, response):
        analysis = ""
        labels = ""

        match = re.search(self.analysis_pattern, response)
        if match:
            analysis = match.group(1)

        match = re.search(self.labels_pattern, response)
        if match:
            labels = match.group(1)

        return analysis, labels

    def process_single(self, sample, rank=None):
        meta = sample[Fields.meta]
        if self.labels_key in meta and self.analysis_key in meta:
            return sample

        client = get_model(self.model_key, rank=rank)

        analysis_list = []
        labels_list = []
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
                    analysis, labels = self.parse_output(response)
                    if len(analysis) > 0:
                        break
                except Exception as e:
                    logger.warning(f"Exception: {e}")

            analysis_list.append(analysis)
            labels_list.append(labels)

            history.append(self.query_template.format(query=qa[0]))
            history.append(self.analysis_template.format(analysis=analysis))
            history.append(self.labels_template.format(labels=labels))
            history.append(self.response_template.format(response=qa[1]))

        meta[self.labels_key] = labels_list
        meta[self.analysis_key] = analysis_list

        return sample
