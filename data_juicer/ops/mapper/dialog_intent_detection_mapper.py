import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import NonNegativeInt, PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "dialog_intent_detection_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogIntentDetectionMapper(Mapper):
    """Generates user's intent labels in a dialog by analyzing the history, query, and
    response.

    This operator processes a dialog to identify and label the user's intent. It uses a
    predefined system prompt and templates to build input prompts for an API call. The API
    model (e.g., GPT-4) is used to analyze the dialog and generate intent labels and
    analysis. The results are stored in the meta field under 'dialog_intent_labels' and
    'dialog_intent_labels_analysis'. The operator supports customizing the system prompt,
    templates, and patterns for parsing the API response. If the intent candidates are
    provided, they are included in the input prompt. The operator retries the API call up to
    a specified number of times if there are errors."""

    DEFAULT_SYSTEM_PROMPT = (
        "请判断用户和LLM多轮对话中用户的意图。\n"
        "要求：\n"
        "- 需要先进行分析，然后列出用户所具有的意图，下面是一个样例，请模仿样例格式输出"
        "。\n"
        "用户：你好，我最近对人工智能很感兴趣，能给我讲讲什么是机器学习吗？\n"
        "意图分析：用户在请求信息，希望了解有关机器学习的基础知识。\n"
        "意图类别：信息查找\n"
        "LLM：你好！当然可以。机器学习是一种人工智能方法，允许计算机通过数据自动改进和学习。\n"
        "用户：听起来很有趣，有没有推荐的入门书籍或资料？\n"
        "意图分析：用户在请求建议，希望获取关于机器学习的入门资源。\n"
        "意图类别：请求建议\n"
        "LLM：有很多不错的入门书籍和资源。一本常被推荐的书是《Python机器学习实践》（Python"
        " Machine Learning），它涵盖了基础知识和一些实际案例。此外，您还可以参考Coursera"
        "或edX上的在线课程，这些课程提供了系统的学习路径。\n"
        "用户：谢谢你的建议！我还想知道，学习机器学习需要什么样的数学基础？\n"
        "意图分析：用户在寻求信息，希望了解学习机器学习所需的前提条件，特别是在数学方面。\n"
        "意图类别：信息查找\n"
        "LLM：学习机器学习通常需要一定的数学基础，特别是线性代数、概率论和统计学。这些数学领"
        "域帮助理解算法的工作原理和数据模式分析。如果您对这些主题不太熟悉，建议先从相关基础"
        "书籍或在线资源开始学习。\n"
        "用户：明白了，我会先补习这些基础知识。再次感谢你的帮助！\n"
        "意图分析：用户表达感谢，并表示计划付诸行动来补充所需的基础知识。\n"
        "意图类别：其他"
    )
    DEFAULT_QUERY_TEMPLATE = "用户：{query}\n"
    DEFAULT_RESPONSE_TEMPLATE = "LLM：{response}\n"
    DEFAULT_CANDIDATES_TEMPLATE = "备选意图类别：[{candidate_str}]"
    DEFAULT_ANALYSIS_TEMPLATE = "意图分析：{analysis}\n"
    DEFAULT_LABELS_TEMPLATE = "意图类别：{labels}\n"
    DEFAULT_ANALYSIS_PATTERN = "意图分析：(.*?)\n"
    DEFAULT_LABELS_PATTERN = "意图类别：(.*?)($|\n)"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        intent_candidates: Optional[List[str]] = None,
        max_round: NonNegativeInt = 10,
        *,
        labels_key: str = MetaKeys.dialog_intent_labels,
        analysis_key: str = MetaKeys.dialog_intent_labels_analysis,
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
        :param intent_candidates: The output intent candidates. Use the
            intent labels of the open domain if it is None.
        :param max_round: The max num of round in the dialog to build the
            prompt.
        :param labels_key: The key name in the meta field to store the
            output labels. It is 'dialog_intent_labels' in default.
        :param analysis_key: The key name in the meta field to store the
            corresponding analysis. It is 'dialog_intent_labels_analysis'
            in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param query_template: Template for query part to build the input
            prompt.
        :param response_template: Template for response part to build the
            input prompt.
        :param candidate_template: Template for intent candidates to
            build the input prompt.
        :param analysis_template: Template for analysis part to build the
            input prompt.
        :param labels_template: Template for labels to build the
            input prompt.
        :param analysis_pattern: Pattern to parse the return intent
            analysis.
        :param labels_pattern: Pattern to parse the return intent
            labels.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.intent_candidates = intent_candidates
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
        if self.intent_candidates:
            input_prompt = self.candidate_template.format(candidate_str=",".join(self.intent_candidates))
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
