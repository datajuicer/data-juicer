import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import NonNegativeInt, PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "dialog_sentiment_detection_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogSentimentDetectionMapper(Mapper):
    """Generates sentiment labels and analysis for user queries in a dialog.

    This operator processes a dialog to detect and label the sentiments expressed by the
    user. It uses the provided history, query, and response keys to construct prompts for an
    API call. The API returns sentiment analysis and labels, which are then parsed and
    stored in the sample's metadata under the 'dialog_sentiment_labels' and
    'dialog_sentiment_labels_analysis' keys. The operator supports custom templates and
    patterns for prompt construction and output parsing. If no sentiment candidates are
    provided, it uses open-domain sentiment labels. The operator retries the API call up to
    a specified number of times in case of errors."""

    DEFAULT_SYSTEM_PROMPT = (
        "请判断用户和LLM多轮对话中用户所具有的情绪。\n"
        "要求：\n"
        "- 需要先进行分析，然后罗列用户所具有的情绪，下面是一个样例，请模仿样例格式输出"
        "。\n"
        "用户：最近工作压力好大，我觉得整个人都快被压垮了。\n"
        "情感分析：用户的言语中透露出明显的压力和疲惫感，可能还夹杂着一些无助和焦虑。\n"
        "情感类别：压力、疲惫、无助、焦虑\n"
        "LLM：听起来你真的承受了很多，面临这种情况确实不容易。有没有考虑过找一些放松的"
        "方式，比如听音乐或者散步来减轻压力呢？\n"
        "用户：试过了，但是好像没什么效果，每天的事情都堆积如山。\n"
        "情感分析：用户感到无力解决现状，有挫败感，并且对尝试放松的方式失去信心。\n"
        "情感类别：无力、挫败\n"
        "LLM：我理解你的感受，有时候压力积累到一定程度确实让人难以承受。或许你可以尝试"
        "规划一下时间，把任务分成小块来完成，这样可能会减少一些压力感。\n"
        "用户：这个主意不错，我会试着让自己更有条理一些，谢谢你的建议。\n"
        "情感分析：用户对建议表现出认同和感激，同时展现出试图积极面对问题的态度。\n"
        "情感类别：认同、感激、积极\n"
        "LLM：不用谢，我很高兴能帮到你。记得给自己一些时间去适应新的计划，有任何需要"
        "随时可以跟我说哦！\n"
    )
    DEFAULT_QUERY_TEMPLATE = "用户：{query}\n"
    DEFAULT_RESPONSE_TEMPLATE = "LLM：{response}\n"
    DEFAULT_CANDIDATES_TEMPLATE = "备选情感类别：[{candidate_str}]"
    DEFAULT_ANALYSIS_TEMPLATE = "情感分析：{analysis}\n"
    DEFAULT_LABELS_TEMPLATE = "情感类别：{labels}\n"
    DEFAULT_ANALYSIS_PATTERN = "情感分析：(.*?)\n"
    DEFAULT_LABELS_PATTERN = "情感类别：(.*?)($|\n)"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        sentiment_candidates: Optional[List[str]] = None,
        max_round: NonNegativeInt = 10,
        *,
        labels_key: str = MetaKeys.dialog_sentiment_labels,
        analysis_key: str = MetaKeys.dialog_sentiment_labels_analysis,
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
        :param sentiment_candidates: The output sentiment candidates. Use
            open-domain sentiment labels if it is None.
        :param max_round: The max num of round in the dialog to build the
            prompt.
        :param labels_key: The key name in the meta field to store the
            output labels. It is 'dialog_sentiment_labels' in default.
        :param analysis_key: The key name in the meta field to store the
            corresponding analysis. It is
            'dialog_sentiment_labels_analysis' in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param query_template: Template for query part to build the input
            prompt.
        :param response_template: Template for response part to build the
            input prompt.
        :param candidate_template: Template for sentiment candidates to
            build the input prompt.
        :param analysis_template: Template for analysis part to build the
            input prompt.
        :param labels_template: Template for labels part to build the
            input prompt.
        :param analysis_pattern: Pattern to parse the return sentiment
            analysis.
        :param labels_pattern: Pattern to parse the return sentiment
            labels.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.sentiment_candidates = sentiment_candidates
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
        if self.sentiment_candidates:
            input_prompt = self.candidate_template.format(candidate_str=",".join(self.sentiment_candidates))
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
