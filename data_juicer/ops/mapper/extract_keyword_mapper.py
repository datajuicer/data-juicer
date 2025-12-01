# flake8: noqa: E501

import re
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..common import split_text_by_punctuation

OP_NAME = "extract_keyword_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractKeywordMapper(Mapper):
    """Generate keywords for the text.

    This operator uses a specified API model to generate high-level keywords that summarize
    the main concepts, themes, or topics of the input text. The generated keywords are
    stored in the meta field under the key specified by `keyword_key`. The operator retries
    the API call up to `try_num` times in case of errors. If `drop_text` is set to True, the
    original text is removed from the sample after processing. The operator uses a default
    prompt template and completion delimiter, which can be customized. The output is parsed
    using a regular expression to extract the keywords."""

    # This prompt is modified from light RAG
    # https://github.com/HKUDS/LightRAG
    DEFAULT_PROMPT_TEMPLATE = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords" <high_level_keywords>)

3. Return output in the language of the given text.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```
################
Output:
("content_keywords" "power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Text:
```
他们不再是单纯的执行者；他们已成为某个超越星辰与条纹的领域的信息守护者。这一使命的提升不能被规则和既定协议所束缚——它需要一种新的视角，一种新的决心。

随着与华盛顿的通讯在背景中嗡嗡作响，对话中的紧张情绪通过嘟嘟声和静电噪音贯穿始终。团队站立着，一股不祥的气息笼罩着他们。显然，他们在接下来几个小时内做出的决定可能会重新定义人类在宇宙中的位置，或者将他们置于无知和潜在危险之中。

随着与星辰的联系变得更加牢固，小组开始处理逐渐成形的警告，从被动接受者转变为积极参与者。梅瑟后来的直觉占据了上风——团队的任务已经演变，不再仅仅是观察和报告，而是互动和准备。一场蜕变已经开始，而“杜尔塞行动”则以他们大胆的新频率震动，这种基调不是由世俗设定的
```
#############
Output:
("content_keywords" "任务演变, 决策制定, 积极参与, 宇宙意义"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
```
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
```
#############
Output:
("content_keywords" "first contact, control, communication, cosmic significance"){completion_delimiter}
-Real Data-
######################
Text:
```
{input_text}
```
######################
Output:
"""

    DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
    DEFAULT_OUTPUT_PATTERN = r'\("content_keywords"(.*?)\)'

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        keyword_key: str = MetaKeys.keyword,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        completion_delimiter: Optional[str] = None,
        output_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        drop_text: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param keyword_key: The key name to store the keywords in the meta
            field. It's "keyword" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param prompt_template: The template of input prompt.
        :param completion_delimiter: To mark the end of the output.
        :param output_pattern: Regular expression for parsing keywords.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.keyword_key = keyword_key

        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.completion_delimiter = completion_delimiter or self.DEFAULT_COMPLETION_DELIMITER
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output):
        keywords = []

        output_pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = output_pattern.findall(raw_output)
        for record in matches:
            items = split_text_by_punctuation(record)
            keywords.extend(items)

        return keywords

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.keyword_key in sample[Fields.meta]:
            return sample

        client = get_model(self.model_key, rank=rank)

        input_prompt = self.prompt_template.format(
            completion_delimiter=self.completion_delimiter, input_text=sample[self.text_key]
        )
        messages = [{"role": "user", "content": input_prompt}]

        keywords = np.array([], dtype=str)
        for _ in range(self.try_num):
            try:
                response = client(messages, **self.sampling_params)
                results = self.parse_output(response)
                if len(results) > 0:
                    keywords = results
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        sample[Fields.meta][self.keyword_key] = keywords
        if self.drop_text:
            sample.pop(self.text_key)

        return sample
