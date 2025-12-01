# This OP is modified from light RAG
# https://github.com/HKUDS/LightRAG

# flake8: noqa: E501

import re
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import NonNegativeInt, PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.common_utils import is_float
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..common import split_text_by_punctuation

OP_NAME = "extract_entity_relation_mapper"


# TODO: LLM-based inference.
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractEntityRelationMapper(Mapper):
    """Extracts entities and relations from text to build a knowledge graph.

    - Identifies entities based on specified types and extracts their names, types, and
      descriptions.
    - Identifies relationships between the entities, including source and target entities,
      relationship descriptions, keywords, and strength scores.
    - Uses a Hugging Face tokenizer and a predefined prompt template to guide the extraction
      process.
    - Outputs entities and relations in a structured format, using delimiters for
      separation.
    - Caches the results in the sample's metadata under the keys 'entity' and 'relation'.
    - Supports multiple retries and gleaning to ensure comprehensive extraction.
    - The default entity types include 'organization', 'person', 'geo', and 'event'."""

    DEFAULT_PROMPT_TEMPLATE = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Return output in the language of the given text as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is reversed by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
#############################
Example 2:

Entity_types: [人物, 技术, 任务, 组织, 地点]
Text:
```
他们不再是单纯的执行者；他们已成为某个超越星辰与条纹的领域的信息守护者。这一使命的提升不能被规则和既定协议所束缚——它需要一种新的视角，一种新的决心。

随着与华盛顿的通讯在背景中嗡嗡作响，对话中的紧张情绪通过嘟嘟声和静电噪音贯穿始终。团队站立着，一股不祥的气息笼罩着他们。显然，他们在接下来几个小时内做出的决定可能会重新定义人类在宇宙中的位置，或者将他们置于无知和潜在危险之中。

随着与星辰的联系变得更加牢固，小组开始处理逐渐成形的警告，从被动接受者转变为积极参与者。梅瑟后来的直觉占据了上风——团队的任务已经演变，不再仅仅是观察和报告，而是互动和准备。一场蜕变已经开始，而“杜尔塞行动”则以他们大胆的新频率震动，这种基调不是由世俗设定的
```
#############
Output:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"地点"{tuple_delimiter}"华盛顿是正在接收通讯的地方，表明其在决策过程中的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"杜尔塞行动"{tuple_delimiter}"任务"{tuple_delimiter}"杜尔塞行动被描述为一项已演变为互动和准备的任务，显示出目标和活动的重大转变。"){record_delimiter}
("entity"{tuple_delimiter}"团队"{tuple_delimiter}"组织"{tuple_delimiter}"团队被描绘成一群从被动观察者转变为积极参与者的人，展示了他们角色的动态变化。"){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"华盛顿"{tuple_delimiter}"团队收到来自华盛顿的通讯，这影响了他们的决策过程。"{tuple_delimiter}"决策、外部影响"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"杜尔塞行动"{tuple_delimiter}"团队直接参与杜尔塞行动，执行其演变后的目标和活动。"{tuple_delimiter}"任务演变、积极参与"{tuple_delimiter}9){completion_delimiter}
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
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
#############################
-Real Data-
######################
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
######################
Output:
"""
    DEFAULT_CONTINUE_PROMPT = (
        "MANY entities were missed in the last extraction.  Add them below using the same format:\n"
    )
    DEFAULT_IF_LOOP_PROMPT = "It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.\n"

    DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
    DEFAULT_TUPLE_DELIMITER = "<|>"
    DEFAULT_RECORD_DELIMITER = "##"
    DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
    DEFAULT_ENTITY_PATTERN = r'\("entity"(.*?)\)'
    DEFAULT_RELATION_PATTERN = r'\("relationship"(.*?)\)'

    def __init__(
        self,
        api_model: str = "gpt-4o",
        entity_types: List[str] = None,
        *,
        entity_key: str = MetaKeys.entity,
        relation_key: str = MetaKeys.relation,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        tuple_delimiter: Optional[str] = None,
        record_delimiter: Optional[str] = None,
        completion_delimiter: Optional[str] = None,
        max_gleaning: NonNegativeInt = 1,
        continue_prompt: Optional[str] = None,
        if_loop_prompt: Optional[str] = None,
        entity_pattern: Optional[str] = None,
        relation_pattern: Optional[str] = None,
        try_num: PositiveInt = 3,
        drop_text: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        Initialization method.
        :param api_model: API model name.
        :param entity_types: Pre-defined entity types for knowledge graph.
        :param entity_key: The key name to store the entities in the meta
            field. It's "entity" in default.
        :param relation_key: The field name to store the relations between
            entities. It's "relation" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param prompt_template: The template of input prompt.
        :param tuple_delimiter: Delimiter to separate items in outputs.
        :param record_delimiter: Delimiter to separate records in outputs.
        :param completion_delimiter: To mark the end of the output.
        :param max_gleaning: the extra max num to call LLM to glean entities
            and relations.
        :param continue_prompt: the prompt for gleaning entities and
            relations.
        :param if_loop_prompt: the prompt to determine whether to stop
            gleaning.
        :param entity_pattern: Regular expression for parsing entity record.
        :param relation_pattern: Regular expression for parsing relation
            record.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES

        self.entity_key = entity_key
        self.relation_key = relation_key

        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.tuple_delimiter = tuple_delimiter or self.DEFAULT_TUPLE_DELIMITER
        self.record_delimiter = record_delimiter or self.DEFAULT_RECORD_DELIMITER
        self.completion_delimiter = completion_delimiter or self.DEFAULT_COMPLETION_DELIMITER
        self.max_gleaning = max_gleaning
        self.continue_prompt = continue_prompt or self.DEFAULT_CONTINUE_PROMPT
        self.if_loop_prompt = if_loop_prompt or self.DEFAULT_IF_LOOP_PROMPT
        self.entity_pattern = entity_pattern or self.DEFAULT_ENTITY_PATTERN
        self.relation_pattern = relation_pattern or self.DEFAULT_RELATION_PATTERN

        self.sampling_params = sampling_params
        self.model_key = prepare_model(
            model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
        )

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output):
        entities, relations = [], []

        def remove_outer_quotes(text):
            if not text:
                return text
            if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
                return text[1:-1]
            else:
                return text

        def split_by_tuple_delimiter(record):
            items = record.split(self.tuple_delimiter)
            items = [remove_outer_quotes(item.strip()) for item in items]
            items = [item.strip() for item in items if item.strip()]
            return tuple(items)

        entity_pattern = re.compile(self.entity_pattern, re.VERBOSE | re.DOTALL)
        matches = entity_pattern.findall(raw_output)
        for record in matches:
            items = split_by_tuple_delimiter(record)
            if len(items) != 3:
                continue
            entities.append(items)
        entities = list(set(entities))
        entities = [
            {MetaKeys.entity_name: e[0], MetaKeys.entity_type: e[1], MetaKeys.entity_description: e[2]}
            for e in entities
        ]

        relation_pattern = re.compile(self.relation_pattern, re.VERBOSE | re.DOTALL)
        matches = relation_pattern.findall(raw_output)
        for record in matches:
            items = split_by_tuple_delimiter(record)
            if len(items) != 5 or not is_float(items[4]):
                continue
            relations.append(items)
        relations = list(set(relations))
        relations = [
            {
                MetaKeys.source_entity: r[0],
                MetaKeys.target_entity: r[1],
                MetaKeys.relation_description: r[2],
                MetaKeys.relation_keywords: split_text_by_punctuation(r[3]),
                MetaKeys.relation_strength: float(r[4]),
            }
            for r in relations
        ]

        return entities, relations

    def add_message(self, messages, role, content):
        return messages + [{"role": role, "content": content}]

    def light_rag_extraction(self, messages, rank=None):
        client = get_model(self.model_key, rank=rank)

        final_result = client(messages, **self.sampling_params)
        history = self.add_message(messages, "assistant", final_result)

        for glean_index in range(self.max_gleaning):
            messages = self.add_message(history, "user", self.continue_prompt)
            glean_result = client(messages, **self.sampling_params)
            history = self.add_message(messages, "assistant", glean_result)
            final_result += glean_result

            if glean_index == self.max_gleaning - 1:
                break

            messages = self.add_message(history, "user", self.if_loop_prompt)
            if_loop_result = client(messages, **self.sampling_params)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        return final_result

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.entity_key in sample[Fields.meta] and self.relation_key in sample[Fields.meta]:
            return sample

        input_prompt = self.prompt_template.format(
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
            entity_types=", ".join(self.entity_types),
            input_text=sample[self.text_key],
        )
        messages = [{"role": "user", "content": input_prompt}]

        entities = [{MetaKeys.entity_name: "", MetaKeys.entity_type: "", MetaKeys.entity_description: ""}]
        relations = [
            {
                MetaKeys.source_entity: "",
                MetaKeys.target_entity: "",
                MetaKeys.relation_description: "",
                MetaKeys.relation_keywords: np.array([], dtype=str),
                MetaKeys.relation_strength: 0.0,
            }
        ]
        for _ in range(self.try_num):
            try:
                result = self.light_rag_extraction(messages, rank=rank)
                cur_entities, cur_relations = self.parse_output(result)
                if len(cur_entities) > 0:
                    entities = cur_entities
                    if len(cur_relations) > 0:
                        relations = cur_relations
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        sample[Fields.meta][self.entity_key] = entities
        sample[Fields.meta][self.relation_key] = relations
        return sample
