import random

from itertools import chain
from loguru import logger
from collections import Counter

from data_juicer.ops.aggregator import NestedAggregator
from data_juicer.ops.aggregator import EntityAttributeAggregator
from data_juicer.ops.mapper import RelationIdentityMapper
from data_juicer.utils.constant import BatchMetaKeys, Fields, MetaKeys
from data_juicer.core.data import NestedDataset as Dataset


api_model = 'qwen2.5-72b-instruct'

main_entity ="孙悟空"
query_attributes = ["语言风格", "角色性格", "角色武艺和能力"]
system_prompt_key = 'system_prompt'
example_num_limit = 5
max_relevant_roles_num = 5

role_info_template = "# {entity}\n## 身份背景\n{identity}\n## 人物经历\n{experience}"
relation_identity_text_template = """
{source_entity}的信息：
{source_entity_info}
{target_entity}的信息：
{target_entity_info}
{source_entity}对{target_entity}的称呼：{nicknames}
"""

nested_sum = NestedAggregator(
    api_model=api_model,
    try_num=3)

def dedup_sort_val_by_chunk_id(sample, id_key, meta_key):
    chunk_ids = sample[id_key]
    vals = [d[meta_key] for d in sample[Fields.meta]]
    id_to_val = {}
    for id, val in zip(chunk_ids, vals):
        id_to_val[id] = val
    sorted_ids = list(id_to_val.keys())
    sorted_ids.sort()
    sorted_vals = [id_to_val[id] for id in sorted_ids]
    return list(chain(*sorted_vals))

def get_attributes(sample):
    main_entities = dedup_sort_val_by_chunk_id(sample, 'chunk_id', MetaKeys.main_entities)
    attribute_names = dedup_sort_val_by_chunk_id(sample, 'chunk_id', MetaKeys.attributes)
    attribute_descs = dedup_sort_val_by_chunk_id(sample, 'chunk_id', MetaKeys.attribute_descriptions)
    attribute_support_texts = dedup_sort_val_by_chunk_id(sample, 'chunk_id', MetaKeys.attribute_support_texts)
    attributes = {}
    support_texts = {}
    for attr in query_attributes:
        attributes[attr] = []
        support_texts[attr] = []
    for entity, attr_name, attr_desc, sub_support_texts in \
            zip(main_entities, attribute_names, attribute_descs, attribute_support_texts):
        if entity == main_entity and attr_name in query_attributes:
            attributes[attr_name].append(attr_desc)
            support_texts[attr_name].append(sub_support_texts)
    return attributes, support_texts

def get_nicknames(sample):
    nicknames = dedup_sort_val_by_chunk_id(sample, 'chunk_id', MetaKeys.nickname)
    nickname_map = {}
    for nr in nicknames:
        if nr[MetaKeys.source_entity] == main_entity:
            role_name = nr[MetaKeys.target_entity]
            if role_name not in nickname_map:
                nickname_map[role_name] = []
            nickname_map[role_name].append(nr[MetaKeys.relation_description])
    
    max_nums = 3
    for role_name, nickname_list in nickname_map.items():
            th = (len(nickname_list)+1) // 2
            count = Counter(nickname_list)
            sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)
            most_common_nicknames = []
            idx = 0
            while th > 0 and idx < min(len(sorted_items), max_nums):
                most_common_nicknames.append(sorted_items[idx][0])
                th -= sorted_items[idx][1]
                idx += 1
            nickname_map[role_name] = most_common_nicknames
    return nickname_map


def get_system_prompt(sample):

    main_role_identity = sample[Fields.batch_meta]['role_background']
    main_role_experience = sample[Fields.batch_meta]['role_experience']
    attributes, support_texts = get_attributes(sample)
    main_role_character = nested_sum.recursive_summary(attributes['角色性格'])
    main_role_skill = nested_sum.recursive_summary(attributes['角色武艺和能力'])
    main_role_lang_style = nested_sum.recursive_summary(attributes['语言风格'])
    lang_style_examples = list(chain(*support_texts['语言风格']))
    lang_style_example_num = min(example_num_limit, len(lang_style_examples))
    lang_style_examples = random.sample(lang_style_examples, lang_style_example_num)

    main_role_info = role_info_template.format(
            entity=main_entity,
            identity=main_role_identity,
            experience=main_role_experience
        )

    nicknames = get_nicknames(sample)

    relation_detail = ""
    relevant_roles = sample[Fields.batch_meta]['important_relevant_roles']
    for role_name in relevant_roles[:max_relevant_roles_num]:
        if role_name == main_entity:
            continue

        cur_sample = {k: sample[k] for k in sample if k != Fields.batch_meta}

        dataset = Dataset.from_list([cur_sample])
        # get sub role identity
        op = EntityAttributeAggregator(
            api_model=api_model,
            entity=role_name,
            attribute='身份背景',
            input_key='event_description',
            output_key='role_background',
            word_limit=30
        )
        dataset = op.run(dataset)
        role_identity = dataset[0][Fields.batch_meta]['role_background'].replace('\n', '')

        # get sub role experience
        op = EntityAttributeAggregator(
            api_model=api_model,
            entity=role_name,
            attribute='主要经历',
            input_key='event_description',
            output_key='role_experience',
            word_limit=100
        )
        dataset = op.run(dataset)
        role_experience = dataset[0][Fields.batch_meta]['role_experience'].replace('\n', '')

        # get relation identity with main role
        role_info = role_info_template.format(
            entity=role_name,
            identity=role_identity,
            experience=role_experience
        )
        op = RelationIdentityMapper(
            api_model=api_model,
            source_entity=main_entity,
            target_entity=role_name,
            output_key='relation_identity'
        )
        if role_name in nicknames:
            cur_nicknames = '、'.join(nicknames[role_name])
        else:
            cur_nicknames = role_name
        text = relation_identity_text_template.format(
            source_entity=main_entity,
            source_entity_info=main_role_info,
            target_entity=role_name,
            target_entity_info=role_info,
            nicknames = cur_nicknames
        )
        tmp_sample = {'text': text}
        dataset = Dataset.from_list([tmp_sample])
        dataset = op.run(dataset)
        relation = dataset[0][Fields.meta]['relation_identity']

        relation_detail += f"\n{role_name} (称呼:{cur_nicknames})"
        if relation:
            relation_detail += f"{main_entity}的{relation}。"
        relation_detail += f"{role_identity}{role_experience}".replace('\n', '')
    
    full_system_prompt = f"""扮演{main_entity}与用户进行对话。\n"""
    full_system_prompt += """# 角色身份\n"""
    full_system_prompt += main_role_identity.replace('\n', '')
    full_system_prompt += """\n# 角色经历\n"""
    full_system_prompt += main_role_experience.replace('\n', '')
    full_system_prompt += """\n# 角色性格\n"""
    full_system_prompt += main_role_character.replace('\n', '')
    full_system_prompt += """\n# 角色能力\n"""
    full_system_prompt += main_role_skill.replace('\n', '')

    full_system_prompt += """\n# 人际关系"""
    full_system_prompt += relation_detail

    full_system_prompt += """\n# 语言风格\n"""
    full_system_prompt += main_role_lang_style.replace('\n', '')
    full_system_prompt += f"""\n供参考语言风格的部分{main_entity}台词：\n"""
    full_system_prompt += "\n````\n"
    full_system_prompt += '\n'.join(lang_style_examples)
    full_system_prompt += "\n````\n"  

    logger.info(full_system_prompt)

    sample[system_prompt_key] = full_system_prompt

    return sample