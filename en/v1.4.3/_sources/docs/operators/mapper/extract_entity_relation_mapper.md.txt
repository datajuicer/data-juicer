# extract_entity_relation_mapper

Extracts entities and relations from text to build a knowledge graph.

- Identifies entities based on specified types and extracts their names, types, and descriptions.
- Identifies relationships between the entities, including source and target entities, relationship descriptions, keywords, and strength scores.
- Uses a Hugging Face tokenizer and a predefined prompt template to guide the extraction process.
- Outputs entities and relations in a structured format, using delimiters for separation.
- Caches the results in the sample's metadata under the keys 'entity' and 'relation'.
- Supports multiple retries and gleaning to ensure comprehensive extraction.
- The default entity types include 'organization', 'person', 'geo', and 'event'.

从文本中提取实体和关系以构建知识图谱。

- 根据指定的类型识别实体，并提取它们的名称、类型和描述。
- 识别实体之间的关系，包括源实体和目标实体、关系描述、关键词和强度分数。
- 使用 Hugging Face 分词器和预定义的提示模板来指导提取过程。
- 以结构化格式输出实体和关系，使用分隔符进行分隔。
- 将结果缓存在样本的元数据中，键名为 'entity' 和 'relation'。
- 支持多次重试和搜集以确保全面提取。
- 默认的实体类型包括 'organization'、'person'、'geo' 和 'event'。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `entity_types` | typing.List[str] | `None` | Pre-defined entity types for knowledge graph. |
| `entity_key` | <class 'str'> | `'entity'` | The key name to store the entities in the meta |
| `relation_key` | <class 'str'> | `'relation'` | The field name to store the relations between |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `prompt_template` | typing.Optional[str] | `None` | The template of input prompt. |
| `tuple_delimiter` | typing.Optional[str] | `None` | Delimiter to separate items in outputs. |
| `record_delimiter` | typing.Optional[str] | `None` | Delimiter to separate records in outputs. |
| `completion_delimiter` | typing.Optional[str] | `None` | To mark the end of the output. |
| `max_gleaning` | typing.Annotated[int, Ge(ge=0)] | `1` | the extra max num to call LLM to glean entities |
| `continue_prompt` | typing.Optional[str] | `None` | the prompt for gleaning entities and |
| `if_loop_prompt` | typing.Optional[str] | `None` | the prompt to determine whether to stop |
| `entity_pattern` | typing.Optional[str] | `None` | Regular expression for parsing entity record. |
| `relation_pattern` | typing.Optional[str] | `None` | Regular expression for parsing relation |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_entity_relation_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_entity_relation_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)