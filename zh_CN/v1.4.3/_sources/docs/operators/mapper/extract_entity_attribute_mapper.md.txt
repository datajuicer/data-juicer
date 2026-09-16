# extract_entity_attribute_mapper

Extracts attributes for given entities from the text and stores them in the sample's metadata.

This operator uses an API model to extract specified attributes for given entities from the input text. It constructs prompts based on provided templates and parses the model's output to extract attribute descriptions and supporting text. The extracted data is stored in the sample's metadata under the specified keys. If the required metadata fields already exist, the operator skips processing for that sample. The operator retries the API call and parsing up to a specified number of times in case of errors. The default system prompt, input template, and parsing patterns are used if not provided.

从文本中提取给定实体的属性，并将其存储在样本的元数据中。

该算子使用API模型从输入文本中提取给定实体的指定属性。它基于提供的模板构建提示，并解析模型的输出以提取属性描述和支持文本。提取的数据存储在样本的元数据中指定的键下。如果所需的元数据字段已经存在，该算子将跳过对该样本的处理。该算子在出现错误时最多重试指定次数的API调用和解析。如果没有提供默认系统提示、输入模板和解析模式，则使用默认值。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `query_entities` | typing.List[str] | `[]` | Entity list to be queried. |
| `query_attributes` | typing.List[str] | `[]` | Attribute list to be queried. |
| `entity_key` | <class 'str'> | `'main_entities'` | The key name in the meta field to store the |
| `attribute_key` | <class 'str'> | `'attributes'` |  |
| `attribute_desc_key` | <class 'str'> | `'attribute_descriptions'` | The key name in the meta field to store |
| `support_text_key` | <class 'str'> | `'attribute_support_texts'` | The key name in the meta field to store |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt_template` | typing.Optional[str] | `None` | System prompt template for the |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `attr_pattern_template` | typing.Optional[str] | `None` | Pattern for parsing the attribute from |
| `demo_pattern` | typing.Optional[str] | `None` |  |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_entity_attribute_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_entity_attribute_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)