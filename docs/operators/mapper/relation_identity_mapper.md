# relation_identity_mapper

Identify the relation between two entities in a given text.

This operator uses an API model to analyze the relationship between two specified entities in the text. It constructs a prompt with the provided system and input templates, then sends it to the API model for analysis. The output is parsed using a regular expression to extract the relationship. If the two entities are the same, the relationship is identified as "another identity." The result is stored in the meta field under the key 'role_relation' by default. The operator retries the API call up to a specified number of times in case of errors. If `drop_text` is set to True, the original text is removed from the sample after processing.

识别给定文本中两个实体之间的关系。

此算子使用 API 模型分析文本中两个指定实体之间的关系。它使用提供的系统和输入模板构建提示，然后发送给 API 模型进行分析。输出通过正则表达式解析以提取关系。如果两个实体相同，则关系被识别为 "another identity"。结果默认存储在 meta 字段的 'role_relation' 键下。算子在出现错误时最多重试 API 调用指定次数。如果 `drop_text` 设置为 True，在处理后将从样本中移除原始文本。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `source_entity` | <class 'str'> | `None` | The source entity of the relation to be identified. |
| `target_entity` | <class 'str'> | `None` | The target entity of the relation to be identified. |
| `output_key` | <class 'str'> | `'role_relation'` | The output key in the meta field in the samples. It is 'role_relation' in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt_template` | typing.Optional[str] | `None` | System prompt template for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `output_pattern_template` | typing.Optional[str] | `None` | Regular expression template for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/relation_identity_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_relation_identity_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)