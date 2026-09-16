# extract_event_mapper

Extracts events and relevant characters from the text.

This operator uses an API model to summarize the text into multiple events and extract the relevant characters for each event. The summary and character extraction follow a predefined format. The operator retries the API call up to a specified number of times if there is an error. The extracted events and characters are stored in the meta field of the samples. If no events are found, the original samples are returned. The operator can optionally drop the original text after processing.

从文本中提取事件和相关角色。

此算子使用 API 模型将文本总结为多个事件，并提取每个事件的相关角色。摘要和角色提取遵循预定义的格式。如果出现错误，该算子将重试 API 调用最多指定次数。提取的事件和角色存储在样本的 meta 字段中。如果没有找到事件，则返回原始样本。该算子可以选择在处理后删除原始文本。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `event_desc_key` | <class 'str'> | `'event_description'` | The key name to store the event descriptions |
| `relevant_char_key` | <class 'str'> | `'relevant_characters'` | The field name to store the relevant |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_event_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_event_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)