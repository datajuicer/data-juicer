# extract_keyword_mapper

Generate keywords for the text.

This operator uses a specified API model to generate high-level keywords that summarize the main concepts, themes, or topics of the input text. The generated keywords are stored in the meta field under the key specified by `keyword_key`. The operator retries the API call up to `try_num` times in case of errors. If `drop_text` is set to True, the original text is removed from the sample after processing. The operator uses a default prompt template and completion delimiter, which can be customized. The output is parsed using a regular expression to extract the keywords.

为文本生成关键词。

此算子使用指定的 API 模型生成高层次的关键词，这些关键词总结了输入文本的主要概念、主题或话题。生成的关键词存储在 meta 字段中，键名为 `keyword_key`。如果出现错误，该算子将重试 API 调用最多 `try_num` 次。如果 `drop_text` 设置为 True，则在处理后从样本中删除原始文本。该算子使用默认的提示模板和完成分隔符，这些可以自定义。输出通过正则表达式解析以提取关键词。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `keyword_key` | <class 'str'> | `'keyword'` | The key name to store the keywords in the meta |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `prompt_template` | typing.Optional[str] | `None` | The template of input prompt. |
| `completion_delimiter` | typing.Optional[str] | `None` | To mark the end of the output. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing keywords. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_keyword_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_keyword_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)