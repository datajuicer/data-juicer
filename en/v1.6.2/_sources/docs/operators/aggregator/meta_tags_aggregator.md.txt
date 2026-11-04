# meta_tags_aggregator

Merge similar meta tags into a single, unified tag.

This operator aggregates and consolidates similar meta tags from the input data. It can handle two scenarios:
- When a set of target tags is provided, it maps the original tags to these predefined categories. If a "miscellaneous" or "other" category is included, any tags that do not fit into the specified categories are grouped under this label.
- When no target tags are provided, it generates reasonable categories based on the similarity and frequency of the input tags.

The operator uses a language model (default: gpt-4o) to analyze and merge the tags. The system prompt, input template, and output pattern can be customized. The aggregated tags are then updated in the input sample's metadata.

将相似的元标签合并为一个统一的标签。

该算子聚合并整合输入数据中的相似元标签。它可以处理两种情况：
- 当提供了一组目标标签时，它将原始标签映射到这些预定义类别。如果包含“杂项”或“其他”类别，则任何不符合指定类别的标签将被归入此类别。
- 当没有提供目标标签时，它根据输入标签的相似性和频率生成合理的类别。

该算子使用语言模型（默认：gpt-4o）来分析和合并标签。可以自定义系统提示、输入模板和输出模式。聚合后的标签将更新到输入样本的元数据中。

Type 算子类型: **aggregator**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `meta_tag_key` | <class 'str'> | `'dialog_sentiment_labels'` | The key of the meta tag to be mapped. |
| `target_tags` | typing.Optional[typing.List[str]] | `None` | The tags that is supposed to be mapped to. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | The system prompt. |
| `input_template` | typing.Optional[str] | `None` | The input template. |
| `target_tag_template` | typing.Optional[str] | `None` | The tap template for target tags. |
| `tag_template` | typing.Optional[str] | `None` | The tap template for each tag and its frequency. |
| `output_pattern` | typing.Optional[str] | `None` | The output pattern. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/aggregator/meta_tags_aggregator.py)
- [unit test 单元测试](../../../tests/ops/aggregator/test_meta_tags_aggregator.py)
- [Return operator list 返回算子列表](../../Operators.md)