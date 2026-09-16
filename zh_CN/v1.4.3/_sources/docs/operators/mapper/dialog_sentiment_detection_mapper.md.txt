# dialog_sentiment_detection_mapper

Generates sentiment labels and analysis for user queries in a dialog.

This operator processes a dialog to detect and label the sentiments expressed by the user. It uses the provided history, query, and response keys to construct prompts for an API call. The API returns sentiment analysis and labels, which are then parsed and stored in the sample's metadata under the 'dialog_sentiment_labels' and 'dialog_sentiment_labels_analysis' keys. The operator supports custom templates and patterns for prompt construction and output parsing. If no sentiment candidates are provided, it uses open-domain sentiment labels. The operator retries the API call up to a specified number of times in case of errors.

为用户查询在对话中生成情感标签和分析。

该算子处理对话以检测并标记用户表达的情感。它使用提供的历史记录、查询和响应键来构建API调用的提示。API返回情感分析和标签，然后解析并将结果存储在样本的元数据中的'dialog_sentiment_labels'和'dialog_sentiment_labels_analysis'键下。该算子支持自定义模板和模式用于提示构建和输出解析。如果没有提供情感候选，则使用开放领域情感标签。该算子在出现错误时最多重试指定次数的API调用。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `sentiment_candidates` | typing.Optional[typing.List[str]] | `None` | The output sentiment candidates. Use |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the |
| `labels_key` | <class 'str'> | `'dialog_sentiment_labels'` | The key name in the meta field to store the |
| `analysis_key` | <class 'str'> | `'dialog_sentiment_labels_analysis'` | The key name in the meta field to store the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the |
| `candidate_template` | typing.Optional[str] | `None` | Template for sentiment candidates to |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the |
| `labels_template` | typing.Optional[str] | `None` | Template for labels part to build the |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment |
| `labels_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_sentiment_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_sentiment_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)