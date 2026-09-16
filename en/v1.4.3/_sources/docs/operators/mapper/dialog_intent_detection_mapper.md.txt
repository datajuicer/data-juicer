# dialog_intent_detection_mapper

Generates user's intent labels in a dialog by analyzing the history, query, and response.

This operator processes a dialog to identify and label the user's intent. It uses a predefined system prompt and templates to build input prompts for an API call. The API model (e.g., GPT-4) is used to analyze the dialog and generate intent labels and analysis. The results are stored in the meta field under 'dialog_intent_labels' and 'dialog_intent_labels_analysis'. The operator supports customizing the system prompt, templates, and patterns for parsing the API response. If the intent candidates are provided, they are included in the input prompt. The operator retries the API call up to a specified number of times if there are errors.

通过分析历史记录、查询和响应，在对话中生成用户的意图标签。

此算子处理对话以识别并标记用户的意图。它使用预定义的系统提示和模板构建API调用的输入提示。使用API模型（例如GPT-4）分析对话并生成意图标签和分析。结果存储在元字段下的'dialog_intent_labels'和'dialog_intent_labels_analysis'中。算子支持自定义系统提示、模板和解析API响应的模式。如果提供了意图候选，则将其包含在输入提示中。如果出现错误，算子将重试API调用最多指定次数。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `intent_candidates` | typing.Optional[typing.List[str]] | `None` | The output intent candidates. Use the |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the |
| `labels_key` | <class 'str'> | `'dialog_intent_labels'` | The key name in the meta field to store the |
| `analysis_key` | <class 'str'> | `'dialog_intent_labels_analysis'` | The key name in the meta field to store the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the |
| `candidate_template` | typing.Optional[str] | `None` | Template for intent candidates to |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the |
| `labels_template` | typing.Optional[str] | `None` | Template for labels to build the |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return intent |
| `labels_pattern` | typing.Optional[str] | `None` | Pattern to parse the return intent |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_intent_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_intent_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)