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
| `intent_candidates` | typing.Optional[typing.List[str]] | `None` | The output intent candidates. Use the intent labels of the open domain if it is None. |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the prompt. |
| `labels_key` | <class 'str'> | `'dialog_intent_labels'` | The key name in the meta field to store the output labels. It is 'dialog_intent_labels' in default. |
| `analysis_key` | <class 'str'> | `'dialog_intent_labels_analysis'` | The key name in the meta field to store the corresponding analysis. It is 'dialog_intent_labels_analysis' in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input prompt. |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the input prompt. |
| `candidate_template` | typing.Optional[str] | `None` | Template for intent candidates to build the input prompt. |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the input prompt. |
| `labels_template` | typing.Optional[str] | `None` | Template for labels to build the input prompt. |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return intent analysis. |
| `labels_pattern` | typing.Optional[str] | `None` | Pattern to parse the return intent labels. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_default
```python
DialogIntentDetectionMapper(api_model='qwen2.5-72b-instruct')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;)</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;]</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_intent_labels</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>表达观点/寻求反馈</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>表达不同意见/讽刺</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>表达不满/讽刺</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>表达困惑/请求澄清</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_intent_labels_analysis</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户在表达对“李莲花”这一人物或品牌的正面评价，可能是想分享自己的看法或是询问他人对“李莲花”的看法。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户可能是在回应LLM的回答，但语气中带有讽刺或者不赞同，似乎认为李莲花（假设为LLM的角色）谦虚过头了，实际上有很多值得称赞的地方。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户继续以一种带有些许讽刺的语气回应，似乎在指出LLM之前的说法与现在反应之间的矛盾，同时也表达了对LLM反应的不满。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户在表达困惑，对LLM的回复内容无法理解，可能希望得到更清晰的解释或说明。</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the basic usage of the operator, where it analyzes a conversation history to generate intent labels and analysis for each round. The operator uses an API model (qwen2.5-72b-instruct) to process the input data and returns the results in the 'dialog_intent_labels' and 'dialog_intent_labels_analysis' fields. Each round of the conversation is analyzed, and the corresponding intent and analysis are provided.
这个示例展示了算子的基本用法，它分析对话历史以生成每轮的意图标签和分析。算子使用API模型（qwen2.5-72b-instruct）处理输入数据，并在'dialog_intent_labels'和'dialog_intent_labels_analysis'字段中返回结果。每一轮对话都被分析，并提供了相应的意图和分析。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_intent_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_intent_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)