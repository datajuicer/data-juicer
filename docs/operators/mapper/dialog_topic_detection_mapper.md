# dialog_topic_detection_mapper

Generates user's topic labels and analysis in a dialog.

This operator processes a dialog to detect and label the topics discussed by the user. It takes input from `history_key`, `query_key`, and `response_key` and outputs lists of labels and analysis for each query in the dialog. The operator uses a predefined system prompt and templates to build the input prompt for the API call. It supports customizing the system prompt, templates, and patterns for parsing the API response. The results are stored in the `meta` field under the keys specified by `labels_key` and `analysis_key`. If these keys already exist in the `meta` field, the operator skips processing. The operator retries the API call up to `try_num` times in case of errors.

在对话中生成用户的话题标签和分析。

该算子处理对话以检测并标记用户讨论的话题。它从`history_key`、`query_key`和`response_key`获取输入，并为对话中的每个查询输出标签和分析列表。该算子使用预定义的系统提示和模板来构建API调用的输入提示。它支持自定义系统提示、模板和模式以解析API响应。结果存储在`meta`字段下的`labels_key`和`analysis_key`指定的键下。如果这些键已经存在于`meta`字段中，该算子将跳过处理。该算子在出现错误时最多重试`try_num`次API调用。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `topic_candidates` | typing.Optional[typing.List[str]] | `None` | The output topic candidates. Use open-domain topic labels if it is None. |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the prompt. |
| `labels_key` | <class 'str'> | `'dialog_topic_labels'` | The key name in the meta field to store the output labels. It is 'dialog_topic_labels' in default. |
| `analysis_key` | <class 'str'> | `'dialog_topic_labels_analysis'` | The key name in the meta field to store the corresponding analysis. It is 'dialog_topic_labels_analysis' in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input prompt. |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the input prompt. |
| `candidate_template` | typing.Optional[str] | `None` | Template for topic candidates to build the input prompt. |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the input prompt. |
| `labels_template` | typing.Optional[str] | `None` | Template for labels part to build the input prompt. |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return topic analysis. |
| `labels_pattern` | typing.Optional[str] | `None` | Pattern to parse the return topic labels. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_default
```python
DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;)</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;]</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>dialog_topic_labels</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;人物评价&#x27;, &#x27;人物评价&#x27;, &#x27;角色扮演/对话互动&#x27;, &#x27;技术问题/沟通障碍&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_topic_labels_analysis</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户提到“李莲花”，但没有提供足够的背景信息来确定具体的人物或作品。不过从“有口皆碑”一词来看，李莲花可能是一位受人尊敬或广受好评的人物。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户再次提到“李莲花”并回应LLM的回答，确认了李莲花是一个普通大夫，且没有什么值得夸耀的。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户和LLM之间的对话似乎是在角色扮演或模拟某种情景，用户提到李莲花自谦为普通大夫，而LLM则以李莲花的身份回应，表现出一种委屈的情绪。用户再次回应，指出LLM之前的话，形成了一种互动。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户表示听不懂LLM的回复，可能是LLM出现了技术问题，导致输出的内容无法理解。</td></tr></table></div></div>

#### ✨ explanation 解释
This example uses the default settings of the DialogTopicDetectionMapper operator to detect and label the topics in a conversation. The operator processes each round of the dialog, identifying the main topics discussed and providing an analysis for each. The output includes a list of topic labels and a corresponding analysis for each round of the dialog.
这个例子使用了DialogTopicDetectionMapper算子的默认设置来检测和标注对话中的主题。算子处理对话中的每一轮，识别讨论的主要话题，并为每一轮提供分析。输出包括每轮对话的主题标签列表和相应的分析。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_topic_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_topic_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)