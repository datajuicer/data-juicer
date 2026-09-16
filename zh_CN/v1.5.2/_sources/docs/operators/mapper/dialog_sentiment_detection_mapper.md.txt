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
| `sentiment_candidates` | typing.Optional[typing.List[str]] | `None` | The output sentiment candidates. Use open-domain sentiment labels if it is None. |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the prompt. |
| `labels_key` | <class 'str'> | `'dialog_sentiment_labels'` | The key name in the meta field to store the output labels. It is 'dialog_sentiment_labels' in default. |
| `analysis_key` | <class 'str'> | `'dialog_sentiment_labels_analysis'` | The key name in the meta field to store the corresponding analysis. It is 'dialog_sentiment_labels_analysis' in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input prompt. |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the input prompt. |
| `candidate_template` | typing.Optional[str] | `None` | Template for sentiment candidates to build the input prompt. |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the input prompt. |
| `labels_template` | typing.Optional[str] | `None` | Template for labels part to build the input prompt. |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment analysis. |
| `labels_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment labels. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_default
```python
DialogSentimentDetectionMapper(api_model='qwen2.5-72b-instruct')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;)</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;]</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_sentiment_labels</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>赞赏、肯定</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>讽刺、不满</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>生气、愤怒、不满</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>困惑、不耐烦、烦躁</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_sentiment_labels_analysis</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户提到“李莲花有口皆碑”，这表明用户对李莲花的评价很高，认为她受到了广泛的赞誉和认可，语气中带有赞赏和肯定。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户重复了LLM的话，但语气中似乎带有一些讽刺或不满，可能是因为觉得LLM过于谦虚，没有接受自己的赞美。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户在回应时显得有些生气，可能是因为觉得LLM没有理解自己的本意，反而产生了误解。用户的情绪中包含了一些愤怒和不满。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户表示听不懂LLM说的话，语气中带有困惑和不耐烦，可能是因为LLM的回答毫无意义，让用户感到更加烦躁。</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the default behavior of the operator, which analyzes the sentiment of each dialog in the provided history. The operator uses an API to detect and label the sentiments expressed by the user. The output includes both the sentiment labels (e.g., '赞赏、肯定', '讽刺、不满') and a detailed analysis of each sentiment (e.g., '用户提到“李莲花有口皆碑”，这表明用户对李莲花的评价很高，认为她受到了广泛的赞誉和认可，语气中带有赞赏和肯定。').
此示例展示了算子的默认行为，它分析所提供历史记录中每个对话的情绪。算子使用API来检测并标记用户表达的情绪。输出包括情绪标签（例如，“赞赏、肯定”，“讽刺、不满”）以及对每种情绪的详细分析（例如，“用户提到‘李莲花有口皆碑’，这表明用户对李莲花的评价很高，认为她受到了广泛的赞誉和认可，语气中带有赞赏和肯定。”）。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_sentiment_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_sentiment_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)