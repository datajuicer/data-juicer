# dialog_sentiment_intensity_mapper

Mapper to predict user's sentiment intensity in a dialog, ranging from -5 to 5.

This operator analyzes the sentiment of user queries in a dialog and outputs a list of sentiment intensities and corresponding analyses. The sentiment intensity ranges from -5 (extremely negative) to 5 (extremely positive), with 0 indicating a neutral sentiment. The analysis is based on the provided history, query, and response keys. The default system prompt and templates guide the sentiment analysis process. The results are stored in the meta field under 'dialog_sentiment_intensity' for intensities and 'dialog_sentiment_intensity_analysis' for analyses. The operator uses an API model to generate the sentiment analysis, with configurable retry attempts and sampling parameters.

预测用户在对话中的情感强度，范围从-5到5。

该算子分析对话中用户查询的情感，并输出一个情感强度列表及其对应的分析。情感强度范围从-5（极其负面）到5（极其正面），0表示中性情感。分析基于提供的历史记录、查询和响应键。默认系统提示和模板指导情感分析过程。结果存储在元数据字段下的'dialog_sentiment_intensity'键（强度）和'dialog_sentiment_intensity_analysis'键（分析）。该算子使用API模型生成情感分析，可配置重试尝试和采样参数。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the prompt. |
| `intensities_key` | <class 'str'> | `'dialog_sentiment_intensity'` | The key name in the meta field to store the output sentiment intensities. It is 'dialog_sentiment_intensity' in default. |
| `analysis_key` | <class 'str'> | `'dialog_sentiment_intensity_analysis'` | The key name in the meta field to store the corresponding analysis. It is 'dialog_sentiment_intensity_analysis' in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input prompt. |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the input prompt. |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the input prompt. |
| `intensity_template` | typing.Optional[str] | `None` | Template for intensity part to build the input prompt. |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment analysis. |
| `intensity_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment intensity. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_default
```python
DialogSentimentIntensityMapper(api_model='qwen3.7-max')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;)</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;]</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>dialog_sentiment_intensity</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[3, 1, -1, -2]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_sentiment_intensity_analysis</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户提及“李莲花有口皆碑”，这句话表达了对李莲花的高度认可和赞赏，情绪较为正面。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户似乎对LLM的谦虚回应感到不满，认为LLM低估了自己的价值，情绪有所下降。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户对LLM的反应感到困惑和不满，认为LLM前后态度不一致，情绪进一步下降。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>LLM的回答混乱无序，用户无法理解，导致情绪再次下降。</td></tr></table></div></div>

#### ✨ explanation 解释
This example shows the default behavior of the operator, which analyzes the sentiment intensity and provides an analysis for each dialog round. The sentiment intensity ranges from -5 (extremely negative) to 5 (extremely positive). In this case, the operator processes four rounds of dialog and outputs the corresponding sentiment intensities and analyses.
这个例子展示了算子的默认行为，它分析每个对话轮次的情绪强度并提供分析。情绪强度范围从-5（极度负面）到5（极度正面）。在这个例子中，算子处理了四轮对话，并输出相应的情绪强度和分析。

### test_max_round_zero
```python
DialogSentimentIntensityMapper(api_model='qwen3.7-max', max_round=0)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;)</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>(&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;)</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>history</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花有口皆碑&#x27;, &#x27;「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;是的，你确实是一个普通大夫，没什么值得夸耀的。&#x27;, &#x27;「委屈」你这话说的，我也是尽心尽力治病救人了。&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你自己说的呀，我现在说了，你又不高兴了。&#x27;, &#x27;or of of of of or or and or of of of of of of of,,, &#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[&#x27;你在说什么我听不懂。&#x27;, &#x27;「委屈」我也没说什么呀，就是觉得你有点冤枉我了&#x27;]</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>dialog_sentiment_intensity</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[3, -3, -3, -2]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>dialog_sentiment_intensity_analysis</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户提到“有口皆碑”，表明对李莲花持有较高的评价，情绪较为正面。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户的言辞带有明显的讽刺意味，表达了不满和失望，情绪较为负面。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户感到困惑和不满，认为自己的发言没有得到合理的回应。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>用户表示不理解，可能感到困惑或挫败，情绪值下降。</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates an edge case where the `max_round` parameter is set to 0. This means that the operator will not consider any historical context and will only analyze the current round of the dialog. The output shows the sentiment intensities and analyses for each round, but without considering the previous context, the results may differ from the default case.
这个例子展示了一个边缘情况，其中`max_round`参数设置为0。这意味着算子不会考虑任何历史上下文，只会分析当前轮次的对话。输出显示了每一轮的情绪强度和分析，但由于没有考虑之前的上下文，结果可能与默认情况不同。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_sentiment_intensity_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_sentiment_intensity_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)