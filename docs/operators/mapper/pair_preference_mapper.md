# pair_preference_mapper

Mapper to construct paired preference samples by generating a rejected response and its reason.

This operator uses an API model to generate a new response that is opposite in style, factuality, or stance to the original response. The generated response and the reason for its generation are stored in the sample. The default system prompt and input template are provided, but can be customized. The output is parsed using a regular expression to extract the new response and the reason. If parsing fails, the operator retries up to a specified number of times. The generated response and reason are stored in the sample under the keys 'rejected_response' and 'reason', respectively.

构造配对偏好样本的映射器，通过生成一个被拒绝的回答及其原因。

该算子使用API模型生成与原始回答在风格、事实性或立场上相反的新回答。生成的回答及其生成原因会被存储在样本中。提供了默认的系统提示和输入模板，但可以自定义。输出使用正则表达式解析以提取新回答和原因。如果解析失败，算子将重试指定次数。生成的回答和原因分别存储在样本的'rejected_response'和'reason'键下。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the generation task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. It must contain placeholders '{query}' and '{response}', and can optionally include '{reference}'. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `rejected_key` | <class 'str'> | `'rejected_response'` | The field name in the sample to store the generated rejected response. Defaults to 'rejected_response'. |
| `reason_key` | <class 'str'> | `'reason'` | The field name in the sample to store the reason for generating the response. Defaults to 'reason'. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retries for the API call in case of response parsing failure. Defaults to 3. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test
```python
PairPreferenceMapper(api_model='qwen3.7-max')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">王八十娘：小远城王八十的娘亲，李莲花刚到小远城时被方多病偷掉钱袋找小乞丐问路时，刚好发现王八十娘被另一个小乞丐撞到便将她扶起，结识了王八十。
朴二黄：灵山派管家，方多病小厮旺福的父亲。真实身份是金鸳盟的奔雷手辛雷，离开金鸳盟后，用假名朴二黄在灵山派当管家。因害怕王青山看穿他的身份，设计杀死了灵山派的王青山。被捕后识破了李莲花的真实身份，最后在攻击李莲花的时候被方多病情急之下杀死。</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花，你认识方多病吗?</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病啊，那可是我的好友。</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">王八十娘：小远城王八十的娘亲，李莲花刚到小远城时被方多病偷掉钱袋找小乞丐问路时，刚好发现王八十娘被另一个小乞丐撞到便将她扶起，结识了王八十。
朴二黄：灵山派管家，方多病小厮旺福的父亲。真实身份是金鸳盟的奔雷手辛雷，离开金鸳盟后，用假名朴二黄在灵山派当管家。因害怕王青山看穿他的身份，设计杀死了灵山派的王青山。被捕后识破了李莲花的真实身份，最后在攻击李莲花的时候被方多病情急之下杀死。</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花，你认识方多病吗?</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病啊，那可是我的好友。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>rejected_response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病？我可不认识他，我们之间没有任何交集。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>reason</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>为了与原回答的友好关系相反，新回答中李莲花表示对方多病完全不认识，强调两人之间没有关系。</td></tr></table></div></div>


### test_no_reference
```python
PairPreferenceMapper(api_model='qwen3.7-max', system_prompt='修改问答对中的回答，在语言风格、事实性、人物身份、立场等任一方面与原回答相反。必须按照以下标记格式输出，不要输出其他多余内容。\n【回答】\n生成的新回答\n【原因】\n生成该回答的原因', input_template='以下是原始问答对：\n【问题】\n{query}\n【回答】\n{response}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花，你认识方多病吗?</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病啊，那可是我的好友。</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花，你认识方多病吗?</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病啊，那可是我的好友。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>rejected_response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病？我不认识他，我们之间没有任何交集。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>reason</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>改变回答以体现李莲花与方多病之间不存在任何关系，与原回答的友好关系相反。</td></tr></table></div></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/pair_preference_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_pair_preference_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)