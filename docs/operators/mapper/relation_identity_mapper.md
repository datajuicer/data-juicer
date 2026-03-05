# relation_identity_mapper

Identify the relation between two entities in a given text.

This operator uses an API model to analyze the relationship between two specified entities in the text. It constructs a prompt with the provided system and input templates, then sends it to the API model for analysis. The output is parsed using a regular expression to extract the relationship. If the two entities are the same, the relationship is identified as "another identity." The result is stored in the meta field under the key 'role_relation' by default. The operator retries the API call up to a specified number of times in case of errors. If `drop_text` is set to True, the original text is removed from the sample after processing.

识别给定文本中两个实体之间的关系。

此算子使用 API 模型分析文本中两个指定实体之间的关系。它使用提供的系统和输入模板构建提示，然后发送给 API 模型进行分析。输出通过正则表达式解析以提取关系。如果两个实体相同，则关系被识别为 "another identity"。结果默认存储在 meta 字段的 'role_relation' 键下。算子在出现错误时最多重试 API 调用指定次数。如果 `drop_text` 设置为 True，在处理后将从样本中移除原始文本。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `source_entity` | <class 'str'> | `None` | The source entity of the relation to be identified. |
| `target_entity` | <class 'str'> | `None` | The target entity of the relation to be identified. |
| `output_key` | <class 'str'> | `'role_relation'` | The output key in the meta field in the samples. It is 'role_relation' in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt_template` | typing.Optional[str] | `None` | System prompt template for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `output_pattern_template` | typing.Optional[str] | `None` | Regular expression template for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_default
```python
RelationIdentityMapper(api_model='qwen2.5-72b-instruct', source_entity='李莲花', target_entity='方多病', output_key='role_relation')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (232 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋。
在采莲庄，他解决了新娘溺水案，找到了狮魂的线索，并在南门园圃挖出单孤刀的药棺。在玉楼春的案件中，李莲花和方多病揭露了玉楼春的阴谋，救出了被拐的清儿。
在石寿村，他们发现了柔肠玉酿的秘密，并救出了被控制的武林高手。李莲花与方多病在白水园设下机关，救出方多病的母亲何晓惠，并最终在云隐山找到了治疗碧茶之毒的方法。
在天机山庄，他揭露了单孤刀的野心，救出了被控制的大臣。在皇宫，李莲花与方多病揭露了魔僧... [truncated, total 900 chars]</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>__dj__meta__</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>{}</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (232 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋。
在采莲庄，他解决了新娘溺水案，找到了狮魂的线索，并在南门园圃挖出单孤刀的药棺。在玉楼春的案件中，李莲花和方多病揭露了玉楼春的阴谋，救出了被拐的清儿。
在石寿村，他们发现了柔肠玉酿的秘密，并救出了被控制的武林高手。李莲花与方多病在白水园设下机关，救出方多病的母亲何晓惠，并最终在云隐山找到了治疗碧茶之毒的方法。
在天机山庄，他揭露了单孤刀的野心，救出了被控制的大臣。在皇宫，李莲花与方多病揭露了魔僧... [truncated, total 900 chars]</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>role_relation</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>徒弟兼好友</td></tr></table></div></div>


### test_rename_key
```python
RelationIdentityMapper(api_model='qwen2.5-72b-instruct', source_entity='李莲花', target_entity='方多病', output_key='output')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (232 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋。
在采莲庄，他解决了新娘溺水案，找到了狮魂的线索，并在南门园圃挖出单孤刀的药棺。在玉楼春的案件中，李莲花和方多病揭露了玉楼春的阴谋，救出了被拐的清儿。
在石寿村，他们发现了柔肠玉酿的秘密，并救出了被控制的武林高手。李莲花与方多病在白水园设下机关，救出方多病的母亲何晓惠，并最终在云隐山找到了治疗碧茶之毒的方法。
在天机山庄，他揭露了单孤刀的野心，救出了被控制的大臣。在皇宫，李莲花与方多病揭露了魔僧... [truncated, total 900 chars]</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>__dj__meta__</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>{}</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (232 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋。
在采莲庄，他解决了新娘溺水案，找到了狮魂的线索，并在南门园圃挖出单孤刀的药棺。在玉楼春的案件中，李莲花和方多病揭露了玉楼春的阴谋，救出了被拐的清儿。
在石寿村，他们发现了柔肠玉酿的秘密，并救出了被控制的武林高手。李莲花与方多病在白水园设下机关，救出方多病的母亲何晓惠，并最终在云隐山找到了治疗碧茶之毒的方法。
在天机山庄，他揭露了单孤刀的野心，救出了被控制的大臣。在皇宫，李莲花与方多病揭露了魔僧... [truncated, total 900 chars]</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>output</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>徒弟兼好友</td></tr></table></div></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/relation_identity_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_relation_identity_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)