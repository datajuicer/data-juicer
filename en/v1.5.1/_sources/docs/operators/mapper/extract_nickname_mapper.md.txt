# extract_nickname_mapper

Extracts nickname relationships in the text using a language model.

This operator uses a language model to identify and extract nickname relationships from the input text. It follows specific instructions to ensure accurate extraction, such as identifying the speaker, the person being addressed, and the nickname used. The extracted relationships are stored in the meta field under the specified key. The operator uses a default system prompt, input template, and output pattern, but these can be customized. The results are parsed and validated to ensure they meet the required format. If the text already contains the nickname information, it is not processed again. The operator retries the API call a specified number of times if an error occurs.

使用语言模型从文本中提取昵称关系。

此算子使用语言模型从输入文本中识别并提取昵称关系。它遵循特定的指令以确保准确提取，例如识别说话者、被称呼的人以及使用的昵称。提取的关系存储在指定键的 meta 字段中。该算子使用默认的系统提示、输入模板和输出模式，但这些可以自定义。结果经过解析和验证以确保符合所需的格式。如果文本已经包含昵称信息，则不再进行处理。如果出现错误，该算子将重试 API 调用指定次数。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `nickname_key` | <class 'str'> | `'nickname'` | The key name to store the nickname relationship in the meta field. It's "nickname" in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test
```python
ExtractNicknameMapper(api_model='qwen2.5-72b-instruct', response_path=None)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△李莲花又指出刚才门框上的痕迹。
△李莲花：门框上也是人的掌痕和爪印。指力能嵌入硬物寸余，七分力道主上，三分力道垫下，还有辅以的爪式，看样子这还有昆仑派的外家功夫。
方多病看着李莲花，愈发生疑os：通过痕迹就能判断出功夫和门派，这绝对只有精通武艺之人才能做到，李莲花你到底是什么人？！
笛飞声环顾四周：有朝月派，还有昆仑派，看来必是一群武林高手在这发生了决斗！
李莲花：如果是武林高手过招，为何又会出现如此多野兽的痕迹。方小宝，你可听过江湖上有什么门派是驯兽来斗？方小宝？方小宝？
方多病回过神：不、不曾听过。
李莲花：还有这些人都去了哪里？
笛飞声：打架不管是输是赢，自然是打完就走。
李莲花摇头：...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (106 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△李莲花又指出刚才门框上的痕迹。
△李莲花：门框上也是人的掌痕和爪印。指力能嵌入硬物寸余，七分力道主上，三分力道垫下，还有辅以的爪式，看样子这还有昆仑派的外家功夫。
方多病看着李莲花，愈发生疑os：通过痕迹就能判断出功夫和门派，这绝对只有精通武艺之人才能做到，李莲花你到底是什么人？！
笛飞声环顾四周：有朝月派，还有昆仑派，看来必是一群武林高手在这发生了决斗！
李莲花：如果是武林高手过招，为何又会出现如此多野兽的痕迹。方小宝，你可听过江湖上有什么门派是驯兽来斗？方小宝？方小宝？
方多病回过神：不、不曾听过。
李莲花：还有这些人都去了哪里？
笛飞声：打架不管是输是赢，自然是打完就走。
李莲花摇头：就算打完便走，但这里是客栈，为何这么多年一直荒在这里，甚至没人来收拾一下？
笛飞声：闹鬼？这里死过这么多人，楼下又画了那么多符，所以不敢进来？
△这时，梁上又出现有东西移动的声响，李莲花、笛飞声都猛然回头看去。
</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△李莲花又指出刚才门框上的痕迹。
△李莲花：门框上也是人的掌痕和爪印。指力能嵌入硬物寸余，七分力道主上，三分力道垫下，还有辅以的爪式，看样子这还有昆仑派的外家功夫。
方多病看着李莲花，愈发生疑os：通过痕迹就能判断出功夫和门派，这绝对只有精通武艺之人才能做到，李莲花你到底是什么人？！
笛飞声环顾四周：有朝月派，还有昆仑派，看来必是一群武林高手在这发生了决斗！
李莲花：如果是武林高手过招，为何又会出现如此多野兽的痕迹。方小宝，你可听过江湖上有什么门派是驯兽来斗？方小宝？方小宝？
方多病回过神：不、不曾听过。
李莲花：还有这些人都去了哪里？
笛飞声：打架不管是输是赢，自然是打完就走。
李莲花摇头：...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (106 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△李莲花又指出刚才门框上的痕迹。
△李莲花：门框上也是人的掌痕和爪印。指力能嵌入硬物寸余，七分力道主上，三分力道垫下，还有辅以的爪式，看样子这还有昆仑派的外家功夫。
方多病看着李莲花，愈发生疑os：通过痕迹就能判断出功夫和门派，这绝对只有精通武艺之人才能做到，李莲花你到底是什么人？！
笛飞声环顾四周：有朝月派，还有昆仑派，看来必是一群武林高手在这发生了决斗！
李莲花：如果是武林高手过招，为何又会出现如此多野兽的痕迹。方小宝，你可听过江湖上有什么门派是驯兽来斗？方小宝？方小宝？
方多病回过神：不、不曾听过。
李莲花：还有这些人都去了哪里？
笛飞声：打架不管是输是赢，自然是打完就走。
李莲花摇头：就算打完便走，但这里是客栈，为何这么多年一直荒在这里，甚至没人来收拾一下？
笛飞声：闹鬼？这里死过这么多人，楼下又画了那么多符，所以不敢进来？
△这时，梁上又出现有东西移动的声响，李莲花、笛飞声都猛然回头看去。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>nickname</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'><div style='margin:2px 0 6px 0; padding-left:8px;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花你</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_keywords</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;nickname&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_source_entity</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_strength</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>None</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_target_entity</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花</td></tr></table></div></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'><div style='margin:2px 0 6px 0; padding-left:8px;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方小宝</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_keywords</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;nickname&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_source_entity</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李莲花</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_strength</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>None</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:50px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relation_target_entity</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>方多病</td></tr></table></div></td></tr></table></div></div>

#### ✨ explanation 解释
The operator identifies and extracts nickname relationships from the input text. In this example, it correctly identifies that '方多病' refers to '李莲花' as '李莲花你', and '李莲花' refers to '方多病' as '方小宝'. The output data shows these relationships in a structured format within the meta field.
算子从输入文本中识别并提取昵称关系。在这个例子中，它正确地识别出“方多病”称呼“李莲花”为“李莲花你”，以及“李莲花”称呼“方多病”为“方小宝”。输出数据以结构化格式在meta字段中展示了这些关系。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_nickname_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_nickname_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)