# extract_entity_attribute_mapper

Extracts attributes for given entities from the text and stores them in the sample's metadata.

This operator uses an API model to extract specified attributes for given entities from the input text. It constructs prompts based on provided templates and parses the model's output to extract attribute descriptions and supporting text. The extracted data is stored in the sample's metadata under the specified keys. If the required metadata fields already exist, the operator skips processing for that sample. The operator retries the API call and parsing up to a specified number of times in case of errors. The default system prompt, input template, and parsing patterns are used if not provided.

从文本中提取给定实体的属性，并将其存储在样本的元数据中。

该算子使用API模型从输入文本中提取给定实体的指定属性。它基于提供的模板构建提示，并解析模型的输出以提取属性描述和支持文本。提取的数据存储在样本的元数据中指定的键下。如果所需的元数据字段已经存在，该算子将跳过对该样本的处理。该算子在出现错误时最多重试指定次数的API调用和解析。如果没有提供默认系统提示、输入模板和解析模式，则使用默认值。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `query_entities` | typing.List[str] | `[]` | Entity list to be queried. |
| `query_attributes` | typing.List[str] | `[]` | Attribute list to be queried. |
| `entity_key` | <class 'str'> | `'main_entities'` | The key name in the meta field to store the given main entity for attribute extraction. It's "entity" in default. |
| `attribute_key` | <class 'str'> | `'attributes'` | The key name in the meta field to store the given attribute to be extracted. It's "attribute" in default. |
| `attribute_desc_key` | <class 'str'> | `'attribute_descriptions'` | The key name in the meta field to store the extracted attribute description. It's "attribute_description" in default. |
| `support_text_key` | <class 'str'> | `'attribute_support_texts'` | The key name in the meta field to store the attribute support text extracted from the raw text. It's "support_text" in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt_template` | typing.Optional[str] | `None` | System prompt template for the task. Need to be specified by given entity and attribute. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `attr_pattern_template` | typing.Optional[str] | `None` | Pattern for parsing the attribute from output. Need to be specified by given attribute. |
| `demo_pattern` | typing.Optional[str] | `None` | Pattern for parsing the demonstration from output to support the attribute. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test
```python
ExtractEntityAttributeMapper(api_model='qwen2.5-72b-instruct', query_entities=['李莲花', '方多病'], query_attributes=['语言风格', '角色性格'], response_path=None)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△笛飞声独自坐在莲花楼屋顶上。李莲花边走边悠闲地给马喂草。方多病则走在一侧，却总不时带着怀疑地盯向楼顶的笛飞声。
方多病走到李莲花身侧：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！
李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。
方多病：认识十几年？你上次才说是一面之缘？
李莲花忙圆谎：见得不多，但知根知底。哎，这老马吃得也太多了。
方多病一把夺过李莲花手中的草料：别转移话题！——快说！
李莲花：阿飞啊，脾气不太好，他......这十年也没出过几次门，所以见识短，你不要和他计较。还有他是个武痴，武功深藏不露，你平时别惹他。
方多病：呵...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (352 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△笛飞声独自坐在莲花楼屋顶上。李莲花边走边悠闲地给马喂草。方多病则走在一侧，却总不时带着怀疑地盯向楼顶的笛飞声。
方多病走到李莲花身侧：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！
李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。
方多病：认识十几年？你上次才说是一面之缘？
李莲花忙圆谎：见得不多，但知根知底。哎，这老马吃得也太多了。
方多病一把夺过李莲花手中的草料：别转移话题！——快说！
李莲花：阿飞啊，脾气不太好，他......这十年也没出过几次门，所以见识短，你不要和他计较。还有他是个武痴，武功深藏不露，你平时别惹他。
方多病：呵，阿飞武功高？编瞎话能不能用心点？
李莲花：可都是大实话啊。反正，我和他彼此了解得很。你就别瞎操心了。
方多病很是质疑：(突然反应过来)等等！你说你和他认识十几年？你们彼此了解？！这么说，就我什么都不知道？！
△李莲花一愣，意外方多病是如此反应。
方多病很是不爽：不行，你们现在投奔我，我必须对我的手下都了解清楚。现在换我来问你，你，李莲花究竟籍贯何处？今年多大？家里还有什么人？平时都有些什么喜好？还有，可曾婚配？
△此时的笛飞声正坐在屋顶，从他的位置远远地向李莲花和方多病二人看去，二人声音渐弱。
李莲花：鄙人李莲花，有个兄弟叫李莲蓬，莲花山莲花镇莲花村人，曾经订过亲，但媳妇跟人跑子。这一辈子呢，没什么抱负理想，只想种种萝卜、逗逗狗，平时豆花爱吃甜的，粽子要肉的......
方多病：没一句实话。
</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△笛飞声独自坐在莲花楼屋顶上。李莲花边走边悠闲地给马喂草。方多病则走在一侧，却总不时带着怀疑地盯向楼顶的笛飞声。
方多病走到李莲花身侧：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！
李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。
方多病：认识十几年？你上次才说是一面之缘？
李莲花忙圆谎：见得不多，但知根知底。哎，这老马吃得也太多了。
方多病一把夺过李莲花手中的草料：别转移话题！——快说！
李莲花：阿飞啊，脾气不太好，他......这十年也没出过几次门，所以见识短，你不要和他计较。还有他是个武痴，武功深藏不露，你平时别惹他。
方多病：呵...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (352 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△笛飞声独自坐在莲花楼屋顶上。李莲花边走边悠闲地给马喂草。方多病则走在一侧，却总不时带着怀疑地盯向楼顶的笛飞声。
方多病走到李莲花身侧：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！
李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。
方多病：认识十几年？你上次才说是一面之缘？
李莲花忙圆谎：见得不多，但知根知底。哎，这老马吃得也太多了。
方多病一把夺过李莲花手中的草料：别转移话题！——快说！
李莲花：阿飞啊，脾气不太好，他......这十年也没出过几次门，所以见识短，你不要和他计较。还有他是个武痴，武功深藏不露，你平时别惹他。
方多病：呵，阿飞武功高？编瞎话能不能用心点？
李莲花：可都是大实话啊。反正，我和他彼此了解得很。你就别瞎操心了。
方多病很是质疑：(突然反应过来)等等！你说你和他认识十几年？你们彼此了解？！这么说，就我什么都不知道？！
△李莲花一愣，意外方多病是如此反应。
方多病很是不爽：不行，你们现在投奔我，我必须对我的手下都了解清楚。现在换我来问你，你，李莲花究竟籍贯何处？今年多大？家里还有什么人？平时都有些什么喜好？还有，可曾婚配？
△此时的笛飞声正坐在屋顶，从他的位置远远地向李莲花和方多病二人看去，二人声音渐弱。
李莲花：鄙人李莲花，有个兄弟叫李莲蓬，莲花山莲花镇莲花村人，曾经订过亲，但媳妇跟人跑子。这一辈子呢，没什么抱负理想，只想种种萝卜、逗逗狗，平时豆花爱吃甜的，粽子要肉的......
方多病：没一句实话。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>attribute_descriptions</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>李莲花的语言风格轻松幽默，善于用平易近人的语气和细节来化解紧张的气氛。他常常以调侃的方式回答问题，即使在被质疑时也能巧妙地圆谎，表现出一种机智和从容。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>李莲花是一个随和、悠闲且善于圆滑应对的人。他对朋友有着深厚的信任，同时在面对质疑时能够巧妙地转移话题或圆谎。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>方多病的语言风格直率、质疑、急躁。他说话直接，常常带着怀疑和不满，喜欢追问和质疑别人的话，显得有些急躁和不耐烦。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>方多病是一个多疑、警惕性高的人，对周围的人和事总是保持高度的怀疑态度。他同时也是一个有责任感的人，对自己身边的人非常关心，尤其是当他认为这些人可能对自己或他人构成威胁时。</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'><strong>attribute_support_texts</strong></td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。&#x27;, &#x27;李莲花：鄙人李莲花，有个兄弟叫李莲蓬，莲花山莲花镇莲花村人，曾经订过亲，但媳妇跟人跑子。这一辈子呢，没什么抱负理想，只想种种萝卜、逗逗狗，平时豆花爱吃甜的，粽子要肉的......&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。&#x27;, &#x27;李莲花：阿飞啊，脾气不太好，他......这十年也没出过几次门，所以见识短，你不要和他计较。还有他是个武痴，武功深藏不露，你平时别惹他。&#x27;, &#x27;李莲花：鄙人李莲花，有个兄弟叫李莲蓬，莲花山莲花镇莲花村人，曾经订过亲，但媳妇跟人跑子。这一辈子呢，没什么抱负理想，只想种种萝卜、逗逗狗，平时豆花爱吃甜的，粽子要肉的......&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>[&#x27;方多病：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！&#x27;, &#x27;方多病：认识十几年？你上次才说是一面之缘？&#x27;, &#x27;方多病一把夺过李莲花手中的草料：别转移话题！——快说！&#x27;, &#x27;方多病：呵，阿飞武功高？编瞎话能不能用心点？&#x27;, &#x27;方多病很是不爽：不行，你们现在投奔我，我必须对我的手下都了解清楚。现在换我来问你，你，李莲花究竟籍贯何处？今年多大？家里还有什么人？平时都有些什么喜好？还有，可曾婚配？&#x27;]</td></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:36px; border-bottom:1px solid #e3e3e3;'>[&#x27;方多病则走在一侧，却总不时带着怀疑地盯向楼顶的笛飞声。&#x27;, &#x27;方多病：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！&#x27;, &#x27;方多病很是不爽：不行，你们现在投奔我，我必须对我的手下都了解清楚。现在换我来问你，你，李莲花究竟籍贯何处？今年多大？家里还有什么人？平时都有些什么喜好？还有，可曾婚配？&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>attributes</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;语言风格&#x27;, &#x27;角色性格&#x27;, &#x27;语言风格&#x27;, &#x27;角色性格&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>main_entities</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;李莲花&#x27;, &#x27;李莲花&#x27;, &#x27;方多病&#x27;, &#x27;方多病&#x27;]</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the ExtractEntityAttributeMapper operator's functionality by extracting attributes for the entities '李莲花' and '方多病' from a given text. The operator uses an API model to identify and extract the specified attributes (such as '语言风格' and '角色性格') for these entities. The extracted information, including attribute descriptions and supporting texts, is then stored in the sample's metadata. 
这个例子展示了ExtractEntityAttributeMapper算子的功能，从给定的文本中提取实体'李莲花'和'方多病'的属性。算子使用一个API模型来识别并抽取这些实体的指定属性（如'语言风格'和'角色性格'）。提取的信息，包括属性描述和支持文本，会被存储在样本的元数据中。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_entity_attribute_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_entity_attribute_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)