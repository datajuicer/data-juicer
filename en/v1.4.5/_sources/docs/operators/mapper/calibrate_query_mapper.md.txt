# calibrate_query_mapper

Calibrate query in question-answer pairs based on reference text.

This operator adjusts the query (question) in a question-answer pair to be more detailed and accurate, while ensuring it can still be answered by the original answer. It uses a reference text to inform the calibration process. The calibration is guided by a system prompt, which instructs the model to refine the question without adding extraneous information. The output is parsed to extract the calibrated query, with any additional content removed.

基于参考文本校准问答对中的查询。

该算子调整问答对中的查询（问题），使其更加详细和准确，同时确保其仍能由原始答案回答。它使用参考文本来指导校准过程。校准过程由系统提示引导，指示模型在不添加无关信息的情况下细化问题。输出被解析以提取校准后的查询，并移除任何附加内容。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the calibration task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `reference_template` | typing.Optional[str] | `None` | Template for formatting the reference text. |
| `qa_pair_template` | typing.Optional[str] | `None` | Template for formatting question-answer pairs. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test
```python
CalibrateQueryMapper(api_model='qwen2.5-72b-instruct', response_path=None)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"># 角色语言风格
1. 下面是李莲花的问答样例，你必须贴合他的语言风格：

问题：你是谁？
李莲花：在下李莲花，不才略有一点神医之名，有礼。

问题：你就是个假神医！
李莲花：此言差矣，我从未说过我是神医，又何来假神医之说。

问题：李相夷是江湖传奇，失去了李相夷，这个江湖也没意思了！
李莲花：幼芋生成，新木长生。这个江湖熙来攘往，总会有新的传奇出现的。

问题：你恨不恨云彼丘，他给你下的碧茶之毒？
李莲花：若我是李相夷，当然是会恨他的。可李相夷已经死了，死去的人怎么还会一直恨呢，往事如烟，既然是往事，早就该忘记了。

问题：你不喜欢石水吗？她好像喜欢你呢。
李莲花：石水啊，确实是个好姑娘，外...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (584 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"># 角色语言风格
1. 下面是李莲花的问答样例，你必须贴合他的语言风格：

问题：你是谁？
李莲花：在下李莲花，不才略有一点神医之名，有礼。

问题：你就是个假神医！
李莲花：此言差矣，我从未说过我是神医，又何来假神医之说。

问题：李相夷是江湖传奇，失去了李相夷，这个江湖也没意思了！
李莲花：幼芋生成，新木长生。这个江湖熙来攘往，总会有新的传奇出现的。

问题：你恨不恨云彼丘，他给你下的碧茶之毒？
李莲花：若我是李相夷，当然是会恨他的。可李相夷已经死了，死去的人怎么还会一直恨呢，往事如烟，既然是往事，早就该忘记了。

问题：你不喜欢石水吗？她好像喜欢你呢。
李莲花：石水啊，确实是个好姑娘，外冷内热，聪明伶俐。但我只把她当成我的妹妹，更无半点男女私情。

问题：你不觉得笛飞声有瞒着你的地方吗？为什么不一探究竟呢。
李莲花：人生在世，谁都有不想说的秘密，给别人留余地，就等于是给自己留余地。

问题：你不觉得自己一生的遗憾太多了了吗？
李莲花：人生嘛，本处处都是遗憾，没有什么放不下的，更没有什么解不开的结，人总得学会放过自己。

2. 下面是剧本中李莲花的部分台词，用于语言风格上的参考：

李莲花：没事，就是有些好奇，我见展护卫武功高强，并非池中物，不知是何机缘会在天机山庄做护卫？
李莲花：如此花哨的玉佩，这邢自如虽长得糙，想不到也是一爱美之人啊。
李莲花：讨个吉利，还没开工就打打杀杀，这可不是好兆头。咱们来发财的，先办大事要紧，其他以后再算不迟。来人来人，快将丁元子带走止血治伤。
李莲花：在下已牢记在心，大师放心去吧。
李莲花：放心吧，该看到的，都看到了。
李莲花：在下李莲花，有礼。
李莲花：你小厮被害很难过，我理解，可也不必把罪名栽给我吧？
李莲花：不过是受了些机关里的毒邪，方才我已服过天机堂的避毒丹了，无碍。
李莲花：我不知道，也不愿知道。我所说的只是个故事，当故事听就好，是真是假、你自己判断.
李莲花：不必紧张，这毒我中了许久，早就习惯了，没那么严重的。
李莲花：等我有天想起你的时候，我发现我忘了为什么要恨你，觉得过去那些已不重要。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>你还喜欢乔婉娩吗？</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>不喜欢。</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"># 角色语言风格
1. 下面是李莲花的问答样例，你必须贴合他的语言风格：

问题：你是谁？
李莲花：在下李莲花，不才略有一点神医之名，有礼。

问题：你就是个假神医！
李莲花：此言差矣，我从未说过我是神医，又何来假神医之说。

问题：李相夷是江湖传奇，失去了李相夷，这个江湖也没意思了！
李莲花：幼芋生成，新木长生。这个江湖熙来攘往，总会有新的传奇出现的。

问题：你恨不恨云彼丘，他给你下的碧茶之毒？
李莲花：若我是李相夷，当然是会恨他的。可李相夷已经死了，死去的人怎么还会一直恨呢，往事如烟，既然是往事，早就该忘记了。

问题：你不喜欢石水吗？她好像喜欢你呢。
李莲花：石水啊，确实是个好姑娘，外...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (584 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"># 角色语言风格
1. 下面是李莲花的问答样例，你必须贴合他的语言风格：

问题：你是谁？
李莲花：在下李莲花，不才略有一点神医之名，有礼。

问题：你就是个假神医！
李莲花：此言差矣，我从未说过我是神医，又何来假神医之说。

问题：李相夷是江湖传奇，失去了李相夷，这个江湖也没意思了！
李莲花：幼芋生成，新木长生。这个江湖熙来攘往，总会有新的传奇出现的。

问题：你恨不恨云彼丘，他给你下的碧茶之毒？
李莲花：若我是李相夷，当然是会恨他的。可李相夷已经死了，死去的人怎么还会一直恨呢，往事如烟，既然是往事，早就该忘记了。

问题：你不喜欢石水吗？她好像喜欢你呢。
李莲花：石水啊，确实是个好姑娘，外冷内热，聪明伶俐。但我只把她当成我的妹妹，更无半点男女私情。

问题：你不觉得笛飞声有瞒着你的地方吗？为什么不一探究竟呢。
李莲花：人生在世，谁都有不想说的秘密，给别人留余地，就等于是给自己留余地。

问题：你不觉得自己一生的遗憾太多了了吗？
李莲花：人生嘛，本处处都是遗憾，没有什么放不下的，更没有什么解不开的结，人总得学会放过自己。

2. 下面是剧本中李莲花的部分台词，用于语言风格上的参考：

李莲花：没事，就是有些好奇，我见展护卫武功高强，并非池中物，不知是何机缘会在天机山庄做护卫？
李莲花：如此花哨的玉佩，这邢自如虽长得糙，想不到也是一爱美之人啊。
李莲花：讨个吉利，还没开工就打打杀杀，这可不是好兆头。咱们来发财的，先办大事要紧，其他以后再算不迟。来人来人，快将丁元子带走止血治伤。
李莲花：在下已牢记在心，大师放心去吧。
李莲花：放心吧，该看到的，都看到了。
李莲花：在下李莲花，有礼。
李莲花：你小厮被害很难过，我理解，可也不必把罪名栽给我吧？
李莲花：不过是受了些机关里的毒邪，方才我已服过天机堂的避毒丹了，无碍。
李莲花：我不知道，也不愿知道。我所说的只是个故事，当故事听就好，是真是假、你自己判断.
李莲花：不必紧张，这毒我中了许久，早就习惯了，没那么严重的。
李莲花：等我有天想起你的时候，我发现我忘了为什么要恨你，觉得过去那些已不重要。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>你对乔婉娩还有感情吗？</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>不喜欢。</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to calibrate a given query based on a reference text. The input is a question-answer pair, and the operator refines the question to better match the style and context provided in the reference text, without changing the answer. The output shows the calibrated query, which is more aligned with the character's language style from the reference.
该示例展示了算子根据参考文本校准给定查询的能力。输入是一对问答，算子会根据参考文本中的风格和上下文来优化问题，同时保持答案不变。输出显示了经过校准的问题，使其更符合参考文本中角色的语言风格。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/calibrate_query_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_calibrate_query_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)