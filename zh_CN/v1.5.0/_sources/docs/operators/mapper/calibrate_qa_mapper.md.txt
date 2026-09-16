# calibrate_qa_mapper

Calibrates question-answer pairs based on reference text using an API model.

This operator uses a specified API model to calibrate question-answer pairs, making them more detailed and accurate. It constructs the input prompt by combining the reference text and the question-answer pair, then sends it to the API for calibration. The output is parsed to extract the calibrated question and answer. The operator retries the API call and parsing up to a specified number of times in case of errors. The default system prompt, input templates, and output pattern can be customized. The operator supports additional parameters for model initialization and sampling.

根据参考文本使用API模型校准问答对。

该算子使用指定的API模型来校准问答对，使其更加详细和准确。它通过将参考文本和问答对结合起来构建输入提示，然后将其发送到API进行校准。输出被解析以提取校准后的问题和答案。如果出现错误，算子会重试API调用和解析，最多重试指定次数。默认的系统提示、输入模板和输出模式可以自定义。该算子支持用于模型初始化和采样的附加参数。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

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
CalibrateQAMapper(api_model='qwen2.5-72b-instruct')
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
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>你还喜欢乔婉娩吗？</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>乔婉娩啊，曾经确实有过一段情愫，但如今早已风轻云淡。往事如烟，既然是往事，早就该忘记了。</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the basic usage of the CalibrateQAMapper. It takes a reference text and a question-answer pair, then uses an API model (in this case, 'qwen2.5-72b-instruct') to generate a more detailed and accurate answer. 
此示例展示了CalibrateQAMapper的基本用法。它接收一段参考文本和一个问答对，然后使用API模型（此处为'qwen2.5-72b-instruct'）生成更加详细和准确的答案。

### test_args
```python
CalibrateQAMapper(api_model='qwen2.5-72b-instruct', api_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', response_path='choices.0.message.content')
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
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>query</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>你还喜欢乔婉娩吗？</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>response</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>乔婉娩啊，她曾是我生命中的一段过往，但如今早已随风而散。人总要学会放下，过去的感情就像流水，一去不复返。如今的我，只愿在这江湖上自在行走，不再为情所困。</td></tr></table></div></div>

#### ✨ explanation 解释
This example illustrates the use of additional arguments with the CalibrateQAMapper. It specifies the API endpoint and the path to the response content in the output, demonstrating the flexibility of configuring the operator.
此示例展示了如何在CalibrateQAMapper中使用额外参数。它指定了API端点以及输出中响应内容的路径，展示了配置算子的灵活性。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/calibrate_qa_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_calibrate_qa_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)