# extract_event_mapper

Extracts events and relevant characters from the text.

This operator uses an API model to summarize the text into multiple events and extract the relevant characters for each event. The summary and character extraction follow a predefined format. The operator retries the API call up to a specified number of times if there is an error. The extracted events and characters are stored in the meta field of the samples. If no events are found, the original samples are returned. The operator can optionally drop the original text after processing.

从文本中提取事件和相关角色。

此算子使用 API 模型将文本总结为多个事件，并提取每个事件的相关角色。摘要和角色提取遵循预定义的格式。如果出现错误，该算子将重试 API 调用最多指定次数。提取的事件和角色存储在样本的 meta 字段中。如果没有找到事件，则返回原始样本。该算子可以选择在处理后删除原始文本。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `event_desc_key` | <class 'str'> | `'event_description'` | The key name to store the event descriptions in the meta field. It's "event_description" in default. |
| `relevant_char_key` | <class 'str'> | `'relevant_characters'` | The field name to store the relevant characters to the events in the meta field. It's "relevant_characters" in default. |
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
ExtractEventMapper(api_model='qwen2.5-72b-instruct', response_path=None, index_key='chunk_id')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (935 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (935 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>芩婆讲述了李家的历史，提到李相夷的父亲与漆木山是挚友。李家因救人事惹怒山匪，全家遭袭，只有李相夷和李相显两兄弟幸存。后来，李相显病死，李相夷被芩婆和漆木山找到并带回。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relevant_characters</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;芩婆&#x27;, &#x27;漆木山&#x27;, &#x27;李相夷&#x27;, &#x27;李相显&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>chunk_id</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (935 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李相显临终前将玉佩交给单孤刀，请求他照顾弟弟李相夷。随后，单孤刀和李相夷在破庙中遭遇一群少年乞丐的袭击，单孤刀保护李相夷，与乞丐们发生激烈冲突，最终受伤。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relevant_characters</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;李相显&#x27;, &#x27;单孤刀&#x27;, &#x27;李相夷&#x27;, &#x27;少年乞丐老大&#x27;, &#x27;乞丐甲&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>chunk_id</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (935 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>芩婆揭示了单孤刀的过去，指出他在被师父找到时手腕受伤，高烧不退，导致失忆。单孤刀对此表示怀疑，但逐渐意识到自己确实记不起之前的很多事情。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relevant_characters</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;芩婆&#x27;, &#x27;单孤刀&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>chunk_id</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (935 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>封磬透露了自己家族一直在寻找萱公主的后代，直到二十几年前找到了一个被认为是萱公主曾孙的孩子，即单孤刀。然而，现在发现李莲花才是真正的主上，封磬感到震惊和懊恼，最终跪倒在李莲花面前。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relevant_characters</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;封磬&#x27;, &#x27;李莲花&#x27;, &#x27;单孤刀&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>chunk_id</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (935 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>笛飞声解释了为什么单孤刀的血对业火独无效，而李莲花的血却能毁掉它。笛飞声对此表示冷笑。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>relevant_characters</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[&#x27;笛飞声&#x27;, &#x27;单孤刀&#x27;, &#x27;李莲花&#x27;]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>chunk_id</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the basic functionality of the ExtractEventMapper. It takes a raw text as input, and through an API model, it extracts key events and relevant characters from the text. The extracted information is then stored in the 'meta' field of the dataset. Specifically, it adds two pieces of information: 'event_description', which is a summary of the event, and 'relevant_characters', which lists the characters involved in the event. 
该示例展示了ExtractEventMapper的基本功能。它接收原始文本作为输入，通过API模型从文本中提取关键事件和相关角色。提取的信息随后被存储在数据集的'meta'字段中。具体来说，它添加了两部分信息：'event_description'，即事件的摘要，以及'relevant_characters'，列出了事件涉及的角色。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_event_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_event_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)