# extract_support_text_mapper

Extracts a supporting sub-text from the original text based on a given summary.

This operator uses an API model to identify and extract a segment of the original text that best matches the provided summary. It leverages a system prompt and input template to guide the extraction process. The extracted support text is stored in the specified meta field key. If the extraction fails or returns an empty string, the original summary is used as a fallback. The operator retries the extraction up to a specified number of times in case of errors.

根据给定的摘要从原始文本中提取支持性的子文本。

此算子使用 API 模型识别并提取与提供的摘要最匹配的原始文本段落。它利用系统提示和输入模板来指导提取过程。提取的支持文本存储在指定的 meta 字段键中。如果提取失败或返回空字符串，则使用原始摘要作为后备。如果出现错误，该算子将重试提取最多指定次数。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `summary_key` | <class 'str'> | `'event_description'` | The key name to store the input summary in the meta field. It's "event_description" in default. |
| `support_text_key` | <class 'str'> | `'support_text'` | The key name to store the output support text for the summary in the meta field. It's "support_text" in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test
```python
ExtractSupportTextMapper(api_model='qwen2.5-72b-instruct')
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
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李相显托付单孤刀。</td></tr></table></div></div>

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
</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>event_description</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李相显托付单孤刀。</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>support_text</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the basic functionality of the ExtractSupportTextMapper operator. Given a long text and a summary, it extracts a part of the text that best supports or explains the summary. The extracted text is then stored in the metadata under the key 'support_text'. If the extraction fails, the original summary is used as a fallback. In this test, the operator successfully finds and extracts a relevant section from the provided text based on the event description: '李相显托付单孤刀。' (Li Xiangxian entrusting Shan Gudao).
该示例展示了ExtractSupportTextMapper算子的基本功能。给定一篇长文和一个摘要，它会提取出最能支持或解释该摘要的文本部分。提取出的文本会被存储在元数据中，键名为'support_text'。如果提取失败，则使用原始摘要作为备选。在这个测试中，算子根据事件描述'李相显托付单孤刀。'（李相显将某物托付给单孤刀）成功地从提供的文本中找到了相关段落并进行了提取。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_support_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_support_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)