# document_simhash_deduplicator

Deduplicates samples at the document level using SimHash.

This operator computes SimHash values for each sample and removes duplicates based on a specified Hamming distance threshold. It supports different tokenization methods: 'space', 'punctuation', and 'character'. The SimHash is computed over shingles of a given window size, and the deduplication process clusters similar documents and retains only one from each cluster. The default mode converts text to lowercase and can ignore specific patterns. The key metric, Hamming distance, is used to determine similarity between SimHash values. Important notes:
- The `ignore_pattern` parameter can be used to exclude certain substrings during SimHash computation.
- For punctuation-based tokenization, the `ignore_pattern` should not include punctuations to avoid conflicts.
- The `hamming_distance` must be less than the number of blocks (`num_blocks`).
- Only the first sample in each cluster is retained by default.

使用SimHash在文档级别去重样本。

该算子为每个样本计算SimHash值，并根据指定的汉明距离阈值移除重复项。它支持不同的分词方法：'space'、'punctuation'和'character'。SimHash是在给定窗口大小的片段上计算的，去重过程将相似文档聚类并从每个聚类中仅保留一个。默认模式将文本转换为小写，并可以忽略特定模式。关键指标汉明距离用于确定SimHash值之间的相似性。重要提示：
- 可以使用`ignore_pattern`参数在SimHash计算过程中排除某些子字符串。
- 对于基于标点符号的分词，`ignore_pattern`不应包含标点符号以避免冲突。
- `hamming_distance`必须小于区块数（`num_blocks`）。
- 默认情况下，仅保留每个聚类中的第一个样本。

Type 算子类型: **deduplicator**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tokenization` | <class 'str'> | `'space'` | tokenization method for sample texts |
| `window_size` | typing.Annotated[int, Gt(gt=0)] | `6` | window size of shingling |
| `lowercase` | <class 'bool'> | `True` | whether to convert text to lower case first |
| `ignore_pattern` | typing.Optional[str] | `None` | whether to ignore sub-strings with specific pattern when computing simhash |
| `num_blocks` | typing.Annotated[int, Gt(gt=0)] | `6` | number of blocks in simhash computing |
| `hamming_distance` | typing.Annotated[int, Gt(gt=0)] | `4` | the max hamming distance threshold in near-duplicate detection. When the hamming distance of two sample texts is <= this threshold, they are regarded as similar samples and this op will only keep one of them after deduplication. This threshold should be always less than num_blocks |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_0
```python
DocumentSimhashDeduplicator(ignore_pattern='\\p{P}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's behavior when there is only one unique document in the input. The operator computes a SimHash value for the single document and, since there are no duplicates, it retains the document as is. 
这个例子展示了当输入中只有一个唯一的文档时，算子的行为。算子为这个单个文档计算一个SimHash值，因为没有重复项，所以保留了该文档。

### test_english_deduplication
```python
DocumentSimhashDeduplicator(ignore_pattern='\\p{P}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sunday and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Do you need a cup of coffee?</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is sunday and it&#x27;s really a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel method on LLM pretraining.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (8927 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must pass within one foot of hundreds of colleagues in the hallways, locker rooms, cafeterias, and cutting lines. The same conditions have spurred Covid-19 outbreaks at meat plants from Minnesota and Wisconsin to Colorado, Nebraska, Missouri, Iowa, Pennsylvania, North Carolina, and Georgia.

801 workers at the Sioux Falls plant have tested positive, together with 206 people close to them. The outbreak has killed Agustín Rodríguez Martínez, aged 64, an employee with two decades of experience originally from El Salvador, and Craig Allen Franken, 61, who worked for Smithfield his entire adult life.

The company knew of its first infection on March 24 or earlier. The virus spread exponentially for several weeks. Ahead of Easter Sunday and Monday (April 12-13), Smithfield promised to “completely shutter” to sanitize and put up cardboard and plastic sheet dividers. This would not end transmission, as potentially hundreds of staff were already carrying the virus. But even during this “shutdown,” many cars were seen in the parking lot. The mayor admits that the company lied, and the local AFL-CIO alleges the plant ran 60 percent production. On Easter, with 238 known infections, Smithfield finally agreed to shut down indefinitely after a request from the mayor and the governor. Yet the company insisted on waiting three more days to actually halt production.

Smithfield denied contributing to the outbreak, saying it took a “very proactive approach.” Relying on racism, the company blamed workers for getting themselves sick. A spokesperson said the outbreak was so severe because of the plant’s “large immigrant population,” claming “Living circumstances in certain cultures are different than they are with your traditional American family.” They slandered the workers as dirty, ignorant, and untrustworthy with help from governor Kristi Noem, who claimed, “99 percent of what’s going on today wasn’t happening inside the facility. It was more at home, where these employees were going home and spreading some of the virus” by living too close together.

One sick worker, Michael Bul Gayo Gatluak, 22 and originally from South Sudan, says, “With how we work on the line, I would say I got sick because of them not taking safety measures.” His job is “really, really close” to other workers chopping fresh-killed pigs. “The job is so heavy. You have to breathe so hard.”

In early March, union officials requested masks, overcoats, entrance checking for fevers, and less crowding in 500-capacity cafeterias. But Smithfield waited on most safety measures until early April. Only April 6 did they start checking for fevers. Instead of protective masks, they gave out beard nets.

Smithfield concealed infections with a policy of informing only employees whose work stations were in the same area as a person who tested positive. The fact that workers are required to move around was willfully ignored. One worker who tested positive said, “I clearly would have gotten it at the factory. This week I have worked on three different floors. I’ve eaten in two different cafeterias … I’ve been walking through the whole place.” Employees from the eighth floor of the plant were quarantined, but everyone else was told to keep working.

What Is Really Going On?

Average plant wages are around $16 an hour. Smithfield never raised them. Instead, they offered $500 to employees who could go all of April without an unapproved day off. The company says their “Responsibility Bonuses” show their “immense gratefulness” to employees “for their selfless sacrifices.”

Meanwhile, the local Argus Leader wrote union members wanted essential-worker hazard pay, which “would be considered hourly compensation about 1.5 or two times their normal pay.” One worker said, “I feel like they’re bribing us with [the bonus] to come to work sick. That’s how you know they don’t care.”

Both Sioux Falls workers killed by Covid-19 were in their sixties. It is unconscionable that they were still working. All meatpackers over 50 should be on paid leave. Agustín Rodríguez, 64, had a rough job sawing the legs off dead pigs. He mopped floors with a fever shortly before he was hospitalized.

When CEO Kenneth Sullivan closed the plant, he claimed, “We have continued to run our facilities for one reason: to sustain our nation’s food supply.” This is an effort to sweep Smithfield’s abuses under the rug, as if the company were operating for public benefit. This patriotic propaganda that all Americans are in it together is like a drug to keep workers from getting organized.

The major union in the industry, including at Smithfield, is the United Food and Commercial Workers union (UFCW). What union leaders have done is ultimately troubling.

Can Workers Fight?

Local AFL-CIO president Kooper Caraway has publicly said management delayed safety action as long as possible for profit. But while some workers were demanding a two-week shutdown, Caraway told the Argus Leader that was unrealistic because the government considers the plant essential. He suggested the union would be happy with minimal safety measures: “Even if 10 people get exposed in a day rather than 11. If you can implement a program where even one or two less people get exposed during a shift, that’s one or two less people.” Of course reducing infections is good, but suggesting workers would be satisfied if the company allowed 90% of the contagion to continue is horrifying.

The response of UFCW leadership was worse. As the disease was exploding, they told the Argus Leader, “We applaud [Smithfield’s] decision to temporarily close the plant [over Easter weekend] to push for an even safer work environment.” What does “even safer” mean in this context?

The union bureaucracy has taken weak action elsewhere. In Pennsylvania, the UFCW negotiated $2 hazard pay for two months with Cargill Meat — the same pandemic premium Amazon gave workers without a union. In Nebraska, the UFCW negotiated $4 hazard pay for one month with meat giant JBS.

The union has said nothing about forcing companies to send older workers home with pay, even though a 70-year-old shop steward and a 78-year-old grandfather working at JBS plants were killed by Covid-19. Smithfield workers were promised only two weeks of shutdown pay. For many, this compensation is half their normal paycheck because they routinely put in 66 hour weeks — overtime that costs exhaustion and chronic pain.

Union officials endeavor to cooperate with the meat companies. An Iowa UFCW president actually suggested it might be impossible for plants to move workers a full six feet apart and told the Des Moines Register, “We can’t stop the plants. If we stop the plants from running, we stop feeding the country. We want to do everything we can to make sure the employees are safe to keep the plant running.”

Every part of this explanation directly overlaps with what the Smithfield CEO said. Unfortunately, it amounts to accepting the company’s excuses.

They claim that workers who do hard physical labor, waking up at 4 a.m. and often working six days a week for years, would be guilty of taking food away from the people and hurting America if they dared to fight for their human needs. But nothing is said about the company raking in profits and even murdering workers to increase them.

Smithfield’s parent company W.H. Group, which slaughters around 30 million pigs per year in plants in both the United States and China, saw its profits skyrocket by about one third in 2019 to $1.38 billion. It is disturbing that UFCW officials do not bring up these soaring profits in their response to the outbreaks. Reuters published a report on the corporation’s financial success in late March. The head of W.H. Group had touted to the media that it got through the pandemic in China with very limited impact on production.

It is true that many Smithfield workers are reasonably afraid for their jobs and want to keep working. A 25-year-old employee explained, “I have a lot of bills. My baby’s coming soon — I have to work.” At the same time, he was afraid of infecting his pregnant wife. His spouse, a former employee, said bitterly, “Smithfield— they don’t care about employees. They only care about their money.”

Workers are pressured in these two painful directions. Nonetheless, work can mean solidarity. Before Smithfield even checked temperatures, there was a “sick-out” strike without union support by 800 to 1,000 workers at a JBS meat factory in Colorado. Hundreds of workers also called in sick days at a Nebraska JBS plant.

Trade union leaders won’t even whisper the word “strike” when thousands of workers are thinking about it. They are limiting themselves to polite requests. We need a workers’ movement that asks who controls the factory, that threatens to disrupt the bosses’ profits, and that allows workers to use their immense power — this could change the meat industry and the world. </pre></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 6:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plants in Sioux Falls, South Dakota. The plant slaughters 19,500 pig a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (8927 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plants in Sioux Falls, South Dakota. The plant slaughters 19,500 pig a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must pass within one foot of hundreds of colleagues in the hallways, locker rooms, cafeterias, and cutting lines. The same conditions have spurred Covid-19 outbreaks at meat plants from Minnesota and Wisconsin to Colorado, Nebraska, Missouri, Iowa, Pennsylvania, North Carolina, and Georgia.

801 workers at the Sioux Falls plant have tested positive, together with 206 people close to them. The outbreak has killed Agustín Rodríguez Martínez, aged 64, an employee with two decades of experience originally from El Salvador, and Craig Allen Franken, 61, who worked for Smithfield his entire adult life.

The company knew of its first infection on March 24 or earlier. The virus spread exponentially for several weeks. Ahead of Easter Sunday and Monday (April 12-13), Smithfield promised to “completely shutter” to sanitize and put up cardboard and plastic sheet dividers. This would not end transmission, as potentially hundreds of staff were already carrying the virus. But even during this “shutdown,” many cars were seen in the parking lot. The mayor admits that the company lied, and the local AFL-CIO alleges the plant ran 60 percent production. On Easter, with 238 known infections, Smithfield finally agreed to shut down indefinitely after a request from the mayor and the governor. Yet the company insisted on waiting three more days to actually halt production.

Smithfield denied contributing to the outbreak, saying it took a “very proactive approach.” Relying on racism, the company blamed workers for getting themselves sick. A spokesperson said the outbreak was so severe because of the plant’s “large immigrant population,” claming “Living circumstances in certain cultures are different than they are with your traditional American family.” They slandered the workers as dirty, ignorant, and untrustworthy with help from governor Kristi Noem, who claimed, “99 percent of what’s going on today wasn’t happening inside the facility. It was more at home, where these employees were going home and spreading some of the virus” by living too close together.

One sick worker, Michael Bul Gayo Gatluak, 22 and originally from South Sudan, says, “With how we work on the line, I would say I got sick because of them not taking safety measures.” His job is “really, really close” to other workers chopping fresh-killed pigs. “The job is so heavy. You have to breathe so hard.”

In early March, union officials requested masks, overcoats, entrance checking for fevers, and less crowding in 500-capacity cafeterias. But Smithfield waited on most safety measures until early April. Only April 6 did they start checking for fevers. Instead of protective masks, they gave out beard nets.

Smithfield concealed infections with a policy of informing only employees whose work stations were in the same area as a person who tested positive. The fact that workers are required to move around was willfully ignored. One worker who tested positive said, “I clearly would have gotten it at the factory. This week I have worked on three different floors. I’ve eaten in two different cafeterias … I’ve been walking through the whole place.” Employees from the eighth floor of the plant were quarantined, but everyone else was told to keep working.

What Is Really Going On?

Average plant wages are around $16 an hour. Smithfield never raised them. Instead, they offered $500 to employees who could go all of April without an unapproved day off. The company says their “Responsibility Bonuses” show their “immense gratefulness” to employees “for their selfless sacrifices.”

Meanwhile, the local Argus Leader wrote union members wanted essential-worker hazard pay, which “would be considered hourly compensation about 1.5 or two times their normal pay.” One worker said, “I feel like they’re bribing us with [the bonus] to come to work sick. That’s how you know they don’t care.”

Both Sioux Falls workers killed by Covid-19 were in their sixties. It is unconscionable that they were still working. All meatpackers over 50 should be on paid leave. Agustín Rodríguez, 64, had a rough job sawing the legs off dead pigs. He mopped floors with a fever shortly before he was hospitalized.

When CEO Kenneth Sullivan closed the plant, he claimed, “We have continued to run our facilities for one reason: to sustain our nation’s food supply.” This is an effort to sweep Smithfield’s abuses under the rug, as if the company were operating for public benefit. This patriotic propaganda that all Americans are in it together is like a drug to keep workers from getting organized.

The major union in the industry, including at Smithfield, is the United Food and Commercial Workers union (UFCW). What union leaders have done is ultimately troubling.

Can Workers Fight?

Local AFL-CIO president Kooper Caraway has publicly said management delayed safety action as long as possible for profit. But while some workers were demanding a two-week shutdown, Caraway told the Argus Leader that was unrealistic because the government considers the plant essential. He suggested the union would be happy with minimal safety measures: “Even if 10 people get exposed in a day rather than 11. If you can implement a program where even one or two less people get exposed during a shift, that’s one or two less people.” Of course reducing infections is good, but suggesting workers would be satisfied if the company allowed 90% of the contagion to continue is horrifying.

The response of UFCW leadership was worse. As the disease was exploding, they told the Argus Leader, “We applaud [Smithfield’s] decision to temporarily close the plant [over Easter weekend] to push for an even safer work environment.” What does “even safer” mean in this context?

The union bureaucracy has taken weak action elsewhere. In Pennsylvania, the UFCW negotiated $2 hazard pay for two months with Cargill Meat — the same pandemic premium Amazon gave workers without a union. In Nebraska, the UFCW negotiated $4 hazard pay for one month with meat giant JBS.

The union has said nothing about forcing companies to send older workers home with pay, even though a 70-year-old shop steward and a 78-year-old grandfather working at JBS plants were killed by Covid-19. Smithfield workers were promised only two weeks of shutdown pay. For many, this compensation is half their normal paycheck because they routinely put in 66 hour weeks — overtime that costs exhaustion and chronic pain.

Union officials endeavor to cooperate with the meat companies. An Iowa UFCW president actually suggested it might be impossible for plants to move workers a full six feet apart and told the Des Moines Register, “We can’t stop the plants. If we stop the plants from running, we stop feeding the country. We want to do everything we can to make sure the employees are safe to keep the plant running.”

Every part of this explanation directly overlaps with what the Smithfield CEO said. Unfortunately, it amounts to accepting the company’s excuses.

They claim that workers who do hard physical labor, waking up at 4 a.m. and often working six days a week for years, would be guilty of taking food away from the people and hurting America if they dared to fight for their human needs. But nothing is said about the company raking in profits and even murdering workers to increase them.

Smithfield’s parent company W.H. Group, which slaughters around 30 million pigs per year in plants in both the United States and China, saw its profits skyrocket by about one third in 2019 to $1.38 billion. It is disturbing that UFCW officials do not bring up these soaring profits in their response to the outbreaks. Reuters published a report on the corporation’s financial success in late March. The head of W.H. Group had touted to the media that it got through the pandemic in China with very limited impact on production.

It is true that many Smithfield workers are reasonably afraid for their jobs and want to keep working. A 25-year-old employee explained, “I have a lot of bills. My baby’s coming soon — I have to work.” At the same time, he was afraid of infecting his pregnant wife. His spouse, a former employee, said bitterly, “Smithfield— they don’t care about employees. They only care about their money.”

Workers are pressured in these two painful directions. Nonetheless, work can mean solidarity. Before Smithfield even checked temperatures, there was a “sick-out” strike without union support by 800 to 1,000 workers at a JBS meat factory in Colorado. Hundreds of workers also called in sick days at a Nebraska JBS plant.

Trade union leaders won’t even whisper the word “strike” when thousands of workers are thinking about it. They are limiting themselves to polite requests. We need a workers’ movement that asks who controls the factory, that threatens to disrupt the bosses’ profits, and that allows workers to use their immense power — this could change the meat industry and the world. </pre></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 7:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (4560 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must pass within one foot of hundreds of colleagues in the hallways, locker rooms, cafeterias, and cutting lines. The same conditions have spurred Covid-19 outbreaks at meat plants from Minnesota and Wisconsin to Colorado, Nebraska, Missouri, Iowa, Pennsylvania, North Carolina, and Georgia.

801 workers at the Sioux Falls plant have tested positive, together with 206 people close to them. The outbreak has killed Agustín Rodríguez Martínez, aged 64, an employee with two decades of experience originally from El Salvador, and Craig Allen Franken, 61, who worked for Smithfield his entire adult life.

The company knew of its first infection on March 24 or earlier. The virus spread exponentially for several weeks. Ahead of Easter Sunday and Monday (April 12-13), Smithfield promised to “completely shutter” to sanitize and put up cardboard and plastic sheet dividers. This would not end transmission, as potentially hundreds of staff were already carrying the virus. But even during this “shutdown,” many cars were seen in the parking lot. The mayor admits that the company lied, and the local AFL-CIO alleges the plant ran 60 percent production. On Easter, with 238 known infections, Smithfield finally agreed to shut down indefinitely after a request from the mayor and the governor. Yet the company insisted on waiting three more days to actually halt production.

Smithfield denied contributing to the outbreak, saying it took a “very proactive approach.” Relying on racism, the company blamed workers for getting themselves sick. A spokesperson said the outbreak was so severe because of the plant’s “large immigrant population,” claming “Living circumstances in certain cultures are different than they are with your traditional American family.” They slandered the workers as dirty, ignorant, and untrustworthy with help from governor Kristi Noem, who claimed, “99 percent of what’s going on today wasn’t happening inside the facility. It was more at home, where these employees were going home and spreading some of the virus” by living too close together.

One sick worker, Michael Bul Gayo Gatluak, 22 and originally from South Sudan, says, “With how we work on the line, I would say I got sick because of them not taking safety measures.” His job is “really, really close” to other workers chopping fresh-killed pigs. “The job is so heavy. You have to breathe so hard.”

In early March, union officials requested masks, overcoats, entrance checking for fevers, and less crowding in 500-capacity cafeterias. But Smithfield waited on most safety measures until early April. Only April 6 did they start checking for fevers. Instead of protective masks, they gave out beard nets.

Smithfield concealed infections with a policy of informing only employees whose work stations were in the same area as a person who tested positive. The fact that workers are required to move around was willfully ignored. One worker who tested positive said, “I clearly would have gotten it at the factory. This week I have worked on three different floors. I’ve eaten in two different cafeterias … I’ve been walking through the whole place.” Employees from the eighth floor of the plant were quarantined, but everyone else was told to keep working.

What Is Really Going On?

Average plant wages are around $16 an hour. Smithfield never raised them. Instead, they offered $500 to employees who could go all of April without an unapproved day off. The company says their “Responsibility Bonuses” show their “immense gratefulness” to employees “for their selfless sacrifices.”

Meanwhile, the local Argus Leader wrote union members wanted essential-worker hazard pay, which “would be considered hourly compensation about 1.5 or two times their normal pay.” One worker said, “I feel like they’re bribing us with [the bonus] to come to work sick. That’s how you know they don’t care.”

Both Sioux Falls workers killed by Covid-19 were in their sixties. It is unconscionable that they were still working. All meatpackers over 50 should be on paid leave. Agustín Rodríguez, 64, had a rough job sawing the legs off dead pigs. He mopped floors with a fever shortly before he was hospitalized.

When CEO Kenneth Sullivan closed the plant, he claimed, “We have continued to run our facilities for one reason: to sustain our nation’s food supply.” This is an effort to sweep Smithfield’s abuses under the rug, as if the company were operating for public benefit. This patriotic propaganda that all Americans are in it together is like a drug to keep workers from getting organized. </pre></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 8:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plants in Sioux Falls, South Dakota. The plant slaughters 19,500 pig a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (4560 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plants in Sioux Falls, South Dakota. The plant slaughters 19,500 pig a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must pass within one foot of hundreds of colleagues in the hallways, locker rooms, cafeterias, and cutting lines. The same conditions have spurred Covid-19 outbreaks at meat plants from Minnesota and Wisconsin to Colorado, Nebraska, Missouri, Iowa, Pennsylvania, North Carolina, and Georgia.

801 workers at the Sioux Falls plant have tested positive, together with 206 people close to them. The outbreak has killed Agustín Rodríguez Martínez, aged 64, an employee with two decades of experience originally from El Salvador, and Craig Allen Franken, 61, who worked for Smithfield his entire adult life.

The company knew of its first infection on March 24 or earlier. The virus spread exponentially for several weeks. Ahead of Easter Sunday and Monday (April 12-13), Smithfield promised to “completely shutter” to sanitize and put up cardboard and plastic sheet dividers. This would not end transmission, as potentially hundreds of staff were already carrying the virus. But even during this “shutdown,” many cars were seen in the parking lot. The mayor admits that the company lied, and the local AFL-CIO alleges the plant ran 60 percent production. On Easter, with 238 known infections, Smithfield finally agreed to shut down indefinitely after a request from the mayor and the governor. Yet the company insisted on waiting three more days to actually halt production.

Smithfield denied contributing to the outbreak, saying it took a “very proactive approach.” Relying on racism, the company blamed workers for getting themselves sick. A spokesperson said the outbreak was so severe because of the plant’s “large immigrant population,” claming “Living circumstances in certain cultures are different than they are with your traditional American family.” They slandered the workers as dirty, ignorant, and untrustworthy with help from governor Kristi Noem, who claimed, “99 percent of what’s going on today wasn’t happening inside the facility. It was more at home, where these employees were going home and spreading some of the virus” by living too close together.

One sick worker, Michael Bul Gayo Gatluak, 22 and originally from South Sudan, says, “With how we work on the line, I would say I got sick because of them not taking safety measures.” His job is “really, really close” to other workers chopping fresh-killed pigs. “The job is so heavy. You have to breathe so hard.”

In early March, union officials requested masks, overcoats, entrance checking for fevers, and less crowding in 500-capacity cafeterias. But Smithfield waited on most safety measures until early April. Only April 6 did they start checking for fevers. Instead of protective masks, they gave out beard nets.

Smithfield concealed infections with a policy of informing only employees whose work stations were in the same area as a person who tested positive. The fact that workers are required to move around was willfully ignored. One worker who tested positive said, “I clearly would have gotten it at the factory. This week I have worked on three different floors. I’ve eaten in two different cafeterias … I’ve been walking through the whole place.” Employees from the eighth floor of the plant were quarantined, but everyone else was told to keep working.

What Is Really Going On?

Average plant wages are around $16 an hour. Smithfield never raised them. Instead, they offered $500 to employees who could go all of April without an unapproved day off. The company says their “Responsibility Bonuses” show their “immense gratefulness” to employees “for their selfless sacrifices.”

Meanwhile, the local Argus Leader wrote union members wanted essential-worker hazard pay, which “would be considered hourly compensation about 1.5 or two times their normal pay.” One worker said, “I feel like they’re bribing us with [the bonus] to come to work sick. That’s how you know they don’t care.”

Both Sioux Falls workers killed by Covid-19 were in their sixties. It is unconscionable that they were still working. All meatpackers over 50 should be on paid leave. Agustín Rodríguez, 64, had a rough job sawing the legs off dead pigs. He mopped floors with a fever shortly before he was hospitalized.

When CEO Kenneth Sullivan closed the plant, he claimed, “We have continued to run our facilities for one reason: to sustain our nation’s food supply.” This is an effort to sweep Smithfield’s abuses under the rug, as if the company were operating for public benefit. This patriotic propaganda that all Americans are in it together is like a drug to keep workers from getting organized. </pre></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 9:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel method on LLM pretraining.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sunday and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Do you need a cup of coffee?</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is sunday and it&#x27;s really a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel method on LLM pretraining.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (8927 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must pass within one foot of hundreds of colleagues in the hallways, locker rooms, cafeterias, and cutting lines. The same conditions have spurred Covid-19 outbreaks at meat plants from Minnesota and Wisconsin to Colorado, Nebraska, Missouri, Iowa, Pennsylvania, North Carolina, and Georgia.

801 workers at the Sioux Falls plant have tested positive, together with 206 people close to them. The outbreak has killed Agustín Rodríguez Martínez, aged 64, an employee with two decades of experience originally from El Salvador, and Craig Allen Franken, 61, who worked for Smithfield his entire adult life.

The company knew of its first infection on March 24 or earlier. The virus spread exponentially for several weeks. Ahead of Easter Sunday and Monday (April 12-13), Smithfield promised to “completely shutter” to sanitize and put up cardboard and plastic sheet dividers. This would not end transmission, as potentially hundreds of staff were already carrying the virus. But even during this “shutdown,” many cars were seen in the parking lot. The mayor admits that the company lied, and the local AFL-CIO alleges the plant ran 60 percent production. On Easter, with 238 known infections, Smithfield finally agreed to shut down indefinitely after a request from the mayor and the governor. Yet the company insisted on waiting three more days to actually halt production.

Smithfield denied contributing to the outbreak, saying it took a “very proactive approach.” Relying on racism, the company blamed workers for getting themselves sick. A spokesperson said the outbreak was so severe because of the plant’s “large immigrant population,” claming “Living circumstances in certain cultures are different than they are with your traditional American family.” They slandered the workers as dirty, ignorant, and untrustworthy with help from governor Kristi Noem, who claimed, “99 percent of what’s going on today wasn’t happening inside the facility. It was more at home, where these employees were going home and spreading some of the virus” by living too close together.

One sick worker, Michael Bul Gayo Gatluak, 22 and originally from South Sudan, says, “With how we work on the line, I would say I got sick because of them not taking safety measures.” His job is “really, really close” to other workers chopping fresh-killed pigs. “The job is so heavy. You have to breathe so hard.”

In early March, union officials requested masks, overcoats, entrance checking for fevers, and less crowding in 500-capacity cafeterias. But Smithfield waited on most safety measures until early April. Only April 6 did they start checking for fevers. Instead of protective masks, they gave out beard nets.

Smithfield concealed infections with a policy of informing only employees whose work stations were in the same area as a person who tested positive. The fact that workers are required to move around was willfully ignored. One worker who tested positive said, “I clearly would have gotten it at the factory. This week I have worked on three different floors. I’ve eaten in two different cafeterias … I’ve been walking through the whole place.” Employees from the eighth floor of the plant were quarantined, but everyone else was told to keep working.

What Is Really Going On?

Average plant wages are around $16 an hour. Smithfield never raised them. Instead, they offered $500 to employees who could go all of April without an unapproved day off. The company says their “Responsibility Bonuses” show their “immense gratefulness” to employees “for their selfless sacrifices.”

Meanwhile, the local Argus Leader wrote union members wanted essential-worker hazard pay, which “would be considered hourly compensation about 1.5 or two times their normal pay.” One worker said, “I feel like they’re bribing us with [the bonus] to come to work sick. That’s how you know they don’t care.”

Both Sioux Falls workers killed by Covid-19 were in their sixties. It is unconscionable that they were still working. All meatpackers over 50 should be on paid leave. Agustín Rodríguez, 64, had a rough job sawing the legs off dead pigs. He mopped floors with a fever shortly before he was hospitalized.

When CEO Kenneth Sullivan closed the plant, he claimed, “We have continued to run our facilities for one reason: to sustain our nation’s food supply.” This is an effort to sweep Smithfield’s abuses under the rug, as if the company were operating for public benefit. This patriotic propaganda that all Americans are in it together is like a drug to keep workers from getting organized.

The major union in the industry, including at Smithfield, is the United Food and Commercial Workers union (UFCW). What union leaders have done is ultimately troubling.

Can Workers Fight?

Local AFL-CIO president Kooper Caraway has publicly said management delayed safety action as long as possible for profit. But while some workers were demanding a two-week shutdown, Caraway told the Argus Leader that was unrealistic because the government considers the plant essential. He suggested the union would be happy with minimal safety measures: “Even if 10 people get exposed in a day rather than 11. If you can implement a program where even one or two less people get exposed during a shift, that’s one or two less people.” Of course reducing infections is good, but suggesting workers would be satisfied if the company allowed 90% of the contagion to continue is horrifying.

The response of UFCW leadership was worse. As the disease was exploding, they told the Argus Leader, “We applaud [Smithfield’s] decision to temporarily close the plant [over Easter weekend] to push for an even safer work environment.” What does “even safer” mean in this context?

The union bureaucracy has taken weak action elsewhere. In Pennsylvania, the UFCW negotiated $2 hazard pay for two months with Cargill Meat — the same pandemic premium Amazon gave workers without a union. In Nebraska, the UFCW negotiated $4 hazard pay for one month with meat giant JBS.

The union has said nothing about forcing companies to send older workers home with pay, even though a 70-year-old shop steward and a 78-year-old grandfather working at JBS plants were killed by Covid-19. Smithfield workers were promised only two weeks of shutdown pay. For many, this compensation is half their normal paycheck because they routinely put in 66 hour weeks — overtime that costs exhaustion and chronic pain.

Union officials endeavor to cooperate with the meat companies. An Iowa UFCW president actually suggested it might be impossible for plants to move workers a full six feet apart and told the Des Moines Register, “We can’t stop the plants. If we stop the plants from running, we stop feeding the country. We want to do everything we can to make sure the employees are safe to keep the plant running.”

Every part of this explanation directly overlaps with what the Smithfield CEO said. Unfortunately, it amounts to accepting the company’s excuses.

They claim that workers who do hard physical labor, waking up at 4 a.m. and often working six days a week for years, would be guilty of taking food away from the people and hurting America if they dared to fight for their human needs. But nothing is said about the company raking in profits and even murdering workers to increase them.

Smithfield’s parent company W.H. Group, which slaughters around 30 million pigs per year in plants in both the United States and China, saw its profits skyrocket by about one third in 2019 to $1.38 billion. It is disturbing that UFCW officials do not bring up these soaring profits in their response to the outbreaks. Reuters published a report on the corporation’s financial success in late March. The head of W.H. Group had touted to the media that it got through the pandemic in China with very limited impact on production.

It is true that many Smithfield workers are reasonably afraid for their jobs and want to keep working. A 25-year-old employee explained, “I have a lot of bills. My baby’s coming soon — I have to work.” At the same time, he was afraid of infecting his pregnant wife. His spouse, a former employee, said bitterly, “Smithfield— they don’t care about employees. They only care about their money.”

Workers are pressured in these two painful directions. Nonetheless, work can mean solidarity. Before Smithfield even checked temperatures, there was a “sick-out” strike without union support by 800 to 1,000 workers at a JBS meat factory in Colorado. Hundreds of workers also called in sick days at a Nebraska JBS plant.

Trade union leaders won’t even whisper the word “strike” when thousands of workers are thinking about it. They are limiting themselves to polite requests. We need a workers’ movement that asks who controls the factory, that threatens to disrupt the bosses’ profits, and that allows workers to use their immense power — this could change the meat industry and the world. </pre></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 6:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (4560 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork. Most of the workers are immigrants from Ethiopia, Mexico, South Sudan, Honduras, Myanmar, Somalia, Guatemala, and other poor countries.

Inevitably workers must pass within one foot of hundreds of colleagues in the hallways, locker rooms, cafeterias, and cutting lines. The same conditions have spurred Covid-19 outbreaks at meat plants from Minnesota and Wisconsin to Colorado, Nebraska, Missouri, Iowa, Pennsylvania, North Carolina, and Georgia.

801 workers at the Sioux Falls plant have tested positive, together with 206 people close to them. The outbreak has killed Agustín Rodríguez Martínez, aged 64, an employee with two decades of experience originally from El Salvador, and Craig Allen Franken, 61, who worked for Smithfield his entire adult life.

The company knew of its first infection on March 24 or earlier. The virus spread exponentially for several weeks. Ahead of Easter Sunday and Monday (April 12-13), Smithfield promised to “completely shutter” to sanitize and put up cardboard and plastic sheet dividers. This would not end transmission, as potentially hundreds of staff were already carrying the virus. But even during this “shutdown,” many cars were seen in the parking lot. The mayor admits that the company lied, and the local AFL-CIO alleges the plant ran 60 percent production. On Easter, with 238 known infections, Smithfield finally agreed to shut down indefinitely after a request from the mayor and the governor. Yet the company insisted on waiting three more days to actually halt production.

Smithfield denied contributing to the outbreak, saying it took a “very proactive approach.” Relying on racism, the company blamed workers for getting themselves sick. A spokesperson said the outbreak was so severe because of the plant’s “large immigrant population,” claming “Living circumstances in certain cultures are different than they are with your traditional American family.” They slandered the workers as dirty, ignorant, and untrustworthy with help from governor Kristi Noem, who claimed, “99 percent of what’s going on today wasn’t happening inside the facility. It was more at home, where these employees were going home and spreading some of the virus” by living too close together.

One sick worker, Michael Bul Gayo Gatluak, 22 and originally from South Sudan, says, “With how we work on the line, I would say I got sick because of them not taking safety measures.” His job is “really, really close” to other workers chopping fresh-killed pigs. “The job is so heavy. You have to breathe so hard.”

In early March, union officials requested masks, overcoats, entrance checking for fevers, and less crowding in 500-capacity cafeterias. But Smithfield waited on most safety measures until early April. Only April 6 did they start checking for fevers. Instead of protective masks, they gave out beard nets.

Smithfield concealed infections with a policy of informing only employees whose work stations were in the same area as a person who tested positive. The fact that workers are required to move around was willfully ignored. One worker who tested positive said, “I clearly would have gotten it at the factory. This week I have worked on three different floors. I’ve eaten in two different cafeterias … I’ve been walking through the whole place.” Employees from the eighth floor of the plant were quarantined, but everyone else was told to keep working.

What Is Really Going On?

Average plant wages are around $16 an hour. Smithfield never raised them. Instead, they offered $500 to employees who could go all of April without an unapproved day off. The company says their “Responsibility Bonuses” show their “immense gratefulness” to employees “for their selfless sacrifices.”

Meanwhile, the local Argus Leader wrote union members wanted essential-worker hazard pay, which “would be considered hourly compensation about 1.5 or two times their normal pay.” One worker said, “I feel like they’re bribing us with [the bonus] to come to work sick. That’s how you know they don’t care.”

Both Sioux Falls workers killed by Covid-19 were in their sixties. It is unconscionable that they were still working. All meatpackers over 50 should be on paid leave. Agustín Rodríguez, 64, had a rough job sawing the legs off dead pigs. He mopped floors with a fever shortly before he was hospitalized.

When CEO Kenneth Sullivan closed the plant, he claimed, “We have continued to run our facilities for one reason: to sustain our nation’s food supply.” This is an effort to sweep Smithfield’s abuses under the rug, as if the company were operating for public benefit. This patriotic propaganda that all Americans are in it together is like a drug to keep workers from getting organized. </pre></details></div>

#### ✨ explanation 解释
This example illustrates the operator's ability to identify and remove near-duplicate documents based on their content. The operator computes SimHash values for each document and removes those that are too similar (based on Hamming distance). In this case, two sentences with very similar content but slight differences are considered duplicates, and only one of them is kept. 
这个例子说明了算子基于内容识别并移除近似重复文档的能力。算子为每个文档计算SimHash值，并根据汉明距离移除那些太相似的文档。在这个例子中，两个内容非常相似但有细微差别的句子被视为重复项，只保留其中一个。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/document_simhash_deduplicator.py)
- [unit test 单元测试](../../../tests/ops/deduplicator/test_document_simhash_deduplicator.py)
- [Return operator list 返回算子列表](../../Operators.md)