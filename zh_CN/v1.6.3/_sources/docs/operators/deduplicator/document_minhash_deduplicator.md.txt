# document_minhash_deduplicator

Deduplicates samples at the document level using MinHash LSH.

This operator computes MinHash values for each sample and uses Locality-Sensitive Hashing (LSH) to identify and remove near-duplicate documents. The Jaccard similarity threshold determines when two documents are considered duplicates. The tokenization method can be customized, and a Hugging Face tokenizer can be used for 'sentencepiece' tokenization. The minhash values are stored as bytes and are not kept in the final dataset. The number of bands and rows per band in LSH can be set manually or determined by an optimal parameter computation algorithm. Important notes:
- If using 'punctuation' tokenization with an ignore pattern, ensure the pattern does not include punctuations.
- For 'sentencepiece' tokenization, a tokenizer model path is required.
- The deduplication process involves clustering and filtering, and only unique samples or the first sample in a cluster are retained.

使用MinHash LSH在文档级别去重样本。

该算子为每个样本计算MinHash值，并使用局部敏感哈希（LSH）来识别和删除近似重复的文档。Jaccard相似度阈值决定了两个文档何时被视为重复。可以自定义分词方法，并且可以使用Hugging Face的分词器进行'sentencepiece'分词。MinHash值以字节形式存储，不会保留在最终数据集中。可以在LSH中手动设置或通过最优参数计算算法确定区段数和每区段行数。重要提示：
- 如果使用带有忽略模式的'punctuation'分词，请确保该模式不包括标点符号。
- 对于'sentencepiece'分词，需要提供一个分词器模型路径。
- 去重过程涉及聚类和过滤，仅保留唯一样本或每个聚类中的第一个样本。

Type 算子类型: **deduplicator**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tokenization` | <class 'str'> | `'space'` | tokenization method for sample texts. It should be one of [space, punctuation, character, sentencepiece]. For English-like languages, we recommend to use 'space', for Chinese-like languages, we recommend to use 'character', and for multiple languages, we recommend to use 'sentencepiece'. If using 'sentencepiece', please provided the model path in the 'tokenizer_model' field. |
| `window_size` | typing.Annotated[int, Gt(gt=0)] | `5` | window size of shingling |
| `lowercase` | <class 'bool'> | `True` | whether to convert text to lower case first |
| `ignore_pattern` | typing.Optional[str] | `None` | whether to ignore sub-strings with specific pattern when computing minhash |
| `num_permutations` | typing.Annotated[int, Gt(gt=0)] | `256` | number of permutations in minhash computing |
| `jaccard_threshold` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `0.7` | the min jaccard similarity threshold in near-duplicate detection. When the jaccard similarity of two sample texts is >= this threshold, they are regarded as similar samples and this op will only keep one of them after deduplication |
| `num_bands` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | number of bands in LSH. Default it's None, and it will be determined by an optimal params computation algorithm by minimize the weighted sum of probs of False Positives and False Negatives |
| `num_rows_per_band` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | number of rows in each band in LSH. Default it's None, and it will be determined by an optimal params computation algorithm |
| `tokenizer_model` | typing.Optional[str] | `None` | path for the sentencepiece model, used for sentencepiece tokenization. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_english_deduplication
```python
DocumentMinhashDeduplicator(ignore_pattern='\\p{P}')
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
The operator computes MinHash values and uses LSH to identify and remove near-duplicate documents. The Jaccard similarity threshold is used to determine duplicates. In this case, the long texts about the Smithfield plant are considered duplicates and only one of them is kept, while the shorter, distinct texts remain in the dataset.
算子计算MinHash值并使用LSH来识别和移除近似重复的文档。Jaccard相似度阈值用于确定哪些是重复项。在这个例子中，关于Smithfield工厂的长篇文本被认为是重复的，只保留其中一个，而其他较短且不同的文本则保留在数据集中。

### test_chinese_deduplication
```python
DocumentMinhashDeduplicator(tokenization='character', ignore_pattern='\\p{P}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">你好，请问你是谁</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">第九届会议
2003年7月28日至8月8日
牙买加金斯敦
为来自发展中国家的法律和技术委员会以及财务委员会成员
参加委员会会议支付费用的方式
1. 国际海底管理局大会第八届会议请秘书长采取一项临时措施，设立一个自愿信托基金，以便支付来自发展中国家的法律和技术委员会成员以及来自发展中国家的财务委员会成员参加委员会会议的费用。
2. 由于秘书长向会员国发出为该信托基金捐款的请求，已收到三笔捐款，共计10 500美元。 管理局已为基金设立一个单独的账户。
3. 管理局第八届会议还决定，由财务委员会审查资助参加这两个委员会会议的方式，包括审查是否可能从管理局行政预算中提供经费。
4. 自愿信托基金迄今...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (986 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">第九届会议
2003年7月28日至8月8日
牙买加金斯敦
为来自发展中国家的法律和技术委员会以及财务委员会成员
参加委员会会议支付费用的方式
1. 国际海底管理局大会第八届会议请秘书长采取一项临时措施，设立一个自愿信托基金，以便支付来自发展中国家的法律和技术委员会成员以及来自发展中国家的财务委员会成员参加委员会会议的费用。
2. 由于秘书长向会员国发出为该信托基金捐款的请求，已收到三笔捐款，共计10 500美元。 管理局已为基金设立一个单独的账户。
3. 管理局第八届会议还决定，由财务委员会审查资助参加这两个委员会会议的方式，包括审查是否可能从管理局行政预算中提供经费。
4. 自愿信托基金迄今收到的捐款数额很小。 这两个委员会成员虽然由缔约国提名，但他们以个人身份当选。 因此，必须确保这些机构的成员在任期内能够参加会议并且持续不断地履行职务。 现已注意到，这两个委员会若干成员因旅费和生活津贴费用方面有困难而未能出席会议。 来自发展中国家成员参加会议的费用估计数见附件，其中比较了经济舱和公务舱机票价格以及适用于金斯敦的每日生活津贴费用。 从表中可以看出，根据不同的人数、机舱等级和会议持续时间，每年平均需要捐款120 000美元至215 000美元。
5. 为了指导委员会确定提供经费的方式，对某些国际组织的现行办法作了一次简要调查。 为支付参加会议的旅费和生活费而设立信托基金最相关的实例是2000年大会为来自发展中国家的大陆架界限委员会成员设立的自愿信托基金。 目前这一基金正在运作，但现有资源有限。 联合国制定的程序表明，委员会成员的政府应在规定时间内尽可能提前提出请求。 这种请求按照先到先核可的办法处理。 提供的机票将是最直接路线的经济舱机票，每日生活津贴将按照联合国费率提供。 购买机票的所有安排均由联合国秘书处执行。
6. 虽然已经设立了临时性的自愿信托基金，但是，对该基金的捐款数额很小，捐款速度很慢。 因此，除了对信托基金提供自愿捐款的办法之外，建议委员会还可以考虑采用下列办法：
(a) 从管理局一般行政经费累计利息中拨出一定数额的经费；
(b) 每年从上一年预算未动用部分中拨出规定的数额；
(c) 从先驱投资者基金利息中拨出规定的数额。
7. 委员会还不妨建议由管理局秘书处依照行政规则和程序管理该基金，并向财务委员会提出一份报告。
附件
资助来自发展中国家的法律和技术委员会以及财务
委员会成员出席会议的指示性费用（美元）
成员
机票
机场
费用
金斯敦每日生活
津贴
转机途中每日生活
7日
共计
14日
经济舱
公务舱
7天=(8天每日生活
津贴)
14天= (15天每日生活津贴)
商务舱
法律和技术委员会
印度尼西亚
(纽约)
黎巴嫩
巴基斯坦
阿根廷
喀麦隆
墨西哥
巴西
塞内加尔
莫桑比克
埃及(纽约)
大韩民国
印度
斐济
智利
中国
纳米比亚
小计
财务委员会
缅甸
乌干达
牙买加
印度(纽约)
尼日利亚
总计
注：估计费用表表明每年资助每个机构一次会议需要经费120 000美元至215 000美元(四舍五入)。</pre></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">第九届会议
时间：2003年7月28日至8月8日
牙买加金斯敦
为来自发展中国家的法律和技术委员会以及财务委员会成员
参加委员会会议支付费用的方式
1. 国际海底管理局大会第八届会议请秘书长采取一项临时措施，设立一个自愿信托基金，以便支付来自发展中国家的法律和技术委员会成员以及来自发展中国家的财务委员会成员参加委员会会议的费用。
2. 由于秘书长向会员国发出为该信托基金捐款的请求，已收到三笔捐款，共计10 500美元。 管理局已为基金设立一个单独的账户。
3. 管理局第八届会议还决定，由财务委员会审查资助参加这两个委员会会议的方式，包括审查是否可能从管理局行政预算中提供经费。
4. 自愿信托基...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (989 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">第九届会议
时间：2003年7月28日至8月8日
牙买加金斯敦
为来自发展中国家的法律和技术委员会以及财务委员会成员
参加委员会会议支付费用的方式
1. 国际海底管理局大会第八届会议请秘书长采取一项临时措施，设立一个自愿信托基金，以便支付来自发展中国家的法律和技术委员会成员以及来自发展中国家的财务委员会成员参加委员会会议的费用。
2. 由于秘书长向会员国发出为该信托基金捐款的请求，已收到三笔捐款，共计10 500美元。 管理局已为基金设立一个单独的账户。
3. 管理局第八届会议还决定，由财务委员会审查资助参加这两个委员会会议的方式，包括审查是否可能从管理局行政预算中提供经费。
4. 自愿信托基金迄今收到的捐款数额很小。 这两个委员会成员虽然由缔约国提名，但他们以个人身份当选。 因此，必须确保这些机构的成员在任期内能够参加会议并且持续不断地履行职务。 现已注意到，这两个委员会若干成员因旅费和生活津贴费用方面有困难而未能出席会议。 来自发展中国家成员参加会议的费用估计数见附件，其中比较了经济舱和公务舱机票价格以及适用于金斯敦的每日生活津贴费用。 从表中可以看出，根据不同的人数、机舱等级和会议持续时间，每年平均需要捐款120 000美元至215 000美元。
5. 为了指导委员会确定提供经费的方式，对某些国际组织的现行办法作了一次简要调查。 为支付参加会议的旅费和生活费而设立信托基金最相关的实例是2000年大会为来自发展中国家的大陆架界限委员会成员设立的自愿信托基金。 目前这一基金正在运作，但现有资源有限。 联合国制定的程序表明，委员会成员的政府应在规定时间内尽可能提前提出请求。 这种请求按照先到先核可的办法处理。 提供的机票将是最直接路线的经济舱机票，每日生活津贴将按照联合国费率提供。 购买机票的所有安排均由联合国秘书处执行。
6. 虽然已经设立了临时性的自愿信托基金，但是，对该基金的捐款数额很小，捐款速度很慢。 因此，除了对信托基金提供自愿捐款的办法之外，建议委员会还可以考虑采用下列办法：
(a) 从管理局一般行政经费累计利息中拨出一定数额的经费；
(b) 每年从上一年预算未动用部分中拨出规定的数额；
(c) 从先驱投资者基金利息中拨出规定的数额。
7. 委员会还不妨建议由管理局秘书处依照行政规则和程序管理该基金，并向财务委员会提出一份报告。
附件
资助来自发展中国家的法律和技术委员会以及财务
委员会成员出席会议的指示性费用（美元）
成员
机票
机场
费用
金斯敦每日生活
津贴
转机途中每日生活
7日
共计
14日
经济舱
公务舱
7天=(8天每日生活
津贴)
14天= (15天每日生活津贴)
商务舱
法律和技术委员会
印度尼西亚
(纽约)
黎巴嫩
巴基斯坦
阿根廷
喀麦隆
墨西哥
巴西
塞内加尔
莫桑比克
埃及(纽约)
大韩民国
印度
斐济
智利
中国
纳米比亚
小计
财务委员会
缅甸
乌干达
牙买加
印度(纽约)
尼日利亚
总计
注：估计费用表表明每年资助每个机构一次会议需要经费120 000美元至215 000美元(四舍五入)。</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">你好，请问你是谁</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">第九届会议
2003年7月28日至8月8日
牙买加金斯敦
为来自发展中国家的法律和技术委员会以及财务委员会成员
参加委员会会议支付费用的方式
1. 国际海底管理局大会第八届会议请秘书长采取一项临时措施，设立一个自愿信托基金，以便支付来自发展中国家的法律和技术委员会成员以及来自发展中国家的财务委员会成员参加委员会会议的费用。
2. 由于秘书长向会员国发出为该信托基金捐款的请求，已收到三笔捐款，共计10 500美元。 管理局已为基金设立一个单独的账户。
3. 管理局第八届会议还决定，由财务委员会审查资助参加这两个委员会会议的方式，包括审查是否可能从管理局行政预算中提供经费。
4. 自愿信托基金迄今...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (986 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">第九届会议
2003年7月28日至8月8日
牙买加金斯敦
为来自发展中国家的法律和技术委员会以及财务委员会成员
参加委员会会议支付费用的方式
1. 国际海底管理局大会第八届会议请秘书长采取一项临时措施，设立一个自愿信托基金，以便支付来自发展中国家的法律和技术委员会成员以及来自发展中国家的财务委员会成员参加委员会会议的费用。
2. 由于秘书长向会员国发出为该信托基金捐款的请求，已收到三笔捐款，共计10 500美元。 管理局已为基金设立一个单独的账户。
3. 管理局第八届会议还决定，由财务委员会审查资助参加这两个委员会会议的方式，包括审查是否可能从管理局行政预算中提供经费。
4. 自愿信托基金迄今收到的捐款数额很小。 这两个委员会成员虽然由缔约国提名，但他们以个人身份当选。 因此，必须确保这些机构的成员在任期内能够参加会议并且持续不断地履行职务。 现已注意到，这两个委员会若干成员因旅费和生活津贴费用方面有困难而未能出席会议。 来自发展中国家成员参加会议的费用估计数见附件，其中比较了经济舱和公务舱机票价格以及适用于金斯敦的每日生活津贴费用。 从表中可以看出，根据不同的人数、机舱等级和会议持续时间，每年平均需要捐款120 000美元至215 000美元。
5. 为了指导委员会确定提供经费的方式，对某些国际组织的现行办法作了一次简要调查。 为支付参加会议的旅费和生活费而设立信托基金最相关的实例是2000年大会为来自发展中国家的大陆架界限委员会成员设立的自愿信托基金。 目前这一基金正在运作，但现有资源有限。 联合国制定的程序表明，委员会成员的政府应在规定时间内尽可能提前提出请求。 这种请求按照先到先核可的办法处理。 提供的机票将是最直接路线的经济舱机票，每日生活津贴将按照联合国费率提供。 购买机票的所有安排均由联合国秘书处执行。
6. 虽然已经设立了临时性的自愿信托基金，但是，对该基金的捐款数额很小，捐款速度很慢。 因此，除了对信托基金提供自愿捐款的办法之外，建议委员会还可以考虑采用下列办法：
(a) 从管理局一般行政经费累计利息中拨出一定数额的经费；
(b) 每年从上一年预算未动用部分中拨出规定的数额；
(c) 从先驱投资者基金利息中拨出规定的数额。
7. 委员会还不妨建议由管理局秘书处依照行政规则和程序管理该基金，并向财务委员会提出一份报告。
附件
资助来自发展中国家的法律和技术委员会以及财务
委员会成员出席会议的指示性费用（美元）
成员
机票
机场
费用
金斯敦每日生活
津贴
转机途中每日生活
7日
共计
14日
经济舱
公务舱
7天=(8天每日生活
津贴)
14天= (15天每日生活津贴)
商务舱
法律和技术委员会
印度尼西亚
(纽约)
黎巴嫩
巴基斯坦
阿根廷
喀麦隆
墨西哥
巴西
塞内加尔
莫桑比克
埃及(纽约)
大韩民国
印度
斐济
智利
中国
纳米比亚
小计
财务委员会
缅甸
乌干达
牙买加
印度(纽约)
尼日利亚
总计
注：估计费用表表明每年资助每个机构一次会议需要经费120 000美元至215 000美元(四舍五入)。</pre></details></div>

#### ✨ explanation 解释
该算子通过计算MinHash值并使用LSH来识别和移除近似重复的文档。Jaccard相似度阈值用于判断文档是否为重复。在本例中，两段关于会议费用支付方式的中文长文本被识别为重复，因此仅保留了一段，而另外两条不同内容的简短信件保持不变。
The operator computes MinHash values and uses LSH to identify and remove near-duplicate documents. The Jaccard similarity threshold is used to determine duplicates. In this case, two long Chinese texts about the meeting expense payment method are identified as duplicates, thus only one is retained, while the other two short and different messages remain unchanged.


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/document_minhash_deduplicator.py)
- [unit test 单元测试](../../../tests/ops/deduplicator/test_document_minhash_deduplicator.py)
- [Return operator list 返回算子列表](../../Operators.md)