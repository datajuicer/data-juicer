# character_repetition_filter

Filter to keep samples with character-level n-gram repetition ratio within a specific range.

This operator calculates the character-level n-gram repetition ratio for each sample and filters out samples that do not fall within the specified range. The repetition ratio is computed based on the frequency of n-grams in the text. The key metric 'char_rep_ratio' is cached in the stats field. Samples are kept if their 'char_rep_ratio' is between the specified min and max ratios. The n-gram length, minimum, and maximum ratios are configurable.

保留字符级n-gram重复率在特定范围内的样本。

该算子计算每个样本的字符级n-gram重复率，并过滤掉不在指定范围内的样本。重复率基于文本中n-gram的频率计算。关键指标'char_rep_ratio'缓存在统计字段中。如果样本的'char_rep_ratio'在指定的最小值和最大值之间，则保留该样本。n-gram长度、最小值和最大值比率是可配置的。

Type 算子类型: **filter**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `rep_len` | typing.Annotated[int, Gt(gt=0)] | `10` | Repetition length for char-level n-gram. |
| `min_ratio` | <class 'float'> | `0.0` | The min filter ratio in this op, samples will be filtered if their char-level n-gram repetition ratio is below this parameter. |
| `max_ratio` | <class 'float'> | `0.5` | The max filter ratio in this op, samples will be filtered if their char-level n-gram repetition ratio exceeds this parameter. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_case
```python
CharacterRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=0.4, batch_size=2)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sund Sund Sund Sund Sund Sunda and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a v s e c s f e f g a a a a a a a a a a</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">中文也是一个字算一个长度</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">中文也是一个字算一个长度</pre></div>

#### ✨ explanation 解释
The CharacterRepetitionFilter operator filters out samples based on the character-level 5-gram repetition ratio, keeping only those with a ratio between 0.0 and 0.4. The first two texts are removed because their 5-gram repetition ratios exceed 0.4, while the last two texts have ratios within the specified range, thus they are kept.
CharacterRepetitionFilter算子基于字符级别的5-gram重复率过滤样本，仅保留重复率在0.0到0.4之间的样本。前两个文本因为其5-gram重复率超过了0.4而被移除，而后两个文本的重复率位于指定范围内，因此被保留。

### test_existing_stats
```python
CharacterRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=0.4, batch_size=2)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sund Sund Sund Sund Sund Sunda and it&#x27;s a happy day!</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__stats__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>char_rep_ratio</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0.5</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a v s e c s f e f g a a a a a a a a a a</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__stats__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>char_rep_ratio</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>0.5</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[]</pre></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/character_repetition_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_character_repetition_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)