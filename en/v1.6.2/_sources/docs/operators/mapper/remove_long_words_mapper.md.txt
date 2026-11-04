# remove_long_words_mapper

Mapper to remove long words within a specific range.

This operator filters out words in the text that are either shorter than the specified minimum length or longer than the specified maximum length. Words are first checked with their original length, and if they do not meet the criteria, they are stripped of special characters and re-evaluated. The key metric used is the character-based length of each word. The processed text retains only the words that fall within the defined length range. This operator processes text in batches for efficiency.

映射器，移除特定范围内的长词。

该算子过滤掉文本中长度短于指定最小长度或长于指定最大长度的单词。首先检查单词的原始长度，如果不满足条件，则剥离特殊字符后重新评估。使用的关键指标是每个单词基于字符的长度。处理后的文本只保留符合定义长度范围的单词。该算子批量处理文本以提高效率。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_len` | <class 'int'> | `1` | The min mapper word length in this op, words will be filtered if their length is below this parameter. |
| `max_len` | <class 'int'> | `9223372036854775807` | The max mapper word length in this op, words will be filtered if their length exceeds this parameter. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_normal_case
```python
RemoveLongWordsMapper(min_len=3, max_len=15)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed novel method LLM pretraining.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed novel method LLM pretraining.</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's behavior when all words in the text fall within the specified length range (3 to 15 characters). As a result, no words are removed from the input text, and the output is identical to the input.
这个例子展示了当文本中的所有单词都在指定的长度范围内（3到15个字符）时，算子的行为。因此，输入文本中没有单词被移除，输出与输入完全相同。

### test_special_words_case
```python
RemoveLongWordsMapper(min_len=3, max_len=15)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel eqeqweqwewqenhq😊😠 method on LLM.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur la plateforme MT4, plusieurs manières d&#x27;accéder0123813976125</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed novel eqeqweqwewqenhq😊😠 method LLM.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur plateforme MT4, plusieurs manières d&#x27;accéder0123813976125</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa have eyebrows.</pre></div>

#### ✨ explanation 解释
This example illustrates how the operator handles special characters and very long or short words. Words that do not initially meet the length criteria (like 'doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t' being too long) are stripped of special characters and re-evaluated. If they then fit the length criteria, they are kept; otherwise, they are removed. The presence of emojis and numbers does not affect their evaluation as long as the total character count is within the allowed range.
这个例子说明了算子如何处理特殊字符以及非常长或短的单词。最初不符合长度标准的单词（如'doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t'太长）会被去除特殊字符并重新评估。如果它们之后符合长度标准，则保留；否则，将被移除。只要总字符数在允许的范围内，表情符号和数字的存在不会影响它们的评估。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_long_words_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_long_words_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)