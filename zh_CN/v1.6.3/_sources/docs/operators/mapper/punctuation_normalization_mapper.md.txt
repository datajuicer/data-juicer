# punctuation_normalization_mapper

Normalizes unicode punctuations to their English equivalents in text samples.

This operator processes a batch of text samples and replaces any unicode punctuation with its corresponding English punctuation. The mapping includes common substitutions like "，" to ",", "。" to ".", and "“" to ". It iterates over each character in the text, replacing it if it is found in the predefined punctuation map. The result is a set of text samples with consistent punctuation formatting.

将文本样本中的Unicode标点符号标准化为其英文等效符号。

该算子处理一批文本样本，并将任何Unicode标点符号替换为其对应的英文标点符号。映射包括常见的替换，如"，"替换为", "，"。"替换为"."，以及"“"替换为"。它遍历文本中的每个字符，如果在预定义的标点映射中找到，则进行替换。结果是一组具有统一标点格式的文本样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_case
```python
PunctuationNormalizationMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">,.,&quot;&quot;&quot;&quot;&quot;&quot;&quot;&quot;&quot;&quot;&#x27;::?!();- - . ~&#x27;...-&lt;&gt;[]%-</pre></div>

#### ✨ explanation 解释
This example demonstrates how the PunctuationNormalizationMapper operator converts various Unicode punctuation marks into their English equivalents. The input text contains a series of non-English punctuation marks, and after processing, these are replaced with similar English punctuation marks. For instance, '，' is changed to ',', '。' to '.', and '“”' to '"'. This normalization ensures that the text follows a consistent punctuation style, making it easier to process or analyze further.
这个例子展示了PunctuationNormalizationMapper算子如何将各种Unicode标点符号转换成它们对应的英文标点符号。输入文本包含一系列非英文的标点符号，在处理后，这些符号被替换为相似的英文标点符号。例如，'，' 被改为 ','，'。' 被改为 '.'，以及 '“”' 被改为 '"'。这种规范化确保了文本遵循一致的标点样式，使得进一步处理或分析变得更加容易。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/punctuation_normalization_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_punctuation_normalization_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)