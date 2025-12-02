# remove_words_with_incorrect_substrings_mapper

Mapper to remove words containing specified incorrect substrings.

This operator processes text by removing words that contain any of the specified incorrect substrings. By default, it removes words with substrings like "http", "www", ".com", "href", and "//". The operator can operate in tokenized or non-tokenized mode. In tokenized mode, it uses a Hugging Face tokenizer to tokenize the text before processing. The key metric is not computed; this operator focuses on filtering out specific words.

- If `tokenization` is True, the text is tokenized using a Hugging Face tokenizer, and words are filtered based on the specified substrings.
- If `tokenization` is False, the text is split into sentences and words, and words are filtered based on the specified substrings.
- The filtered text is then merged back into a single string.

The operator processes samples in batches and updates the text in place.

用于移除包含指定错误子字符串的单词的映射器。

该算子通过移除包含任何指定错误子字符串的单词来处理文本。默认情况下，它移除包含 "http", "www", ".com", "href", 和 "//" 等子字符串的单词。算子可以在分词或非分词模式下运行。在分词模式下，它使用 Hugging Face 分词器对文本进行分词后再处理。关键指标不计算；此算子专注于过滤特定单词。

- 如果 `tokenization` 为 True，则使用 Hugging Face 分词器对文本进行分词，并基于指定的子字符串过滤单词。
- 如果 `tokenization` 为 False，则将文本拆分为句子和单词，并基于指定的子字符串过滤单词。
- 过滤后的文本然后合并成一个字符串。

算子批量处理样本并在原地更新文本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `lang` | <class 'str'> | `'en'` | sample in which language |
| `tokenization` | <class 'bool'> | `False` | whether to use model to tokenize documents |
| `substrings` | typing.Optional[typing.List[str]] | `None` | The incorrect substrings in words. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_en_case
```python
RemoveWordsWithIncorrectSubstringsMapper(substrings=['http', 'www', '.com', 'href', '//'])
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel https://whiugc.com method on LLM</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">plusieurs èrdash@hqbchd.ckd d&#x27;accéder à ces wwwasdasd fonc</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel method on LLM</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">plusieurs èrdash@hqbchd.ckd d&#x27;accéder à ces fonc</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to remove words containing specified incorrect substrings such as 'http', 'www', and '.com' from English text. In the first sample, 'https://whiugc.com' is removed because it contains 'http' and '.com'. In the second sample, 'wwwasdasd' is removed because it contains 'www'. The resulting text is free of these incorrect substrings.
这个例子展示了算子从英文文本中移除包含指定错误子字符串（如'http'、'www'和'.com'）的单词的能力。在第一个样本中，'https://whiugc.com'被移除，因为它包含了'http'和'.com'。在第二个样本中，'wwwasdasd'被移除，因为它包含了'www'。结果文本中不再包含这些错误子字符串。

### test_zh_case
```python
RemoveWordsWithIncorrectSubstringsMapper(lang='zh', tokenization=True, substrings=['com', '算子'])
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">你好，请问你是谁</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">根据算子使用情况增量安装方案确定</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">请用百度www.baidu.com进行搜索</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">你好，请问你是谁</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">根据使用情况增量安装方案确定</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">请用百度www.baidu.进行搜索</pre></div>

#### ✨ explanation 解释
This example shows how the operator works with Chinese text when tokenization is enabled. It removes words containing specified incorrect substrings such as 'com' and '算子'. In the third sample, '算子' is removed. In the fourth sample, '.com' is partially removed, leaving 'www.baidu.'. The resulting text is free of these incorrect substrings.
这个例子展示了当启用分词时，算子如何处理中文文本。它移除包含指定错误子字符串（如'com'和'算子'）的单词。在第三个样本中，'算子'被移除。在第四个样本中，'.com'部分被移除，留下了'www.baidu.'。结果文本中不再包含这些错误子字符串。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_words_with_incorrect_substrings_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_words_with_incorrect_substrings_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)