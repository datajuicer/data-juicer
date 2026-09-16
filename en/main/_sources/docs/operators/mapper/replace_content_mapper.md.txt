# replace_content_mapper

Replaces content in the text that matches a specific regular expression pattern with a designated replacement string.

This operator processes text by searching for patterns defined in `pattern` and replacing them with the corresponding `repl` string. If multiple patterns and replacements are provided, each pattern is replaced by its respective replacement. The operator supports both single and multiple patterns and replacements. The regular expressions are compiled with the `re.DOTALL` flag to match across multiple lines. If the length of the patterns and replacements do not match, a `ValueError` is raised. This operation is batched, meaning it processes multiple samples at once.

用指定的替换字符串替换与特定正则表达式模式匹配的内容。

该算子通过搜索 `pattern` 中定义的模式并将它们替换为相应的 `repl` 字符串来处理文本。如果提供了多个模式和替换字符串，则每个模式都会被其对应的替换字符串替换。算子支持单个和多个模式及替换字符串。正则表达式使用 `re.DOTALL` 标志编译，以便跨多行匹配。如果模式和替换字符串的长度不匹配，则会引发 `ValueError`。此操作是批处理的，即一次处理多个样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `pattern` | typing.Union[str, typing.List[str], NoneType] | `None` | regular expression pattern(s) to search for within text |
| `repl` | typing.Union[str, typing.List[str]] | `''` | replacement string(s), default is empty string |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_special_char_pattern_text
```python
ReplaceContentMapper(pattern='●■', repl='<SPEC>')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个干净的文本。Including Chinese and English.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">◆●■►▼▲▴∆▻▷❖♡□</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">多个●■►▼这样的特殊字符可以►▼▲▴∆吗？</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">未指定的●■☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个干净的文本。Including Chinese and English.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">◆&lt;SPEC&gt;►▼▲▴∆▻▷❖♡□</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">多个&lt;SPEC&gt;►▼这样的特殊字符可以►▼▲▴∆吗？</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">未指定的&lt;SPEC&gt;☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？</pre></div>

#### ✨ explanation 解释
This example demonstrates how the operator replaces specific special characters (in this case, '●■') with a designated replacement string ('<SPEC>'). The operator scans through the text and replaces all occurrences of the specified pattern. In the output, we can see that only the targeted special characters are replaced, while other parts of the text remain unchanged. This is a typical use case for cleaning or standardizing text data.
此示例展示了算子如何将特定的特殊字符（此处为'●■'）替换为指定的字符串（'<SPEC>'）。算子会扫描文本并将所有出现的目标模式替换成指定的字符串。在输出中，我们可以看到只有目标特殊字符被替换，而文本的其他部分保持不变。这是清理或标准化文本数据的一个典型用例。

### test_raw_digit_pattern_text
```python
ReplaceContentMapper(pattern='\\d+(?:,\\d+)*', repl='<DIGIT>')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个123。Including 456 and English.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个&lt;DIGIT&gt;。Including &lt;DIGIT&gt; and English.</pre></div>

#### ✨ explanation 解释
In this example, the operator replaces sequences of digits (e.g., '123', '456') in the text with a designated replacement string ('<DIGIT>'). The regular expression used here matches any sequence of digits, including those separated by commas. The result shows that all digit sequences are replaced with '<DIGIT>', which is useful for anonymization or generalization of numeric information in the text. Note that the actual raw output from the operator is the modified text; the test file further compares this output to an expected target to ensure correctness.
在此示例中，算子将文本中的数字序列（例如'123'、'456'）替换为指定的字符串（'<DIGIT>'）。这里使用的正则表达式匹配任何数字序列，包括由逗号分隔的序列。结果表明所有的数字序列都被替换成了'<DIGIT>'，这对于文本中数值信息的匿名化或泛化非常有用。请注意，算子的实际原始输出是修改后的文本；测试文件进一步将此输出与预期目标进行比较以确保正确性。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/replace_content_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_replace_content_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)