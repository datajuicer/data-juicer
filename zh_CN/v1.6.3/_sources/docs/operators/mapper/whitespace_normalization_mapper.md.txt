# whitespace_normalization_mapper

Normalizes various types of whitespace characters to standard spaces in text samples.

This mapper converts all non-standard whitespace characters, such as tabs and newlines, to the standard space character (' ', 0x20). It also trims leading and trailing whitespace from the text. This ensures consistent spacing across all text samples, improving readability and consistency. The normalization process is based on a comprehensive list of whitespace characters, which can be found at https://en.wikipedia.org/wiki/Whitespace_character.

将文本样本中的各种空白字符标准化为空格。

该映射器将所有非标准空白字符（如制表符和换行符）转换为标准空格字符 (' ', 0x20)。它还修剪文本前后的空白。这确保了所有文本样本的一致间距，提高了可读性和一致性。规范化过程基于全面的空白字符列表，可以在 https://en.wikipedia.org/wiki/Whitespace_character 找到。

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
WhitespaceNormalizationMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">x 	              　​‌‍⁠￼y</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">x                       y</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to convert various non-standard whitespace characters, such as tabs and special spaces, into a standard space. In this case, all the unusual whitespace characters between 'x' and 'y' are replaced with a series of standard spaces, resulting in 'x                       y'. This makes the text more consistent and readable.
此示例展示了算子将各种非标准空白字符（例如制表符和特殊空格）转换为标准空格的能力。在这个例子中，'x' 和 'y' 之间的所有不寻常的空白字符都被替换为一系列的标准空格，结果是 'x                       y'。这使得文本更加一致且易于阅读。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/whitespace_normalization_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_whitespace_normalization_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)