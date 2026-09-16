# fix_unicode_mapper

Fixes unicode errors in text samples.

This operator corrects common unicode errors and normalizes the text to a specified Unicode normalization form. The default normalization form is 'NFC', but it can be set to 'NFKC', 'NFD', or 'NFKD' during initialization. It processes text samples in batches, applying the specified normalization to each sample. If an unsupported normalization form is provided, a ValueError is raised.

修复文本样本中的Unicode错误。

此算子纠正常见的Unicode错误，并将文本标准化为指定的Unicode规范化形式。默认的规范化形式是'NFC'，但可以在初始化时设置为'NFKC'、'NFD'或'NFKD'。它以批量方式处理文本样本，对每个样本应用指定的规范化。如果提供了不支持的规范化形式，将引发ValueError。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `normalization` | <class 'str'> | `None` | the specified form of Unicode normalization mode, which can be one of ['NFC', 'NFKC', 'NFD', and 'NFKD'], default 'NFC'. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_bad_unicode_text
```python
FixUnicodeMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">âœ” No problems</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">✔ No problems</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa doesn&#x27;t have eyebrows.</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to fix common unicode errors in text. The input contains two texts with unicode issues, such as 'âœ”' and 'ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢', which are incorrectly displayed characters. After processing by the operator, these problematic characters are corrected to their intended forms, like '✔' and doesn't. This shows how the operator can normalize and correct unicode errors in text.
这个例子展示了算子修复文本中常见unicode错误的能力。输入包含两个带有unicode问题的文本，比如'âœ”'和'ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢'，这些都是显示不正确的字符。经过算子处理后，这些有问题的字符被纠正为它们应有的形式，如'✔'和doesn't。这说明了算子如何能够规范化并修正文本中的unicode错误。

### test_good_unicode_text
```python
FixUnicodeMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">No problems</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">阿里巴巴</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">No problems</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">阿里巴巴</pre></div>

#### ✨ explanation 解释
This example illustrates a case where the input texts do not contain any unicode errors. The first text is in English, and the second is in Chinese. Since there are no unicode issues present, the operator does not make any changes to the texts, and the output remains the same as the input. This demonstrates that the operator only applies corrections when necessary and leaves correctly formatted text unchanged.
这个例子展示了一个输入文本没有任何unicode错误的情况。第一个文本是英文，第二个是中文。由于不存在任何unicode问题，算子不会对文本进行任何更改，输出与输入保持一致。这说明了只有在必要时，算子才会应用修正，并且会保留格式正确的文本不变。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/fix_unicode_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_fix_unicode_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)