# chinese_convert_mapper

Mapper to convert Chinese text between Traditional, Simplified, and Japanese Kanji.

This operator converts Chinese text based on the specified mode. It supports conversions between Simplified Chinese, Traditional Chinese (including Taiwan and Hong Kong variants), and Japanese Kanji. The conversion is performed using a pre-defined set of rules. The available modes include 's2t' for Simplified to Traditional, 't2s' for Traditional to Simplified, and other specific variants like 's2tw', 'tw2s', 's2hk', 'hk2s', 's2twp', 'tw2sp', 't2tw', 'tw2t', 'hk2t', 't2hk', 't2jp', and 'jp2t'. The operator processes text in batches and applies the conversion to the specified text key in the samples.

转换中文文本在繁体、简体和日文汉字之间的映射器。

该算子根据指定的模式转换中文文本。它支持简体中文、繁体中文（包括台湾和香港变体）以及日文汉字之间的转换。转换使用预定义的一组规则进行。可用模式包括：'s2t' 从简体到繁体，'t2s' 从繁体到简体，以及其他特定变体如 's2tw', 'tw2s', 's2hk', 'hk2s', 's2twp', 'tw2sp', 't2tw', 'tw2t', 'hk2t', 't2hk', 't2jp', 和 'jp2t'。该算子批量处理文本，并对样本中指定的文本键应用转换。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mode` | <class 'str'> | `'s2t'` | Choose the mode to convert Chinese: s2t: Simplified Chinese to Traditional Chinese,  t2s: Traditional Chinese to Simplified Chinese,  s2tw: Simplified Chinese to Traditional Chinese (Taiwan Standard),  tw2s: Traditional Chinese (Taiwan Standard) to Simplified Chinese,  s2hk: Simplified Chinese to Traditional Chinese (Hong Kong variant),  hk2s: Traditional Chinese (Hong Kong variant) to Simplified Chinese,  s2twp: Simplified Chinese to Traditional Chinese (Taiwan Standard) with Taiwanese idiom,  tw2sp: Traditional Chinese (Taiwan Standard) to Simplified Chinese with Mainland Chinese idiom,  t2tw: Traditional Chinese to Traditional Chinese (Taiwan Standard),  tw2t: Traditional Chinese (Taiwan standard) to Traditional Chinese,  hk2t: Traditional Chinese (Hong Kong variant) to Traditional Chinese,  t2hk: Traditional Chinese to Traditional Chinese (Hong Kong variant),  t2jp: Traditional Chinese Characters (Kyūjitai) to New Japanese Kanji,  jp2t: New Japanese Kanji (Shinjitai) to Traditional Chinese Characters, |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_s2t
```python
ChineseConvertMapper('s2t')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是几个简体字，会被转换为繁体字</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">如果本身都是繁體字就不會被轉換</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">试试繁体afadf字$#@#和简体字，以及各123213*&amp;dasd種不同字符数字的组合轉換效果</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">這是幾個簡體字，會被轉換爲繁體字</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">如果本身都是繁體字就不會被轉換</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">試試繁體afadf字$#@#和簡體字，以及各123213*&amp;dasd種不同字符數字的組合轉換效果</pre></div>

#### ✨ explanation 解释
This method converts Simplified Chinese text to Traditional Chinese. If the input text is already in Traditional Chinese, it remains unchanged. Non-Chinese characters and symbols are not converted. For example, '这是几个简体字，会被转换为繁体字' is converted to '這是幾個簡體字，會被轉換爲繁體字', while '如果本身都是繁體字就不會被轉換' stays the same.
这个方法将简体中文文本转换为繁体中文。如果输入文本已经是繁体中文，则保持不变。非中文字符和符号不会被转换。例如，“这是几个简体字，会被转换为繁体字”被转换为“這是幾個簡體字，會被轉換爲繁體字”，而“如果本身都是繁體字就不會被轉換”则保持不变。

### test_t2jp
```python
ChineseConvertMapper('s2t')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">他需要修復心臟瓣膜</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">舊字體歷史假名遣 新字體現代假名遣</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">藝術 缺航 飲料罐</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">他需要修復心臓弁膜</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">旧字体歴史仮名遣 新字体現代仮名遣</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">芸術 欠航 飲料缶</pre></div>

#### ✨ explanation 解释
This method converts Traditional Chinese text to Japanese Kanji. The conversion is based on a set of predefined rules. For example, '他需要修復心臟瓣膜' is converted to '他需要修復心臓弁膜'. Some characters that have different forms in Japanese, such as '舊' to '旧', are also converted accordingly. Non-convertible characters and symbols remain unchanged.
这个方法将繁体中文文本转换为日文汉字。转换基于一组预定义的规则。例如，“他需要修復心臟瓣膜”被转换为“他需要修復心臓弁膜”。一些在日语中有不同形式的字符，如“舊”到“旧”，也会相应地进行转换。不可转换的字符和符号保持不变。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/chinese_convert_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_chinese_convert_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)