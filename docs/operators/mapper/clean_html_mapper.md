# clean_html_mapper

Cleans HTML code from text samples, converting HTML to plain text.

This operator processes text samples by removing HTML tags and converting HTML elements to a more readable format. Specifically, it replaces `<li>` and `<ol>` tags with newline and bullet points. The Selectolax HTML parser is used to extract the text content from the HTML. This operation is performed in a batched manner, making it efficient for large datasets.

将HTML代码从文本样本中清理，将HTML转换为纯文本。

此算子通过删除HTML标签并将HTML元素转换为更易读的格式来处理文本样本。具体来说，它将`<li>`和`<ol>`标签替换为换行符和项目符号。使用Selectolax HTML解析器从HTML中提取文本内容。此操作以批量方式执行，使其适用于大型数据集。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_complete_html_text
```python
CleanHtmlMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;header&gt;&lt;nav&gt;&lt;ul&gt;&lt;tile&gt;测试&lt;/title&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Home&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;About&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Services&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Contact&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/nav&gt;&lt;/header&gt;&lt;main&gt;&lt;h1&gt;Welcome to My Website&lt;/h1&gt;&lt;p&gt;Lorem ipsum dolor sit amet, consectetur adipiscing elit.&lt;button&gt;Learn More&lt;/button&gt;&lt;...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (74 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;header&gt;&lt;nav&gt;&lt;ul&gt;&lt;tile&gt;测试&lt;/title&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Home&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;About&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Services&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Contact&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/nav&gt;&lt;/header&gt;&lt;main&gt;&lt;h1&gt;Welcome to My Website&lt;/h1&gt;&lt;p&gt;Lorem ipsum dolor sit amet, consectetur adipiscing elit.&lt;button&gt;Learn More&lt;/button&gt;&lt;/main&gt;&lt;footer&gt;&lt;p&gt;&amp;copy; 2021 My Website. All Rights Reserved.&lt;/p&gt;&lt;/footer&gt;</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">测试
*Home
*About
*Services
*ContactWelcome to My WebsiteLorem ipsum dolor sit amet, consectetur adipiscing elit.Learn More© 2021 My Website. All Rights Reserved.</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to process a full HTML document, converting it into plain text. It removes all HTML tags and preserves the text content. The `<li>` tags are replaced with bullet points, and other elements like headers and paragraphs are flattened into a continuous string. This is useful for extracting readable text from web pages.
此示例展示了算子处理完整HTML文档的能力，将其转换为纯文本。它移除所有HTML标签并保留文本内容。`<li>`标签被替换为项目符号，而其他如标题和段落的元素则被展平成连续的字符串。这对于从网页中提取可读文本非常有用。

### test_no_html_text
```python
CleanHtmlMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a test</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是个测试</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">12345678</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a test</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是个测试</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">12345678</pre></div>

#### ✨ explanation 解释
In this example, the input data does not contain any HTML tags. As a result, the operator simply returns the original text without making any changes. This case illustrates that the operator can handle plain text inputs effectively, ensuring that non-HTML content remains unchanged.
在此示例中，输入数据不包含任何HTML标签。因此，算子直接返回原始文本，不做任何更改。这个案例说明了算子可以有效处理纯文本输入，确保非HTML内容保持不变。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/clean_html_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_clean_html_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)