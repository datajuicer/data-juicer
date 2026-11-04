# extract_tables_from_html_mapper

Extracts tables from HTML content and stores them in a specified field.

This operator processes HTML content to extract tables. It can either retain or remove HTML tags based on the `retain_html_tags` parameter. If `retain_html_tags` is False, it can also include or exclude table headers based on the `include_header` parameter. The extracted tables are stored in the `tables_field_name` field within the sample's metadata. If no tables are found, an empty list is stored. If the tables have already been extracted, the operator will not reprocess the sample.

从HTML内容中提取表格并存储在指定字段中。

此算子处理HTML内容以提取表格。根据`retain_html_tags`参数，它可以保留或移除HTML标签。如果`retain_html_tags`为False，还可以根据`include_header`参数选择包含或排除表格标题。提取的表格将存储在样本元数据中的`tables_field_name`字段内。如果没有找到表格，则会存储一个空列表。如果表格已经被提取，算子将不会重新处理样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tables_field_name` | <class 'str'> | `'html_tables'` | Field name to store the extracted tables. |
| `retain_html_tags` | <class 'bool'> | `False` | If True, retains HTML tags in the tables; otherwise, removes them. |
| `include_header` | <class 'bool'> | `True` | If True, includes the table header; otherwise, excludes it. This parameter is effective             only when `retain_html_tags` is False and applies solely to the extracted table content. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_extract_tables_include_header
```python
ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">
    &lt;!DOCTYPE html&gt;
            &lt;html lang=&quot;zh&quot;&gt;
            &lt;head&gt;
                &lt;meta charset=&quot;UTF-8&quot;&gt;
                &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;&gt;
                &lt;title&gt;表格示例&lt;/title&gt;
            &lt;/head&gt;
            &lt;body&gt;
                &lt;h1&gt;表格示例&lt;/h1&gt;
...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (934 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">
    &lt;!DOCTYPE html&gt;
            &lt;html lang=&quot;zh&quot;&gt;
            &lt;head&gt;
                &lt;meta charset=&quot;UTF-8&quot;&gt;
                &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;&gt;
                &lt;title&gt;表格示例&lt;/title&gt;
            &lt;/head&gt;
            &lt;body&gt;
                &lt;h1&gt;表格示例&lt;/h1&gt;
                &lt;table border=&quot;1&quot;&gt;
                    &lt;thead&gt;
                        &lt;tr&gt;
                            &lt;th&gt;姓名&lt;/th&gt;
                            &lt;th&gt;年龄&lt;/th&gt;
                            &lt;th&gt;城市&lt;/th&gt;
                        &lt;/tr&gt;
                    &lt;/thead&gt;
                    &lt;tbody&gt;
                        &lt;tr&gt;
                            &lt;td&gt;张三&lt;/td&gt;
                            &lt;td&gt;25&lt;/td&gt;
                            &lt;td&gt;北京&lt;/td&gt;
                        &lt;/tr&gt;
                        &lt;tr&gt;
                            &lt;td&gt;李四&lt;/td&gt;
                            &lt;td&gt;30&lt;/td&gt;
                            &lt;td&gt;上海&lt;/td&gt;
                        &lt;/tr&gt;
                        &lt;tr&gt;
                            &lt;td&gt;王五&lt;/td&gt;
                            &lt;td&gt;28&lt;/td&gt;
                            &lt;td&gt;广州&lt;/td&gt;
                        &lt;/tr&gt;
                    &lt;/tbody&gt;
                &lt;/table&gt;
            &lt;/body&gt;
            &lt;/html&gt;
    </pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">
    &lt;!DOCTYPE html&gt;
            &lt;html lang=&quot;zh&quot;&gt;
            &lt;head&gt;
                &lt;meta charset=&quot;UTF-8&quot;&gt;
                &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;&gt;
                &lt;title&gt;表格示例&lt;/title&gt;
            &lt;/head&gt;
            &lt;body&gt;
                &lt;h1&gt;表格示例&lt;/h1&gt;
...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (934 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">
    &lt;!DOCTYPE html&gt;
            &lt;html lang=&quot;zh&quot;&gt;
            &lt;head&gt;
                &lt;meta charset=&quot;UTF-8&quot;&gt;
                &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;&gt;
                &lt;title&gt;表格示例&lt;/title&gt;
            &lt;/head&gt;
            &lt;body&gt;
                &lt;h1&gt;表格示例&lt;/h1&gt;
                &lt;table border=&quot;1&quot;&gt;
                    &lt;thead&gt;
                        &lt;tr&gt;
                            &lt;th&gt;姓名&lt;/th&gt;
                            &lt;th&gt;年龄&lt;/th&gt;
                            &lt;th&gt;城市&lt;/th&gt;
                        &lt;/tr&gt;
                    &lt;/thead&gt;
                    &lt;tbody&gt;
                        &lt;tr&gt;
                            &lt;td&gt;张三&lt;/td&gt;
                            &lt;td&gt;25&lt;/td&gt;
                            &lt;td&gt;北京&lt;/td&gt;
                        &lt;/tr&gt;
                        &lt;tr&gt;
                            &lt;td&gt;李四&lt;/td&gt;
                            &lt;td&gt;30&lt;/td&gt;
                            &lt;td&gt;上海&lt;/td&gt;
                        &lt;/tr&gt;
                        &lt;tr&gt;
                            &lt;td&gt;王五&lt;/td&gt;
                            &lt;td&gt;28&lt;/td&gt;
                            &lt;td&gt;广州&lt;/td&gt;
                        &lt;/tr&gt;
                    &lt;/tbody&gt;
                &lt;/table&gt;
            &lt;/body&gt;
            &lt;/html&gt;
    </pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>html_tables</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[[[&#x27;姓名&#x27;, &#x27;年龄&#x27;, &#x27;城市&#x27;], [&#x27;张三&#x27;, &#x27;25&#x27;, &#x27;北京&#x27;], [&#x27;李四&#x27;, &#x27;30&#x27;, &#x27;上海&#x27;], [&#x27;王五&#x27;, &#x27;28&#x27;, &#x27;广州&#x27;]]]</td></tr></table></div></div>

#### ✨ explanation 解释
This example shows how the operator extracts tables from HTML content, including the table headers. The input is a simple HTML string containing a table with headers and rows. The operator processes this input and extracts the table, storing it in the 'html_tables' field of the metadata. The output includes the original text and the extracted table, which retains the header information.
这个例子展示了算子如何从HTML内容中提取表格，包括表头。输入是一个包含带有表头和行的表格的简单HTML字符串。算子处理这个输入并提取表格，将其存储在元数据的'html_tables'字段中。输出包括原始文本和提取的表格，保留了表头信息。

### test_no_tables
```python
ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;html&gt;&lt;body&gt;New testCase - No tables here!&lt;/body&gt;&lt;/html&gt;</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;html&gt;&lt;body&gt;New testCase - No tables here!&lt;/body&gt;&lt;/html&gt;</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>html_tables</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[]</td></tr></table></div></div>

#### ✨ explanation 解释
In this example, the input is an HTML document that does not contain any tables. The operator will process this input and, since there are no tables to extract, it stores an empty list in the 'html_tables' field of the metadata. The output data remains the same as the input data, with the addition of the empty 'html_tables' list in the metadata, indicating that no tables were found.
在这个例子中，输入是一个不包含任何表格的HTML文档。算子会处理这个输入，由于没有表格可以提取，它会在元数据的'html_tables'字段中存储一个空列表。输出数据与输入数据相同，在元数据中添加了一个空的'html_tables'列表，表明没有找到表格。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_tables_from_html_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_tables_from_html_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)