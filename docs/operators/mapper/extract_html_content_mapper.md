# extract_html_content_mapper

Extracts the main content and metadata of web pages from raw HTML using Trafilatura.

Unlike `clean_html_mapper`, which removes HTML tags but keeps all visible text, this operator separates the main content from boilerplate such as navigation bars, sidebars, cookie notices and footers. It can produce plain text or Markdown, optionally keep tables, links and comments, and store page metadata. If no main content is found, the text becomes an empty string; a downstream length filter can then remove the sample. This operator processes HTML already present in the dataset; it does not download the URL in `url_key`.

使用 Trafilatura 从原始 HTML 中提取网页正文和元数据。

与删除 HTML 标签但保留全部可见文本的 `clean_html_mapper` 不同，该算子会从导航栏、侧边栏、Cookie 提示和页脚等网页模板内容中分离正文。它可以输出纯文本或 Markdown，可选择在正文中保留表格、链接和评论，并可保存页面元数据。若未找到正文，文本会变为空字符串，之后可以用长度过滤算子删除该样本。该算子只处理数据集中已有的 HTML，不会下载 `url_key` 中的 URL。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `output_format` | str | `'txt'` | Output format: `'txt'` or `'markdown'`. 输出格式：`'txt'` 或 `'markdown'`。 |
| `focus` | str | `'balanced'` | Extraction trade-off: `'precision'` reduces noise, `'recall'` reduces missing content, and `'balanced'` sits in between. 提取取舍：`'precision'` 减少噪声，`'recall'` 减少正文遗漏，`'balanced'` 在两者之间。 |
| `include_tables` | bool | `True` | Whether to keep tables in the output text. 是否在输出正文中保留表格。 |
| `include_links` | bool | `False` | Whether to keep link targets; only effective for Markdown output. 是否保留链接地址；仅对 Markdown 输出有效。 |
| `include_comments` | bool | `False` | Whether to keep user comments in addition to the main content. 是否在正文之外保留用户评论。 |
| `fast` | bool | `False` | Skip fallback extraction algorithms for speed. 为提高速度而跳过备用提取算法。 |
| `url_key` | str or null | `None` | Optional field containing the page URL, used to resolve relative links and populate URL metadata. 可选的页面 URL 字段，用于解析相对链接并填充 URL 元数据。 |
| `extract_metadata` | bool | `False` | Whether to extract page metadata. 是否提取页面元数据。 |
| `metadata_field_name` | str | `'html_metadata'` | Field in `meta` used to store metadata. `meta` 中用于保存元数据的字段名。 |

When metadata extraction is enabled, the operator stores `title`, `author`, `date`, `url`, `hostname`, `sitename`, `description`, `categories` and `tags` under `meta[metadata_field_name]`. To store HTML tables as separate structured data, use `extract_tables_from_html_mapper` before this operator rewrites the HTML text.

启用元数据提取后，算子会在 `meta[metadata_field_name]` 中保存 `title`、`author`、`date`、`url`、`hostname`、`sitename`、`description`、`categories` 和 `tags`。如果需要将 HTML 表格单独保存为结构化数据，请在本算子改写 HTML 文本之前使用 `extract_tables_from_html_mapper`。

## 📊 Effect demonstration 效果演示
### test_markdown_output
```python
ExtractHtmlContentMapper(output_format='markdown')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;!DOCTYPE html&gt;
&lt;html lang=&quot;en&quot;&gt;
&lt;head&gt;
  &lt;meta charset=&quot;utf-8&quot;&gt;
  &lt;title&gt;Why Rivers Meander | Example Science&lt;/title&gt;
  &lt;meta name=&quot;author&quot; content=&quot;Jane Doe&quot;&gt;
  &lt;meta name=&quot;description&quot; content=&quot;A short explanation of river meanders.&quot;&gt;
  &lt;meta property=&quot;og:site_name&quot; content=&quot;Example Science&quot;&gt;
  &lt;...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (1657 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;!DOCTYPE html&gt;
&lt;html lang=&quot;en&quot;&gt;
&lt;head&gt;
  &lt;meta charset=&quot;utf-8&quot;&gt;
  &lt;title&gt;Why Rivers Meander | Example Science&lt;/title&gt;
  &lt;meta name=&quot;author&quot; content=&quot;Jane Doe&quot;&gt;
  &lt;meta name=&quot;description&quot; content=&quot;A short explanation of river meanders.&quot;&gt;
  &lt;meta property=&quot;og:site_name&quot; content=&quot;Example Science&quot;&gt;
  &lt;meta property=&quot;article:published_time&quot; content=&quot;2024-05-17&quot;&gt;
&lt;/head&gt;
&lt;body&gt;
  &lt;div class=&quot;cookie-banner&quot;&gt;We use cookies to improve your experience. Accept all cookies&lt;/div&gt;
  &lt;header&gt;&lt;nav&gt;&lt;ul&gt;&lt;li&gt;&lt;a href=&quot;/&quot;&gt;Home&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;/news&quot;&gt;News&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/nav&gt;&lt;/header&gt;
  &lt;aside class=&quot;sidebar&quot;&gt;&lt;h3&gt;Trending now&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;&lt;a href=&quot;/x&quot;&gt;Ten celebrity diets you must try&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/aside&gt;
  &lt;main&gt;
    &lt;article&gt;
      &lt;h1&gt;Why Rivers Meander&lt;/h1&gt;
      &lt;p&gt;Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.&lt;/p&gt;
      &lt;p&gt;On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of erosion and deposition makes the bend grow wider year after year.&lt;/p&gt;
      &lt;h2&gt;Key factors&lt;/h2&gt;
      &lt;ul&gt;
        &lt;li&gt;The slope of the valley floor&lt;/li&gt;
        &lt;li&gt;The amount of sediment carried by the river&lt;/li&gt;
      &lt;/ul&gt;
      &lt;table&gt;
        &lt;tr&gt;&lt;th&gt;River&lt;/th&gt;&lt;th&gt;Sinuosity&lt;/th&gt;&lt;/tr&gt;
        &lt;tr&gt;&lt;td&gt;Mississippi&lt;/td&gt;&lt;td&gt;1.9&lt;/td&gt;&lt;/tr&gt;
      &lt;/table&gt;
      &lt;p&gt;Eventually a meander can be cut off entirely, leaving behind a crescent-shaped oxbow lake that slowly fills with sediment and &lt;a href=&quot;/plants&quot;&gt;vegetation&lt;/a&gt;.&lt;/p&gt;
    &lt;/article&gt;
    &lt;section id=&quot;comments&quot; class=&quot;comments&quot;&gt;
      &lt;h3&gt;Comments&lt;/h3&gt;
      &lt;div class=&quot;comment&quot;&gt;&lt;p&gt;Great article, I finally understand oxbow lakes after visiting one last summer near my hometown.&lt;/p&gt;&lt;/div&gt;
    &lt;/section&gt;
  &lt;/main&gt;
  &lt;footer&gt;&lt;p&gt;Copyright 2024 Example Science. All rights reserved.&lt;/p&gt;&lt;a href=&quot;/privacy&quot;&gt;Privacy policy&lt;/a&gt;&lt;/footer&gt;
&lt;/body&gt;
&lt;/html&gt;</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"># Why Rivers Meander

Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.

On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of ero...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (351 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"># Why Rivers Meander

Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.

On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of erosion and deposition makes the bend grow wider year after year.

## Key factors

- The slope of the valley floor
- The amount of sediment carried by the river

| River | Sinuosity |
|---|---|
| Mississippi | 1.9 |

Eventually a meander can be cut off entirely, leaving behind a crescent-shaped oxbow lake that slowly fills with sediment and vegetation.</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>__dj__meta__</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>{}</td></tr></table></div></div>

#### ✨ explanation 解释
This example uses the HTML fixture from the unit test. The input contains a cookie banner, navigation, a sidebar, an article, a comments section and a footer. The operator removes the boilerplate and comments while preserving the article heading hierarchy, list and table as Markdown.
此示例使用单元测试中的 HTML 样本。输入包含 Cookie 提示、导航栏、侧边栏、文章、评论区和页脚。算子会删除模板噪声和评论，同时以 Markdown 保留文章标题层级、列表和表格。

### test_extract_metadata
```python
ExtractHtmlContentMapper(extract_metadata=True, url_key='url')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;!DOCTYPE html&gt;
&lt;html lang=&quot;en&quot;&gt;
&lt;head&gt;
  &lt;meta charset=&quot;utf-8&quot;&gt;
  &lt;title&gt;Why Rivers Meander | Example Science&lt;/title&gt;
  &lt;meta name=&quot;author&quot; content=&quot;Jane Doe&quot;&gt;
  &lt;meta name=&quot;description&quot; content=&quot;A short explanation of river meanders.&quot;&gt;
  &lt;meta property=&quot;og:site_name&quot; content=&quot;Example Science&quot;&gt;
  &lt;...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (1657 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;!DOCTYPE html&gt;
&lt;html lang=&quot;en&quot;&gt;
&lt;head&gt;
  &lt;meta charset=&quot;utf-8&quot;&gt;
  &lt;title&gt;Why Rivers Meander | Example Science&lt;/title&gt;
  &lt;meta name=&quot;author&quot; content=&quot;Jane Doe&quot;&gt;
  &lt;meta name=&quot;description&quot; content=&quot;A short explanation of river meanders.&quot;&gt;
  &lt;meta property=&quot;og:site_name&quot; content=&quot;Example Science&quot;&gt;
  &lt;meta property=&quot;article:published_time&quot; content=&quot;2024-05-17&quot;&gt;
&lt;/head&gt;
&lt;body&gt;
  &lt;div class=&quot;cookie-banner&quot;&gt;We use cookies to improve your experience. Accept all cookies&lt;/div&gt;
  &lt;header&gt;&lt;nav&gt;&lt;ul&gt;&lt;li&gt;&lt;a href=&quot;/&quot;&gt;Home&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;/news&quot;&gt;News&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/nav&gt;&lt;/header&gt;
  &lt;aside class=&quot;sidebar&quot;&gt;&lt;h3&gt;Trending now&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;&lt;a href=&quot;/x&quot;&gt;Ten celebrity diets you must try&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/aside&gt;
  &lt;main&gt;
    &lt;article&gt;
      &lt;h1&gt;Why Rivers Meander&lt;/h1&gt;
      &lt;p&gt;Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.&lt;/p&gt;
      &lt;p&gt;On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of erosion and deposition makes the bend grow wider year after year.&lt;/p&gt;
      &lt;h2&gt;Key factors&lt;/h2&gt;
      &lt;ul&gt;
        &lt;li&gt;The slope of the valley floor&lt;/li&gt;
        &lt;li&gt;The amount of sediment carried by the river&lt;/li&gt;
      &lt;/ul&gt;
      &lt;table&gt;
        &lt;tr&gt;&lt;th&gt;River&lt;/th&gt;&lt;th&gt;Sinuosity&lt;/th&gt;&lt;/tr&gt;
        &lt;tr&gt;&lt;td&gt;Mississippi&lt;/td&gt;&lt;td&gt;1.9&lt;/td&gt;&lt;/tr&gt;
      &lt;/table&gt;
      &lt;p&gt;Eventually a meander can be cut off entirely, leaving behind a crescent-shaped oxbow lake that slowly fills with sediment and &lt;a href=&quot;/plants&quot;&gt;vegetation&lt;/a&gt;.&lt;/p&gt;
    &lt;/article&gt;
    &lt;section id=&quot;comments&quot; class=&quot;comments&quot;&gt;
      &lt;h3&gt;Comments&lt;/h3&gt;
      &lt;div class=&quot;comment&quot;&gt;&lt;p&gt;Great article, I finally understand oxbow lakes after visiting one last summer near my hometown.&lt;/p&gt;&lt;/div&gt;
    &lt;/section&gt;
  &lt;/main&gt;
  &lt;footer&gt;&lt;p&gt;Copyright 2024 Example Science. All rights reserved.&lt;/p&gt;&lt;a href=&quot;/privacy&quot;&gt;Privacy policy&lt;/a&gt;&lt;/footer&gt;
&lt;/body&gt;
&lt;/html&gt;</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>url</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>https://example.org/science/rivers</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Why Rivers Meander
Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.
On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of erosion...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (340 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Why Rivers Meander
Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.
On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of erosion and deposition makes the bend grow wider year after year.
Key factors
- The slope of the valley floor
- The amount of sediment carried by the river
| River | Sinuosity |
|---|---|
| Mississippi | 1.9 |
Eventually a meander can be cut off entirely, leaving behind a crescent-shaped oxbow lake that slowly fills with sediment and vegetation.</pre></details><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>url</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>https://example.org/science/rivers</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>__dj__meta__</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>{&quot;html_metadata&quot;:{&quot;title&quot;:&quot;Why Rivers Meander&quot;,&quot;author&quot;:&quot;Jane Doe&quot;,&quot;date&quot;:&quot;2024-05-17&quot;,&quot;url&quot;:&quot;https:\/\/example.org\/science\/rivers&quot;,&quot;hostname&quot;:&quot;example.org&quot;,&quot;sitename&quot;:&quot;Example Science&quot;,&quot;description&quot;:&quot;A short explanation of river meanders.&quot;,&quot;categories&quot;:[],&quot;tags&quot;:[]}}</td></tr></table></div></div>

#### ✨ explanation 解释
This example enables metadata extraction and supplies the page URL through the `url` field. The cleaned article replaces the HTML in `text`, while the title, author, publication date, URL, host, site name and description are stored under `__dj__meta__.html_metadata`.
此示例启用元数据提取，并通过 `url` 字段提供页面地址。清洗后的文章会替换 `text` 中的 HTML；标题、作者、发布日期、URL、主机名、站点名和描述会存入 `__dj__meta__.html_metadata`。

## 🔗 related links 相关链接

- [Trafilatura](https://github.com/adbar/trafilatura)
- [source code 源代码](../../../data_juicer/ops/mapper/extract_html_content_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_html_content_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
