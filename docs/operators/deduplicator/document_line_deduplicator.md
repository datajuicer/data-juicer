# document_line_deduplicator

Deduplicates at the line level across documents.

This operator identifies lines that appear in many documents (boilerplate
text, copyright notices, navigation bars, etc.) and removes them.  It works
in two phases:

1. **compute_hash** – splits each document into lines, applies configurable
   skip rules, and computes an MD5 hash for every non-skipped line.
2. **process** – counts in how many *distinct* documents each line hash
   appears.  Lines whose document frequency exceeds
   ``frequency_threshold`` are removed from every document.

在文档间进行行级去重。

该算子识别出现在多个文档中的行（样板文本、版权声明、导航栏等）并将其移除。它分为两个阶段工作：

1. **compute_hash** – 将每个文档拆分为行，应用可配置的跳过规则，并为每个未跳过的行计算 MD5 哈希值。
2. **process** – 统计每个行哈希值出现在多少个*不同*文档中。文档频率超过 ``frequency_threshold`` 的行将从所有文档中移除。

Type 算子类型: **deduplicator**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `frequency_threshold` | <class 'int'> | `6` | document-frequency threshold.  Lines appearing in **more than** this many documents are removed. |
| `lowercase` | <class 'bool'> | `False` | whether to lower-case a line before hashing. |
| `ignore_special_character` | <class 'bool'> | `False` | whether to strip whitespace, digits, and punctuation before hashing. |
| `min_line_length` | <class 'int'> | `2` | lines whose stripped length is below this value are skipped (never considered for dedup). |
| `skip_brackets` | <class 'bool'> | `True` | skip lines consisting solely of bracket / semicolon characters such as ``{ } [ ] ( ) ;``. |
| `skip_markdown_headers` | <class 'bool'> | `True` | skip lines that start with ``#`` (Markdown headings). |
| `skip_latex_env` | <class 'bool'> | `True` | skip LaTeX ``\begin{…}`` / ``\end{…}`` environment declarations. |
| `skip_html_tags` | <class 'bool'> | `True` | skip lines that are pure HTML / XML tags. |
| `args` |  | `` | extra args |
| `kwargs` |  | `` | extra args |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/document_line_deduplicator.py)
- [unit test 单元测试](../../../tests/ops/deduplicator/test_document_line_deduplicator.py)
- [Return operator list 返回算子列表](../../Operators.md)