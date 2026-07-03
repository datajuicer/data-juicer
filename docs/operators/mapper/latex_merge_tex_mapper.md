# latex_merge_tex_mapper

Extracts and concatenates all `.tex` files from a compressed LaTeX project archive into a single text field.

Supported archive formats: `.tar`, `.tar.gz` / `.tgz`, and `.zip`. Plain `.gz` (single-file gzip) is not supported because gzip archives carry no filename metadata, making it impossible to verify that the content is actually a `.tex` file. All `.tex` files found inside the archive are read in-memory and joined with a configurable separator. No ordering or deduplication is applied.

This operator is typically placed before LaTeX-processing operators such as `remove_comments_mapper`, `expand_macro_mapper`, or `latex_figure_context_extractor_mapper`.

从压缩的 LaTeX 项目归档文件中提取并拼接所有 `.tex` 文件到一个文本字段中。

支持的归档格式：`.tar`、`.tar.gz` / `.tgz` 以及 `.zip`。不支持单独的 `.gz`（单文件 gzip），因为 gzip 格式不包含文件名元数据，无法验证内容是否为 `.tex` 文件。归档中所有 `.tex` 文件会被读入内存，并使用可配置的分隔符拼接。不会进行排序或去重。

该算子通常放置在 LaTeX 处理算子（如 `remove_comments_mapper`、`expand_macro_mapper` 或 `latex_figure_context_extractor_mapper`）之前。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `compressed_file_key` | `str` | `'compressed_file'` | Field name that stores the archive file path. 存储归档文件路径的字段名。 |
| `separator` | `str` | `'\n\n'` | String used to join the contents of multiple `.tex` files. 用于拼接多个 `.tex` 文件内容的分隔符。 |
| `max_file_size` | `int` | `52428800` (50 MB) | Maximum allowed uncompressed size in bytes for a single `.tex` entry inside the archive. Entries exceeding this limit are skipped with a warning. Set to `0` to disable the check. 单个 `.tex` 条目允许的最大解压大小（字节）。超过此限制的条目将被跳过并输出警告。设为 `0` 可禁用检查。 |

## 🔗 Related links 相关链接

- [source code 源代码](../../../data_juicer/ops/mapper/latex_merge_tex_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_latex_merge_tex_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
