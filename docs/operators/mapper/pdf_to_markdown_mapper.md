# pdf_to_markdown_mapper

Mapper to convert PDF (path or bytes) to high-precision structured Markdown.

Reads PDF from a field (file path or bytes), converts to a single structured Markdown string. Use with `download_file_mapper` to download PDFs first. Images are dropped by default for chunking or LLM use.

将 PDF（路径或字节）转换为高精度结构化的 Markdown 单文件。可与 download_file_mapper 配合使用。图片默认丢弃。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `pdf_key` | str | `"pdf"` | Field containing PDF path (str) or bytes. 存 PDF 路径或字节的字段名。 |
| `backend` | str | `"mineru"` | `mineru` (推荐，MinerU 高精度) 或 `pdfplumber` (内置回退). 转换后端。 |
| `keep_images` | bool | `False` | Keep image refs in Markdown (mineru only). 是否保留图片引用。 |
| `output_key` | str | `text_key` | Field to write Markdown to. 写入 Markdown 的字段名。 |

## 📦 Dependencies 依赖

- Built-in: `pdfplumber`
- Optional: `magic-pdf`. Install: `pip install py-data-juicer[document]`

## 📋 YAML Example 配置示例

```yaml
- pdf_to_markdown_mapper:
    pdf_key: pdf
    backend: mineru   # recommended; use pdfplumber if magic-pdf not installed
    output_key: text
```

## 🔗 Related links 相关链接

- [source code 源代码](../../../data_juicer/ops/mapper/pdf_to_markdown_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_pdf_to_markdown_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
