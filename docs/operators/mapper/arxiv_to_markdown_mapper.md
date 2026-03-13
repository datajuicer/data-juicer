# arxiv_to_markdown_mapper

Mapper to convert arXiv paper ID to high-precision structured Markdown in a single document.

Given an arXiv paper ID (e.g. `2501.14755` or `hep-th/9901001`), this operator downloads the PDF or crawls the page and converts it to a single structured Markdown string. **MinerU** (default) gives the best conversion quality; **crawl4ai** first tries the **HTML full-text page** (e.g. `https://arxiv.org/html/2501.14755v1`) when available, then the abstract page, then falls back to PDF. Images are dropped by default for smaller output and easier chunking or LLM consumption.

根据 arXiv 论文 ID 将论文转换为高精度结构化的 Markdown 单文件。默认 **MinerU** 转换质量最佳；**crawl4ai** 会优先抓取 **HTML 全文页**（如 `https://arxiv.org/html/2501.14755v1`），若无则抓摘要页，再失败则回退到 PDF。图片默认丢弃。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `arxiv_id_key` | str | `"arxiv_id"` | Field name for arXiv ID or URL. 存 arXiv ID 或 URL 的字段名。 |
| `backend` | str | `"mineru"` | `mineru` (推荐，MinerU 高精度), `pdfplumber` (内置回退，仅文本), `crawl4ai` (优先 HTML 全文页，再摘要页，最后 PDF). 转换后端。 |
| `add_metadata_header` | bool | `True` | Prepend title, authors, abstract from arXiv API. 是否从 API 拉取并写入元数据头部。 |
| `keep_images` | bool | `False` | Keep image refs in Markdown (mineru only). 是否保留图片引用。 |
| `timeout` | int | `60` | Request timeout (seconds). 请求超时（秒）。 |
| `download_delay` | float | `1.0` | Delay before each download (rate limiting). 每次下载前延迟（秒）。 |
| `output_key` | str | `text_key` | Field to write Markdown to. 写入 Markdown 的字段名。 |

## 📦 Dependencies 依赖

- Built-in: `pdfplumber`, `requests`
- Optional: `magic-pdf` (MinerU), `crawl4ai`. Install: `pip install py-data-juicer[document]`

## 📋 YAML Example 配置示例

```yaml
- arxiv_to_markdown_mapper:
    arxiv_id_key: arxiv_id
    backend: mineru   # recommended; use pdfplumber if magic-pdf not installed
    add_metadata_header: true
    output_key: text
```

## 🔗 Related links 相关链接

- [source code 源代码](../../../data_juicer/ops/mapper/arxiv_to_markdown_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_arxiv_to_markdown_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
