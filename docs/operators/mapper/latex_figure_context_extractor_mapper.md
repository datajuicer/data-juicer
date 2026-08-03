# latex_figure_context_extractor_mapper

Extracts figures and their citing context from LaTeX source.

This operator parses figure environments from a paper's LaTeX source, extracts each figure's caption, label, and image path(s), and finds the prose paragraphs that cite each figure. It fans out one paper row into N figure rows (one per figure or subfigure). **Samples that contain no figures with images are dropped from the output.** Supported figure environments: `figure`, `figure*`, `wrapfigure`, `subfigure` (environment), `\subfigure` (command), `\subfloat` (command, subfig package). Supported caption commands: `\caption`, `\caption*`, `\subcaption`, `\captionof{figure}`. Figures without `\includegraphics` are skipped. Subfigures inherit citing paragraphs from their parent figure's label. When building citing paragraphs, float/display environments (figures, tables, tabulars, equations, algorithms, etc.) are stripped so only prose text is searched.

> **Note:** This operator expects the full LaTeX source as a single string. It does **not** resolve `\input` or `\include` directives. If your documents span multiple `.tex` files, concatenate them into a single text field before applying this mapper.

从LaTeX源码中提取图片及其引用上下文。

该算子解析论文LaTeX源码中的figure环境，提取每个图片的标题、标签和图片路径，并找到引用该图片的段落文本。它将一行论文数据展开为N行图片数据（每个图片或子图一行）。**不包含带图片的figure环境的样本将被丢弃。** 支持的图片环境：`figure`、`figure*`、`wrapfigure`、`subfigure`（环境）、`\subfigure`（命令）、`\subfloat`（命令，subfig宏包）。支持的标题命令：`\caption`、`\caption*`、`\subcaption`、`\captionof{figure}`。没有`\includegraphics`的图片会被跳过。子图会继承父图标签的引用段落。构建引用段落时，浮动/展示环境（图片、表格、公式、算法等）会被去除，仅在正文文本中搜索。

> **注意：** 该算子要求完整的LaTeX源码作为单个字符串输入。它**不会**解析`\input`或`\include`指令。如果您的文档分散在多个`.tex`文件中，请在使用此算子之前将它们合并到一个文本字段中。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `citation_commands` | `list` | `['\ref', '\cref', '\Cref', '\autoref']` | LaTeX reference commands to search for when finding citing paragraphs. |
| `paragraph_separator` | `str` | `'\n\n'` | Pattern for splitting LaTeX text into paragraphs. |
| `caption_key` | `str` | `'caption'` | Output field name for the figure caption. |
| `label_key` | `str` | `'label'` | Output field name for the LaTeX label. |
| `context_key` | `str` | `'citing_paragraphs'` | Output field name for citing paragraphs. |
| `parent_caption_key` | `str` | `'parent_caption'` | Output field name for the parent figure's caption. For subfigures this carries the parent figure environment's caption; empty for standalone figures. |
| `parent_label_key` | `str` | `'parent_label'` | Output field name for the parent figure's label. Useful for grouping subfigures that belong to the same figure environment; empty for standalone figures. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📤 Output Fields 输出字段

In addition to all input fields, each output row contains:

除所有输入字段外，每行输出还包含：

| field 字段 | type 类型 | desc 说明 |
|-------|------|------|
| `images` (or custom `image_key`) | `list[str]` | Image paths from `\includegraphics`. 从`\includegraphics`提取的图片路径。 |
| `caption` (or custom `caption_key`) | `str` | Figure caption text. 图片标题文本。 |
| `label` (or custom `label_key`) | `str` | LaTeX label string. LaTeX标签字符串。 |
| `citing_paragraphs` (or custom `context_key`) | `list[str]` | Paragraphs that cite this figure. 引用该图片的段落。 |
| `parent_caption` (or custom `parent_caption_key`) | `str` | Parent figure caption (subfigures only; empty for standalone). 父图标题（仅子图；独立图为空）。 |
| `parent_label` (or custom `parent_label_key`) | `str` | Parent figure label (subfigures only; empty for standalone). 父图标签（仅子图；独立图为空）。 |

## 📊 Effect demonstration 效果演示
### test_single_figure
```python
LatexFigureContextExtractorMapper()
```

#### 📥 input data 输入数据

A LaTeX document with a single figure:

```latex
\begin{document}
Some intro text.

As shown in \ref{fig:arch}, the architecture is novel.

\begin{figure}
\centering
\includegraphics[width=0.8\linewidth]{img/arch.pdf}
\caption{Overall architecture}
\label{fig:arch}
\end{figure}
\end{document}
```

#### 📤 output data 输出数据

One row is produced:
- `caption`: `"Overall architecture"`
- `label`: `"fig:arch"`
- `images`: `["img/arch.pdf"]`
- `citing_paragraphs`: `["As shown in \\ref{fig:arch}, the architecture is novel."]`
- `parent_caption`: `""` (standalone figure, no parent)
- `parent_label`: `""` (standalone figure, no parent)

#### ✨ explanation 解释
The operator extracts the figure environment, parses its caption, label, and image path, then searches the document paragraphs (with float/display/tabular environments stripped) for any paragraph containing `\ref{fig:arch}`. The matching paragraph is returned as the citing context.

算子提取figure环境，解析其标题、标签和图片路径，然后在文档段落中（去除浮动/展示/表格环境后）搜索包含`\ref{fig:arch}`的段落。匹配的段落作为引用上下文返回。

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/latex_figure_context_extractor_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_latex_figure_context_extractor_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
