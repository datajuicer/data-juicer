# remove_comments_mapper

Removes comments from documents, currently supporting only 'tex' format.

This operator removes inline and multiline comments from text samples. It supports both inline and multiline comment removal, controlled by the `inline` and `multiline` parameters. Currently, it is designed to work with 'tex' documents. The operator processes each sample in the batch and applies regular expressions to remove comments. The processed text is then updated in the original samples.

- Inline comments are removed using the pattern `[^\]%.+$`.
- Multiline comments are removed using the pattern `^%.* ?`.

Important notes:
- Only 'tex' document type is supported at present.
- The operator processes the text in place and updates the original samples.

从文档中移除注释，目前仅支持'tex'格式。

该算子从文本样本中移除内联和多行注释。它支持内联和多行注释的移除，由`inline`和`multiline`参数控制。目前，它被设计用于处理'tex'文档。该算子处理批次中的每个样本，并应用正则表达式来移除注释。处理后的文本会更新到原始样本中。

- 内联注释使用模式 `[^\]%.+$` 进行移除。
- 多行注释使用模式 `^%.* ?` 进行移除。

重要说明：
- 目前仅支持'tex'文档类型。
- 该算子就地处理文本并更新原始样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `doc_type` | typing.Union[str, typing.List[str]] | `'tex'` | Type of document to remove comments. |
| `inline` | <class 'bool'> | `True` | Whether to remove inline comments. |
| `multiline` | <class 'bool'> | `True` | Whether to remove multiline comments. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_tex_case
```python
RemoveCommentsMapper(doc_type='tex', inline=True, multiline=True)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&#x27;%%\n%% This is file `sample-sigconf.tex\&#x27;,\n%% The first command in your LaTeX source must be the \\documentclass command.\n\\documentclass[sigconf,review,anonymous]{acmart}\n%% NOTE that a single column version is required for \n%% submission and peer review. This can be done by changing\n\\input{math_commands.tex}\n%% end of the preamble, start of the body of the document source.\n\\begin{document}\n%% The &quot;title&quot; command has an optional parameter,\n\\title{Hierarchical Cross Contrastive Learning of Visual Representations}\n%%\n%% The &quot;author&quot; command and its associated commands are used t...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (2108 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&#x27;%%\n%% This is file `sample-sigconf.tex\&#x27;,\n%% The first command in your LaTeX source must be the \\documentclass command.\n\\documentclass[sigconf,review,anonymous]{acmart}\n%% NOTE that a single column version is required for \n%% submission and peer review. This can be done by changing\n\\input{math_commands.tex}\n%% end of the preamble, start of the body of the document source.\n\\begin{document}\n%% The &quot;title&quot; command has an optional parameter,\n\\title{Hierarchical Cross Contrastive Learning of Visual Representations}\n%%\n%% The &quot;author&quot; command and its associated commands are used to define\n%% the authors and their affiliations.\n\\author{Hesen Chen}\n\\affiliation{%\n  \\institution{Alibaba Group}\n  \\city{Beijing}\n  \\country{China}}\n\\email{hesen.chs@alibaba-inc.com}\n%% By default, the full list of authors will be used in the page\n\\begin{abstract}The rapid\n\\end{abstract}\n\\begin{CCSXML}\n\\ccsdesc[500]{Computing methodologies~Image representations}\n%% Keywords. The author(s) should pick words that accurately describe\n\\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}\n%% page.\n\\begin{teaserfigure}\n\\end{teaserfigure}\n%% This command processes the author and affiliation and title\n\\maketitle\n\\section{Introduction}\n\\begin{itemize}\n\\end{itemize}\n\\section{Related Work}\n\\label{gen_inst} Self-supervised\n\\section{Method}\n\\label{method}In this section,\n\\subsection{Framework} kkk\n\\subsection{Cross Contrastive Loss}\nSince $\\sZ^n$ are extracted\n\\subsection{Implementation details}\n\\textbf{Image augmentations} We use\n\\textbf{Architecture} We use\n\\textbf{Optimization} We adapt \n\\section{Experiments}\n\\label{experiments}In this section\n\\subsection{Linear and Semi-Supervised Evaluations on ImageNet}\n\\textbf{Linear evaluation on ImageNet} We firs\n\\textbf{Semi-supervised learning on ImageNet} We simply\n\\subsection{Transfer to other datasets and tasks}\n\\textbf{Image classification with fixed features} We follow\n\\section{Ablations} We present\n\\subsection{Influence of hierarchical projection head and cross contrastive loss} get out\n\\subsection{Levels and depth of projector network}\n\\end{center}\n\\caption{\\label{figure3} \\textbf{Different way of cross-correlation on 3 level hierarchical projection head.} \&#x27;=\&#x27; denotes stop gradient.}\n\\end{figure}\n\\subsection{Analyze of} In this\n\\textbf{Similarity between} Using SimSiam\n\\textbf{Feature similarity} We extracted\n\\section{Conclusion}\nWe propose HCCL\n\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{sample-base}\n\\end{document}\n\\endinput\n%%\n%% End of file `sample-sigconf.tex\&#x27;.\n&#x27;]</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&quot;\\documentclass[sigconf,review,anonymous]{acmart}\n\\input{math_commands.tex}\n\\begin{document}\n\\title{Hierarchical Cross Contrastive Learning of Visual Representations}\n\\author{Hesen Chen}\n\\affiliation{%\n  \\institution{Alibaba Group}\n  \\city{Beijing}\n  \\country{China}}\n\\email{hesen.chs@alibaba-inc.com}\n\\begin{abstract}The rapid\n\\end{abstract}\n\\begin{CCSXML}\n\\ccsdesc[500]{Computing methodologies~Image representations}\n\\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}\n\\begin{teaserfigure}\n\\end{teaserfigure}\n\\maketitle\n\\sect...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (1378 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&quot;\\documentclass[sigconf,review,anonymous]{acmart}\n\\input{math_commands.tex}\n\\begin{document}\n\\title{Hierarchical Cross Contrastive Learning of Visual Representations}\n\\author{Hesen Chen}\n\\affiliation{%\n  \\institution{Alibaba Group}\n  \\city{Beijing}\n  \\country{China}}\n\\email{hesen.chs@alibaba-inc.com}\n\\begin{abstract}The rapid\n\\end{abstract}\n\\begin{CCSXML}\n\\ccsdesc[500]{Computing methodologies~Image representations}\n\\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}\n\\begin{teaserfigure}\n\\end{teaserfigure}\n\\maketitle\n\\section{Introduction}\n\\begin{itemize}\n\\end{itemize}\n\\section{Related Work}\n\\label{gen_inst} Self-supervised\n\\section{Method}\n\\label{method}In this section,\n\\subsection{Framework} kkk\n\\subsection{Cross Contrastive Loss}\nSince $\\sZ^n$ are extracted\n\\subsection{Implementation details}\n\\textbf{Image augmentations} We use\n\\textbf{Architecture} We use\n\\textbf{Optimization} We adapt \n\\section{Experiments}\n\\label{experiments}In this section\n\\subsection{Linear and Semi-Supervised Evaluations on ImageNet}\n\\textbf{Linear evaluation on ImageNet} We firs\n\\textbf{Semi-supervised learning on ImageNet} We simply\n\\subsection{Transfer to other datasets and tasks}\n\\textbf{Image classification with fixed features} We follow\n\\section{Ablations} We present\n\\subsection{Influence of hierarchical projection head and cross contrastive loss} get out\n\\subsection{Levels and depth of projector network}\n\\end{center}\n\\caption{\\label{figure3} \\textbf{Different way of cross-correlation on 3 level hierarchical projection head.} &#x27;=&#x27; denotes stop gradient.}\n\\end{figure}\n\\subsection{Analyze of} In this\n\\textbf{Similarity between} Using SimSiam\n\\textbf{Feature similarity} We extracted\n\\section{Conclusion}\nWe propose HCCL\n\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{sample-base}\n\\end{document}\n\\endinput\n&quot;]</pre></details></div>

#### ✨ explanation 解释
The operator removes both inline and multiline comments from a LaTeX document. It identifies and deletes any text that starts with '%%' for multiline comments and '%' for inline comments, keeping the rest of the content intact. The result is a cleaned version of the input, where all comments have been removed, leaving only the essential LaTeX code and text.
算子从LaTeX文档中删除单行和多行注释。它识别并删除以'%%'开头的多行注释和以'%'开头的单行注释，同时保留其余内容不变。结果是清理后的输入版本，其中所有注释都已被移除，仅留下必要的LaTeX代码和文本。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_comments_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_comments_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)