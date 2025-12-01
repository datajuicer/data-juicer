# remove_header_mapper

Removes headers at the beginning of documents in LaTeX samples.

This operator identifies and removes headers such as chapter, part, section, subsection, subsubsection, paragraph, and subparagraph. It uses a regular expression to match these headers. If a sample does not contain any headers and `drop_no_head` is set to True, the sample text will be removed. Otherwise, the sample remains unchanged. The operator processes samples in batches for efficiency.

移除LaTeX样本中文档开头的标题。

该算子识别并移除如章节、部分、节、小节、子小节、段落和子段落等标题。它使用正则表达式来匹配这些标题。如果一个样本不包含任何标题且`drop_no_head`设置为True，则该样本文本将被移除。否则，样本保持不变。该算子批量处理样本以提高效率。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `drop_no_head` | <class 'bool'> | `True` | whether to drop sample texts without headers. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_case
```python
RemoveHeaderMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex&#x27;,
%% The first command in your LaTeX source must be the \documentclass command.
\documentclass[sigconf,review,anonymous]{acmart}
%% NOTE that a single column version is required for 
%% submission and peer review. This can be done by changing
\input{math_comman...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (2272 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex&#x27;,
%% The first command in your LaTeX source must be the \documentclass command.
\documentclass[sigconf,review,anonymous]{acmart}
%% NOTE that a single column version is required for 
%% submission and peer review. This can be done by changing
\input{math_commands.tex}
%% end of the preamble, start of the body of the document source.
\begin{document}
%% The &quot;title&quot; command has an optional parameter,
\title{Hierarchical Cross Contrastive Learning of Visual Representations}
%%
%% The &quot;author&quot; command and its associated commands are used to define
%% the authors and their affiliations.
\author{Hesen Chen}
\affiliation{%
  \institution{Alibaba Group}
  \city{Beijing}
  \country{China}}
\email{hesen.chs@alibaba-inc.com}
%% By default, the full list of authors will be used in the page
\begin{abstract}The rapid
\end{abstract}
\begin{CCSXML}
\ccsdesc[500]{Computing methodologies~Image representations}
%% Keywords. The author(s) should pick words that accurately describe
\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}
%% page.
\begin{teaserfigure}
\end{teaserfigure}
%% This command processes the author and affiliation and title
\maketitle
\section{Introduction}
\begin{itemize}
\end{itemize}
\section{Related Work}
\label{gen_inst} Self-supervised
\section{Method}
\label{method}In this section,
\subsection{Framework} kkk
\subsection{Cross Contrastive Loss}
Since $\sZ^n$ are extracted
\subsection{Implementation details}
\textbf{Image augmentations} We use
\textbf{Architecture} We use
\textbf{Optimization} We adapt 
\section{Experiments}
\label{experiments}In this section
\subsection{Linear and Semi-Supervised Evaluations on ImageNet}
\textbf{Linear evaluation on ImageNet} We firs
\textbf{Semi-supervised learning on ImageNet} We simply
\subsection{Transfer to other datasets and tasks}
\textbf{Image classification with fixed features} We follow
\section{Ablations} We present
\subsection{Influence of hierarchical projection head and cross contrastive loss} get out
\subsection{Levels and depth of projector network}
\end{center}
\caption{\label{figure3} \textbf{Different way of cross-correlation on 3 level hierarchical projection head.} &#x27;=&#x27; denotes stop gradient.}
\end{figure}
\subsection{Analyze of} In this
\textbf{Similarity between} Using SimSiam
\textbf{Feature similarity} We extracted
\section{Conclusion}
We propose HCCL
\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">\section{Introduction}
\begin{itemize}
\end{itemize}
\section{Related Work}
\label{gen_inst} Self-supervised
\section{Method}
\label{method}In this section,
\subsection{Framework} kkk
\subsection{Cross Contrastive Loss}
Since $\sZ^n$ are extracted
\subsection{Implementation details}
\textbf{Image au...</pre><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show more 展开更多 (1047 more chars)</summary><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">\section{Introduction}
\begin{itemize}
\end{itemize}
\section{Related Work}
\label{gen_inst} Self-supervised
\section{Method}
\label{method}In this section,
\subsection{Framework} kkk
\subsection{Cross Contrastive Loss}
Since $\sZ^n$ are extracted
\subsection{Implementation details}
\textbf{Image augmentations} We use
\textbf{Architecture} We use
\textbf{Optimization} We adapt 
\section{Experiments}
\label{experiments}In this section
\subsection{Linear and Semi-Supervised Evaluations on ImageNet}
\textbf{Linear evaluation on ImageNet} We firs
\textbf{Semi-supervised learning on ImageNet} We simply
\subsection{Transfer to other datasets and tasks}
\textbf{Image classification with fixed features} We follow
\section{Ablations} We present
\subsection{Influence of hierarchical projection head and cross contrastive loss} get out
\subsection{Levels and depth of projector network}
\end{center}
\caption{\label{figure3} \textbf{Different way of cross-correlation on 3 level hierarchical projection head.} &#x27;=&#x27; denotes stop gradient.}
\end{figure}
\subsection{Analyze of} In this
\textbf{Similarity between} Using SimSiam
\textbf{Feature similarity} We extracted
\section{Conclusion}
We propose HCCL
\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></details></div>

#### ✨ explanation 解释
The operator removes the header content from the LaTeX document, including the preamble and title information, leaving only the main body of the text. The input data contains a full LaTeX document with headers, while the output data shows the document starting from the \section{Introduction} part, meaning all the preceding content (such as \documentclass, \title, \author, etc.) has been removed. In this case, the output data is the direct result of the operator's processing, without any further transformation.
算子从LaTeX文档中移除头部内容，包括前言和标题信息，仅留下正文部分。输入数据包含一个带有头部的完整LaTeX文档，而输出数据显示文档从\section{Introduction}部分开始，意味着所有前面的内容（如\documentclass、\title、\author等）已被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_header_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_header_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)