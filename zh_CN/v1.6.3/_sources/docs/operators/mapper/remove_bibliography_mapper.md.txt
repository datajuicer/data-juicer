# remove_bibliography_mapper

Removes bibliography sections at the end of LaTeX documents.

This operator identifies and removes bibliography sections in LaTeX documents. It uses a regular expression to match common bibliography commands such as \appendix, \begin{references}, \begin{thebibliography}, and \bibliography. The matched sections are removed from the text. The operator processes samples in batch mode for efficiency.

移除LaTeX文档末尾的参考文献部分。

该算子识别并移除LaTeX文档中的参考文献部分。它使用正则表达式匹配常见的参考文献命令，如\appendix、\begin{references}、\begin{thebibliography}和\bibliography。匹配的部分将从文本中删除。该算子以批量模式处理样本以提高效率。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_bibliography_case
```python
RemoveBibliographyMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
\bibliographystyle{ACM-Reference-Format}
</pre></div>

#### ✨ explanation 解释
This example shows how the operator removes a bibliography section marked by the \bibliography command. The input contains a LaTeX document with a bibliography at the end. After processing, the bibliography and everything after it is removed, leaving only the part of the document before the bibliography.
这个例子展示了算子如何移除用\bibliography命令标记的参考文献部分。输入包含一个在末尾有参考文献的LaTeX文档。处理后，参考文献及其后面的所有内容都被移除，只保留了参考文献前的部分文档内容。

### test_ref_case
```python
RemoveBibliographyMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
\begin{references}
\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex&#x27;.
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">%%
%% This is file `sample-sigconf.tex\clearpage
</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's capability to remove a references section indicated by the \begin{references} command. The input text has a references section at the end. After the operator processes the data, the references section and all content following it are eliminated, preserving only the text prior to the references section.
这个例子展示了算子移除用\begin{references}命令标记的参考文献部分的能力。输入文本在末尾有一个参考文献部分。经过算子处理后，参考文献部分及其后面的所有内容都被移除，只保留了参考文献前的部分文本。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_bibliography_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_bibliography_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)