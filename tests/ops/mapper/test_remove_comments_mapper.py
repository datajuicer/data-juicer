import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.remove_comments_mapper import RemoveCommentsMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RemoveCommentsMapperTest(DataJuicerTestCaseBase):

    def _run_remove_comments(self, samples, op):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_inline_comments_preserve_preceding_text(self):
        samples = [
            {'text': 'Hello%comment', 'target': 'Hello'},
            {'text': 'A%comment\nB', 'target': 'A\nB'},
            {'text': r'\textbf{Title}%comment', 'target': r'\textbf{Title}'},
            {'text': 'Text %comment', 'target': 'Text '},
            {'text': r'50\% and text%comment', 'target': r'50\% and text'},
            {'text': r'50\% done', 'target': r'50\% done'},
            {'text': 'a\n%full line\nb', 'target': 'a\nb'},
        ]
        self._run_remove_comments(samples, RemoveCommentsMapper())

    def test_inline_only_preserves_full_line_comments(self):
        samples = [{
            'text': '%first\n%second\nBody%inline',
            'target': '%first\n%second\nBody',
        }, {
            'text': 'Head\r\n%comment\r\nTail%inline',
            'target': 'Head\r\n%comment\r\nTail',
        }]
        self._run_remove_comments(samples, RemoveCommentsMapper(multiline=False))

    def test_multiline_only_preserves_inline_comments(self):
        samples = [{
            'text': '%first\n%second\nBody%inline',
            'target': 'Body%inline',
        }]
        self._run_remove_comments(samples, RemoveCommentsMapper(inline=False))

    def test_no_percent_keeps_text_unchanged(self):
        # Neither pattern can match without a '%', so the text must pass
        # through untouched for every switch combination.
        samples = [
            {'text': 'Plain text without comments.',
             'target': 'Plain text without comments.'},
            {'text': '', 'target': ''},
            {'text': '中文 and emoji \U0001F600 stay',
             'target': '中文 and emoji \U0001F600 stay'},
            {'text': 'a\\b#c{d}\n e', 'target': 'a\\b#c{d}\n e'},
        ]
        for op in (RemoveCommentsMapper(),
                   RemoveCommentsMapper(multiline=False),
                   RemoveCommentsMapper(inline=False),
                   RemoveCommentsMapper(inline=False, multiline=False)):
            self._run_remove_comments(samples, op)

    def test_disabled_options_preserve_null_and_valid_rows(self):
        samples = [
            {'text': None, 'id': 0},
            {'text': 'Body%inline\n%full line', 'id': 1},
            {'text': 'Plain text', 'id': 2},
            {'text': '', 'id': 3},
        ]
        for skip_op_error in (False, True):
            with self.subTest(skip_op_error=skip_op_error):
                op = RemoveCommentsMapper(
                    inline=False, multiline=False,
                    skip_op_error=skip_op_error)
                batch = {'text': [row['text'] for row in samples],
                         'id': [row['id'] for row in samples]}
                self.assertIs(op.process_batched(batch), batch)
                self.assertEqual(batch['text'], [row['text'] for row in samples])
                dataset = Dataset.from_list(samples)
                result = dataset.map(op.process, batch_size=2)
                self.assertEqual(len(result), len(samples))
                self.assertEqual(result.to_list(), samples)

    def test_tex_case(self):

        samples = [{
            'text':
            "%%\n%% This is file `sample-sigconf.tex',\n%% The first command in your LaTeX source must be the \\documentclass command.\n\\documentclass[sigconf,review,anonymous]{acmart}\n%% NOTE that a single column version is required for \n%% submission and peer review. This can be done by changing\n\\input{math_commands.tex}\n%% end of the preamble, start of the body of the document source.\n\\begin{document}\n%% The \"title\" command has an optional parameter,\n\\title{Hierarchical Cross Contrastive Learning of Visual Representations}\n%%\n%% The \"author\" command and its associated commands are used to define\n%% the authors and their affiliations.\n\\author{Hesen Chen}\n\\affiliation{%\n  \\institution{Alibaba Group}\n  \\city{Beijing}\n  \\country{China}}\n\\email{hesen.chs@alibaba-inc.com}\n%% By default, the full list of authors will be used in the page\n\\begin{abstract}The rapid\n\\end{abstract}\n\\begin{CCSXML}\n\\ccsdesc[500]{Computing methodologies~Image representations}\n%% Keywords. The author(s) should pick words that accurately describe\n\\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}\n%% page.\n\\begin{teaserfigure}\n\\end{teaserfigure}\n%% This command processes the author and affiliation and title\n\\maketitle\n\\section{Introduction}\n\\begin{itemize}\n\\end{itemize}\n\\section{Related Work}\n\\label{gen_inst} Self-supervised\n\\section{Method}\n\\label{method}In this section,\n\\subsection{Framework} kkk\n\\subsection{Cross Contrastive Loss}\nSince $\\sZ^n$ are extracted\n\\subsection{Implementation details}\n\\textbf{Image augmentations} We use\n\\textbf{Architecture} We use\n\\textbf{Optimization} We adapt \n\\section{Experiments}\n\\label{experiments}In this section\n\\subsection{Linear and Semi-Supervised Evaluations on ImageNet}\n\\textbf{Linear evaluation on ImageNet} We firs\n\\textbf{Semi-supervised learning on ImageNet} We simply\n\\subsection{Transfer to other datasets and tasks}\n\\textbf{Image classification with fixed features} We follow\n\\section{Ablations} We present\n\\subsection{Influence of hierarchical projection head and cross contrastive loss} get out\n\\subsection{Levels and depth of projector network}\n\\end{center}\n\\caption{\\label{figure3} \\textbf{Different way of cross-correlation on 3 level hierarchical projection head.} '=' denotes stop gradient.}\n\\end{figure}\n\\subsection{Analyze of} In this\n\\textbf{Similarity between} Using SimSiam\n\\textbf{Feature similarity} We extracted\n\\section{Conclusion}\nWe propose HCCL\n\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{sample-base}\n\\end{document}\n\\endinput\n%%\n%% End of file `sample-sigconf.tex'.\n",  # noqa: E501
            'target':
            "\\documentclass[sigconf,review,anonymous]{acmart}\n\\input{math_commands.tex}\n\\begin{document}\n\\title{Hierarchical Cross Contrastive Learning of Visual Representations}\n\\author{Hesen Chen}\n\\affiliation{%\n  \\institution{Alibaba Group}\n  \\city{Beijing}\n  \\country{China}}\n\\email{hesen.chs@alibaba-inc.com}\n\\begin{abstract}The rapid\n\\end{abstract}\n\\begin{CCSXML}\n\\ccsdesc[500]{Computing methodologies~Image representations}\n\\keywords{self-supervised,  ontrastive Learning, hierarchical projection, cross-level}\n\\begin{teaserfigure}\n\\end{teaserfigure}\n\\maketitle\n\\section{Introduction}\n\\begin{itemize}\n\\end{itemize}\n\\section{Related Work}\n\\label{gen_inst} Self-supervised\n\\section{Method}\n\\label{method}In this section,\n\\subsection{Framework} kkk\n\\subsection{Cross Contrastive Loss}\nSince $\\sZ^n$ are extracted\n\\subsection{Implementation details}\n\\textbf{Image augmentations} We use\n\\textbf{Architecture} We use\n\\textbf{Optimization} We adapt \n\\section{Experiments}\n\\label{experiments}In this section\n\\subsection{Linear and Semi-Supervised Evaluations on ImageNet}\n\\textbf{Linear evaluation on ImageNet} We firs\n\\textbf{Semi-supervised learning on ImageNet} We simply\n\\subsection{Transfer to other datasets and tasks}\n\\textbf{Image classification with fixed features} We follow\n\\section{Ablations} We present\n\\subsection{Influence of hierarchical projection head and cross contrastive loss} get out\n\\subsection{Levels and depth of projector network}\n\\end{center}\n\\caption{\\label{figure3} \\textbf{Different way of cross-correlation on 3 level hierarchical projection head.} '=' denotes stop gradient.}\n\\end{figure}\n\\subsection{Analyze of} In this\n\\textbf{Similarity between} Using SimSiam\n\\textbf{Feature similarity} We extracted\n\\section{Conclusion}\nWe propose HCCL\n\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{sample-base}\n\\end{document}\n\\endinput\n"  # noqa: E501
        }]

        op = RemoveCommentsMapper(doc_type='tex', inline=True, multiline=True)
        self._run_remove_comments(samples, op)


if __name__ == '__main__':
    unittest.main()
