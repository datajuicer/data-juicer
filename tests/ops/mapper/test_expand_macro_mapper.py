import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.expand_macro_mapper import ExpandMacroMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExpandMacroMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.op = ExpandMacroMapper()

    def _run_expand_macro(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_case(self):

        samples = [{
            'text':
            '\\documentclass{article}\n% Recommended, but optional, packages for figures and better typesetting:\n\\usepackage{microtype}\n\\usepackage{graphicx}\n\n% Attempt to make hyperref and algorithmic work together better:\n\\newcommand{\\theHalgorithm}{\\arabic{algorithm}}\n% For theorems and such\n\\usepackage{amsmath}\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n% THEOREMS\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\\theoremstyle{plain}\n\\newtheorem{lemma}[theorem]{Lemma}\n\\newtheorem{corollary}[theorem]{Corollary}\n\\theoremstyle{definition}\n\n\\usepackage[textsize=small]{todonotes}\n\\setuptodonotes{inline}\n\n\\usepackage{makecell}\n\\newcommand{\\cmark}{\\ding{51}\\xspace}%\n\\newcommand{\\xmark}{\\ding{55}\\xspace}%\n\n\\def \\alambic {\\includegraphics[height=1.52ex]{img/alembic-crop.pdf}\\xspace}\n\n\\newcommand\\binke[1]{{\\color{blue} \\footnote{\\color{blue}binke: #1}} }\n\\newcommand\\Zerocost{Zero-cost}\n\\newcommand\\imagenet{ImageNet}\n\n\\begin{document}\n\n\\begin{abstract}\nThe wide\n\\end{abstract}\n\\section{Introduction}\n\\label{introduction}\nThe main contributions are summarized as follows:\n\\section{Background and Related Work}\\label{background}\n\\subsection{One-Shot NAS} In one-shot NAS\n\\section{PreNAS}\\label{method}In this\n\\subsection{One-Shot NAS with Preferred Learning}\nIn the specialization stage, the optimal architectures under given  resource constraints can be directly obtained:\n\\begin{equation}\n\\widetilde{\\mathcal{A}}^* = \\widetilde{\\mathcal{A}} .\n\\end{equation}\n\\subsection{Zero-Cost Transformer Selector}\\label{sub:layerNorm}\n\\subsection{Performance Balancing} We discuss\n\\section{Experiments}\\label{experiments}\n\\subsection{Setup}\n\\subsection{Main Results}\\label{sec:sota}\n\\subsection{Analysis and Ablation study}\\label{ablation}\n\\begin{figure}[t]\n\\vskip 0.1in\n    \\centering\n    \\subfigure[Search spaces]{\\includegraphics[width=0.36\\linewidth]{img/search_space.pdf}\\label{fg:search_space:a}}%\n    \\hfil%\n    \\subfigure[Error distributions]{\\includegraphics[width=0.58\\linewidth]{img/cumulation.pdf}\\label{fg:search_space:b}}\n    \\caption{Model quality}\n\\vskip -0.1in\n\\end{figure}\n\\paragraph{Effect of Performance Balancing} During\n\\subsection{Transfer Learning Results}\n\\subsection{CNN Results} in terms of similar FLOPs.\n\\FloatBarrier\n\\section{Conclusion}\\label{conclusion} In this\n% Acknowledgements should only appear in the accepted version.\n\\bibliography{ref}\n\\bibliographystyle{icml2023}\n\\clearpage\n\\appendix\n\\onecolumn\n\\section{Statistical}\n\\label{appendix:snipAnalysis} We analyze\n\\section{The Greedy Algorithm}\n\\label{appendix:greedy}\n\\section{Regularization \\& Data Augmentation}\\label{appendix:aug}\n\\renewcommand{\\arraystretch}{1.2}\n\\end{document}\n',  # noqa: E501
            'target':
            '\\documentclass{article}\n% Recommended, but optional, packages for figures and better typesetting:\n\\usepackage{microtype}\n\\usepackage{graphicx}\n\n% Attempt to make hyperref and algorithmic work together better:\n\\newcommand{\\arabic{algorithm}}{\\arabic{algorithm}}\n% For theorems and such\n\\usepackage{amsmath}\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n% THEOREMS\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\\theoremstyle{plain}\n\\newtheorem{lemma}[theorem]{Lemma}\n\\newtheorem{corollary}[theorem]{Corollary}\n\\theoremstyle{definition}\n\n\\usepackage[textsize=small]{todonotes}\n\\setuptodonotes{inline}\n\n\\usepackage{makecell}\n\\newcommand{\\cmark}{\\ding{51}\\xspace}%\n\\newcommand{\\xmark}{\\ding{55}\\xspace}%\n\n\\def \\includegraphics[height=1.52ex]{img/alembic-crop.pdf}\\xspace {\\includegraphics[height=1.52ex]{img/alembic-crop.pdf}\\xspace}\n\n\\newcommand\\binke[1]{{\\color{blue} \\footnote{\\color{blue}binke: #1}} }\n\\newcommand\\Zerocost{Zero-cost}\n\\newcommand\\imagenet{ImageNet}\n\n\\begin{document}\n\n\\begin{abstract}\nThe wide\n\\end{abstract}\n\\section{Introduction}\n\\label{introduction}\nThe main contributions are summarized as follows:\n\\section{Background and Related Work}\\label{background}\n\\subsection{One-Shot NAS} In one-shot NAS\n\\section{PreNAS}\\label{method}In this\n\\subsection{One-Shot NAS with Preferred Learning}\nIn the specialization stage, the optimal architectures under given  resource constraints can be directly obtained:\n\\begin{equation}\n\\widetilde{\\mathcal{A}}^* = \\widetilde{\\mathcal{A}} .\n\\end{equation}\n\\subsection{Zero-Cost Transformer Selector}\\label{sub:layerNorm}\n\\subsection{Performance Balancing} We discuss\n\\section{Experiments}\\label{experiments}\n\\subsection{Setup}\n\\subsection{Main Results}\\label{sec:sota}\n\\subsection{Analysis and Ablation study}\\label{ablation}\n\\begin{figure}[t]\n\\vskip 0.1in\n    \\centering\n    \\subfigure[Search spaces]{\\includegraphics[width=0.36\\linewidth]{img/search_space.pdf}\\label{fg:search_space:a}}%\n    \\hfil%\n    \\subfigure[Error distributions]{\\includegraphics[width=0.58\\linewidth]{img/cumulation.pdf}\\label{fg:search_space:b}}\n    \\caption{Model quality}\n\\vskip -0.1in\n\\end{figure}\n\\paragraph{Effect of Performance Balancing} During\n\\subsection{Transfer Learning Results}\n\\subsection{CNN Results} in terms of similar FLOPs.\n\\FloatBarrier\n\\section{Conclusion}\\label{conclusion} In this\n% Acknowledgements should only appear in the accepted version.\n\\bibliography{ref}\n\\bibliographystyle{icml2023}\n\\clearpage\n\\appendix\n\\onecolumn\n\\section{Statistical}\n\\label{appendix:snipAnalysis} We analyze\n\\section{The Greedy Algorithm}\n\\label{appendix:greedy}\n\\section{Regularization \\& Data Augmentation}\\label{appendix:aug}\n\\renewcommand{\\arraystretch}{1.2}\n\\end{document}\n'  # noqa: E501
        }]

        self._run_expand_macro(samples)


if __name__ == '__main__':
    unittest.main()
