import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.latex_figure_context_extractor_mapper import (
    LatexFigureContextExtractorMapper,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LatexFigureContextExtractorMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.op = LatexFigureContextExtractorMapper()

    def _run_mapper(self, samples):
        """Helper: run the batched mapper on a list of dicts."""
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(
            self.op.process, batch_size=len(samples),
        )
        return dataset.to_list()

    # ------------------------------------------------------------------
    # 1. Single figure with caption, label, and \includegraphics
    # ------------------------------------------------------------------
    def test_single_figure(self):
        latex = (
            '\\begin{document}\n'
            'Some intro text.\n\n'
            'As shown in \\ref{fig:arch}, the architecture is novel.\n\n'
            '\\begin{figure}\n'
            '\\centering\n'
            '\\includegraphics[width=0.8\\linewidth]{img/arch.pdf}\n'
            '\\caption{Overall architecture}\n'
            '\\label{fig:arch}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'Overall architecture')
        self.assertEqual(results[0]['label'], 'fig:arch')
        self.assertEqual(results[0]['images'], ['img/arch.pdf'])
        self.assertEqual(len(results[0]['citing_paragraphs']), 1)
        self.assertIn('\\ref{fig:arch}',
                       results[0]['citing_paragraphs'][0])
        # Standalone figure: parent fields are empty
        self.assertEqual(results[0]['parent_caption'], '')
        self.assertEqual(results[0]['parent_label'], '')

    # ------------------------------------------------------------------
    # 2. Figure with \begin{subfigure} environments (modern subcaption)
    # ------------------------------------------------------------------
    def test_subfigure_environments(self):
        latex = (
            '\\begin{document}\n'
            'See \\cref{fig:main} for details.\n\n'
            'Also \\ref{fig:sub_b} is interesting.\n\n'
            '\\begin{figure}\n'
            '\\centering\n'
            '\\begin{subfigure}\n'
            '\\includegraphics{img/a.png}\n'
            '\\caption{Sub A}\n'
            '\\label{fig:sub_a}\n'
            '\\end{subfigure}\n'
            '\\begin{subfigure}\n'
            '\\includegraphics{img/b.png}\n'
            '\\caption{Sub B}\n'
            '\\label{fig:sub_b}\n'
            '\\end{subfigure}\n'
            '\\caption{Main caption}\n'
            '\\label{fig:main}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        # Should produce 2 rows (one per subfigure)
        self.assertEqual(len(results), 2)

        # Sub A
        self.assertEqual(results[0]['caption'], 'Sub A')
        self.assertEqual(results[0]['label'], 'fig:sub_a')
        self.assertEqual(results[0]['images'], ['img/a.png'])
        # Sub A inherits parent-level \cref{fig:main} citation
        self.assertTrue(
            any('\\cref{fig:main}' in p
                for p in results[0]['citing_paragraphs'])
        )
        # Sub A carries parent info
        self.assertEqual(results[0]['parent_caption'], 'Main caption')
        self.assertEqual(results[0]['parent_label'], 'fig:main')

        # Sub B
        self.assertEqual(results[1]['caption'], 'Sub B')
        self.assertEqual(results[1]['label'], 'fig:sub_b')
        self.assertEqual(results[1]['images'], ['img/b.png'])
        # Sub B has both parent citation and its own \ref{fig:sub_b}
        contexts_b = results[1]['citing_paragraphs']
        self.assertTrue(
            any('\\cref{fig:main}' in p for p in contexts_b)
        )
        self.assertTrue(
            any('\\ref{fig:sub_b}' in p for p in contexts_b)
        )
        # Sub B also carries same parent info
        self.assertEqual(results[1]['parent_caption'], 'Main caption')
        self.assertEqual(results[1]['parent_label'], 'fig:main')

    # ------------------------------------------------------------------
    # 3. Figure with \subfigure[]{} commands (older subfig package)
    # ------------------------------------------------------------------
    def test_subfigure_commands(self):
        latex = (
            '\\begin{document}\n'
            'Refer to \\ref{fig:old}.\n\n'
            '\\begin{figure}\n'
            '\\centering\n'
            '\\subfigure[Caption X]{\n'
            '  \\includegraphics{img/x.pdf}\n'
            '  \\label{fig:x}\n'
            '}\n'
            '\\subfigure[Caption Y]{\n'
            '  \\includegraphics{img/y.pdf}\n'
            '  \\label{fig:y}\n'
            '}\n'
            '\\caption{Old style}\n'
            '\\label{fig:old}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['caption'], 'Caption X')
        self.assertEqual(results[0]['images'], ['img/x.pdf'])
        self.assertEqual(results[0]['parent_caption'], 'Old style')
        self.assertEqual(results[0]['parent_label'], 'fig:old')
        self.assertEqual(results[1]['caption'], 'Caption Y')
        self.assertEqual(results[1]['images'], ['img/y.pdf'])
        self.assertEqual(results[1]['parent_caption'], 'Old style')
        self.assertEqual(results[1]['parent_label'], 'fig:old')

    # ------------------------------------------------------------------
    # 4. Figure without \includegraphics is skipped
    # ------------------------------------------------------------------
    def test_figure_without_images_skipped(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\caption{No image here}\n'
            '\\label{fig:empty}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 0)

    # ------------------------------------------------------------------
    # 5. No figures at all — sample is dropped
    # ------------------------------------------------------------------
    def test_no_figures_drops_sample(self):
        latex = (
            '\\begin{document}\n'
            'Just some text, no figures.\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 0)

    # ------------------------------------------------------------------
    # 6. Citation via comma-separated \cref{fig:a,fig:b}
    # ------------------------------------------------------------------
    def test_comma_separated_cref(self):
        latex = (
            '\\begin{document}\n'
            'See \\cref{fig:a,fig:b} for comparison.\n\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/a.png}\n'
            '\\caption{Figure A}\n'
            '\\label{fig:a}\n'
            '\\end{figure}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/b.png}\n'
            '\\caption{Figure B}\n'
            '\\label{fig:b}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 2)
        # Both figures should find the paragraph with \cref{fig:a,fig:b}
        for r in results:
            self.assertTrue(
                any('\\cref{fig:a,fig:b}' in p
                    for p in r['citing_paragraphs'])
            )

    # ------------------------------------------------------------------
    # 7. Multiple figures in one document
    # ------------------------------------------------------------------
    def test_multiple_figures(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/one.png}\n'
            '\\caption{First}\n'
            '\\label{fig:one}\n'
            '\\end{figure}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/two.png}\n'
            '\\caption{Second}\n'
            '\\label{fig:two}\n'
            '\\end{figure}\n'
            '\\begin{figure*}\n'
            '\\includegraphics{img/three.png}\n'
            '\\caption{Third wide}\n'
            '\\label{fig:three}\n'
            '\\end{figure*}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 3)
        captions = [r['caption'] for r in results]
        self.assertEqual(captions, ['First', 'Second', 'Third wide'])

    # ------------------------------------------------------------------
    # 8. figure* environment is recognized
    # ------------------------------------------------------------------
    def test_figure_star(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure*}\n'
            '\\includegraphics{img/wide.png}\n'
            '\\caption{Wide figure}\n'
            '\\label{fig:wide}\n'
            '\\end{figure*}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'Wide figure')

    # ------------------------------------------------------------------
    # 9. wrapfigure environment
    # ------------------------------------------------------------------
    def test_wrapfigure(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{wrapfigure}{r}{0.5\\textwidth}\n'
            '\\includegraphics{img/wrap.png}\n'
            '\\caption{Wrapped}\n'
            '\\label{fig:wrap}\n'
            '\\end{wrapfigure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'Wrapped')
        self.assertEqual(results[0]['label'], 'fig:wrap')

    # ------------------------------------------------------------------
    # 10. Nested caption braces (e.g. \textbf{...} inside caption)
    # ------------------------------------------------------------------
    def test_nested_caption_braces(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/nest.png}\n'
            '\\caption{A \\textbf{bold \\emph{italic}} caption}\n'
            '\\label{fig:nest}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertIn('\\textbf{bold \\emph{italic}}',
                       results[0]['caption'])

    # ------------------------------------------------------------------
    # 10b. Deeply nested caption braces (5+ levels)
    # ------------------------------------------------------------------
    def test_deeply_nested_caption_braces(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/deep.png}\n'
            '\\caption{A \\textbf{B \\emph{C \\footnote{D \\cite{E}}}}}\n'
            '\\label{fig:deep}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertIn(
            '\\textbf{B \\emph{C \\footnote{D \\cite{E}}}}',
            results[0]['caption'],
        )

    # ------------------------------------------------------------------
    # 11. \captionof{figure}{...} is recognized
    # ------------------------------------------------------------------
    def test_captionof(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/cof.png}\n'
            '\\captionof{figure}{Caption via captionof}\n'
            '\\label{fig:cof}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'],
                         'Caption via captionof')

    # ------------------------------------------------------------------
    # 12. \autoref citation command
    # ------------------------------------------------------------------
    def test_autoref(self):
        latex = (
            '\\begin{document}\n'
            'See \\autoref{fig:auto} for details.\n\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/auto.png}\n'
            '\\caption{Auto}\n'
            '\\label{fig:auto}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]['citing_paragraphs']), 1)

    # ------------------------------------------------------------------
    # 13. Label fig:a must not false-match fig:ab
    # ------------------------------------------------------------------
    def test_label_boundary(self):
        latex = (
            '\\begin{document}\n'
            'See \\ref{fig:ab} for details.\n\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/a.png}\n'
            '\\caption{A}\n'
            '\\label{fig:a}\n'
            '\\end{figure}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/ab.png}\n'
            '\\caption{AB}\n'
            '\\label{fig:ab}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 2)
        # fig:a should NOT match the paragraph citing \ref{fig:ab}
        fig_a = [r for r in results if r['label'] == 'fig:a'][0]
        self.assertEqual(fig_a['citing_paragraphs'], [])
        # fig:ab should match
        fig_ab = [r for r in results if r['label'] == 'fig:ab'][0]
        self.assertEqual(len(fig_ab['citing_paragraphs']), 1)

    # ------------------------------------------------------------------
    # 14. Custom output keys
    # ------------------------------------------------------------------
    def test_custom_keys(self):
        op = LatexFigureContextExtractorMapper(
            caption_key='fig_caption',
            label_key='fig_label',
            context_key='fig_context',
        )
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/custom.png}\n'
            '\\caption{Custom}\n'
            '\\label{fig:custom}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        dataset = Dataset.from_list([{'text': latex}])
        dataset = dataset.map(op.process, batch_size=1)
        results = dataset.to_list()
        self.assertEqual(len(results), 1)
        self.assertIn('fig_caption', results[0])
        self.assertIn('fig_label', results[0])
        self.assertIn('fig_context', results[0])
        self.assertEqual(results[0]['fig_caption'], 'Custom')

    # ------------------------------------------------------------------
    # 15. Multiple samples in one batch (fan-out + drop)
    # ------------------------------------------------------------------
    def test_batch_mixed(self):
        latex_with_fig = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/ok.png}\n'
            '\\caption{OK}\n'
            '\\label{fig:ok}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        latex_no_fig = (
            '\\begin{document}\n'
            'No figures here.\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([
            {'text': latex_with_fig},
            {'text': latex_no_fig},
        ])
        # Only the first sample produces output
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'OK')


    # ------------------------------------------------------------------
    # 16. \caption[short]{long} — optional short caption
    # ------------------------------------------------------------------
    def test_caption_with_short_form(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/short.png}\n'
            '\\caption[Short form]{Long detailed caption}\n'
            '\\label{fig:short}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'Long detailed caption')

    # ------------------------------------------------------------------
    # 17. \subcaption[short]{long} — optional short caption
    # ------------------------------------------------------------------
    def test_subcaption_with_short_form(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\centering\n'
            '\\begin{subfigure}\n'
            '\\includegraphics{img/sc.png}\n'
            '\\subcaption[Short sub]{Long subcaption text}\n'
            '\\label{fig:sc}\n'
            '\\end{subfigure}\n'
            '\\caption{Parent}\n'
            '\\label{fig:parent_sc}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'Long subcaption text')
        self.assertEqual(results[0]['parent_caption'], 'Parent')

    # ------------------------------------------------------------------
    # 18. \captionof{figure}[short]{long} — optional short caption
    # ------------------------------------------------------------------
    def test_captionof_with_short_form(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/cof2.png}\n'
            '\\captionof{figure}[Short cof]{Long captionof text}\n'
            '\\label{fig:cof2}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], 'Long captionof text')


    # ------------------------------------------------------------------
    # 19. \subfloat[caption]{content} — subfig package
    # ------------------------------------------------------------------
    def test_subfloat_with_caption(self):
        latex = (
            '\\begin{document}\n'
            'See \\ref{fig:sf_parent} for comparison.\n\n'
            '\\begin{figure}\n'
            '\\centering\n'
            '\\subfloat[Float A]{\n'
            '  \\includegraphics{img/fa.png}\n'
            '  \\label{fig:fa}\n'
            '}\n'
            '\\subfloat[Float B]{\n'
            '  \\includegraphics{img/fb.png}\n'
            '  \\label{fig:fb}\n'
            '}\n'
            '\\caption{Subfloat parent}\n'
            '\\label{fig:sf_parent}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['caption'], 'Float A')
        self.assertEqual(results[0]['images'], ['img/fa.png'])
        self.assertEqual(results[0]['parent_caption'], 'Subfloat parent')
        self.assertEqual(results[0]['parent_label'], 'fig:sf_parent')
        self.assertEqual(results[1]['caption'], 'Float B')
        self.assertEqual(results[1]['images'], ['img/fb.png'])
        self.assertEqual(results[1]['parent_caption'], 'Subfloat parent')
        # Parent citation inherited
        self.assertTrue(
            any('\\ref{fig:sf_parent}' in p
                for p in results[0]['citing_paragraphs'])
        )

    # ------------------------------------------------------------------
    # 20. \subfloat{content} — without optional caption
    # ------------------------------------------------------------------
    def test_subfloat_without_caption(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\centering\n'
            '\\subfloat{\n'
            '  \\includegraphics{img/nc.png}\n'
            '  \\label{fig:nc}\n'
            '}\n'
            '\\caption{No-caption subfloats}\n'
            '\\label{fig:nc_parent}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'], '')
        self.assertEqual(results[0]['images'], ['img/nc.png'])
        self.assertEqual(results[0]['parent_caption'],
                         'No-caption subfloats')

    # ------------------------------------------------------------------
    # 21. \caption*{...} — unnumbered caption
    # ------------------------------------------------------------------
    def test_caption_star(self):
        latex = (
            '\\begin{document}\n'
            '\\begin{figure}\n'
            '\\includegraphics{img/star.png}\n'
            '\\caption*{Unnumbered caption text}\n'
            '\\label{fig:star}\n'
            '\\end{figure}\n'
            '\\end{document}\n'
        )
        results = self._run_mapper([{'text': latex}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['caption'],
                         'Unnumbered caption text')


if __name__ == '__main__':
    unittest.main()
