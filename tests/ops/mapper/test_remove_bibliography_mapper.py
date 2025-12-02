import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.remove_bibliography_mapper import \
    RemoveBibliographyMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RemoveBibliographyMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.op = RemoveBibliographyMapper()

    def _run_remove_bibliography(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_bibliography_case(self):

        samples = [{
            'text':
            "%%\n%% This is file `sample-sigconf.tex\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{sample-base}\n\\end{document}\n\\endinput\n%%\n%% End of file `sample-sigconf.tex'.\n",  # noqa: E501
            'target':
            '%%\n%% This is file `sample-sigconf.tex\\clearpage\n\\bibliographystyle{ACM-Reference-Format}\n'  # noqa: E501
        }]

        self._run_remove_bibliography(samples)

    def test_ref_case(self):

        samples = [{
            'text':
            "%%\n%% This is file `sample-sigconf.tex\\clearpage\n\\begin{references}\n\\end{document}\n\\endinput\n%%\n%% End of file `sample-sigconf.tex'.\n",  # noqa: E501
            'target':
            '%%\n%% This is file `sample-sigconf.tex\\clearpage\n'  # noqa: E501
        }]

        self._run_remove_bibliography(samples)


if __name__ == '__main__':
    unittest.main()
