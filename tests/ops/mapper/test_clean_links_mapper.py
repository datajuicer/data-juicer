import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_links_mapper import CleanLinksMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CleanLinksMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.op = CleanLinksMapper()

    def _run_clean_links(self, op, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_lower_ftp_links_text(self):

        samples = [{
            'text': 'ftp://user:password@ftp.example.com:21/',
            'target': ''
        }, {
            'text': 'ftp://www.example.com/path/to/file.txt',
            'target': ''
        }, {
            'text': 'ftp://example.com/my-page.html',
            'target': ''
        }, {
            'text': 'ftp://example.com',
            'target': ''
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_upper_ftp_links_text(self):

        samples = [{
            'text': 'FTP://user:password@ftp.example.COm:21/',
            'target': ''
        }, {
            'text': 'FTP://www.example.com/path/to/file.txt',
            'target': ''
        }, {
            'text': 'Ftp://example.com/my-page.HTMl',
            'target': ''
        }, {
            'text': 'FTP://EXAMPLE.COM',
            'target': ''
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_lower_https_links_text(self):

        samples = [{
            'text':
            'https://www.example.com/file.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text':
            'https://example.com/my-page.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text': 'https://example.com',
            'target': ''
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_upper_https_links_text(self):

        samples = [{
            'text':
            'hTTps://www.example.com/file.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text':
            'HttpS://example.Com/my-page.HTML?param1=value1&param2=value2',
            'target': ''
        }, {
            'text': 'HTTPS://EXAMPLE.COM',
            'target': ''
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_mixed_https_links_text(self):

        samples = [{
            'text': 'This is a test,'
            'https://www.example.com/file.html?param1=value1&param2=value2',
            'target': 'This is a test,'
        }, {
            'text': '这是个测试,'
            'https://example.com/my-page.html?param1=value1&param2=value2',
            'target': '这是个测试,'
        }, {
            'text': '这是个测试,https://example.com',
            'target': '这是个测试,'
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_lower_http_links_text(self):

        samples = [{
            'text':
            'http://example.com/my-page.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text':
            'http://www.example.com/file.html?param1=value1&param2=value2',
            'target': ''
        }, {
            'text': 'https://example.com',
            'target': ''
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_upper_http_links_text(self):

        samples = [
            {
                'text':
                'HTTP://example.com/my-page.html?param1=value1&param2=value2',
                'target': ''
            },
            {
                'text':
                'hTTp://www.example.com/file.html?param1=value1&param2=value2',
                'target': ''
            },
            {
                'text': 'HTTPS://EXAMPLE.COM',
                'target': ''
            },
        ]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_mixed_http_links_text(self):

        samples = [{
            'text': 'This is a test,'
            'http://www.example.com/file.html?param1=value1&param2=value2',
            'target': 'This is a test,'
        }, {
            'text': '这是个测试,'
            'http://example.com/my-page.html?param1=value1&param2=value2',
            'target': '这是个测试,'
        }, {
            'text': '这是个测试,https://example.com',
            'target': '这是个测试,'
        }]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_email_text(self):

        samples = [
            {
                'text': 'This is a sample@example for test',
                'target': 'This is a sample@example for test',
            },
            {
                'text': '这是一个测试, sample@example',
                'target': '这是一个测试, sample@example',
            },
        ]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_fake_links_text(self):

        samples = [
            {
                'text': 'abcd:/e f is a sample for test',
                'target': 'abcd:/e f is a sample for test',
            },
            {
                'text': 'abcd://ef is a sample for test',
                'target': ' is a sample for test',
            },
            {
                'text': 'This is a test,'
                'http测试://www.example.com/file.html?param1=value1',
                'target': 'This is a test,'
            },
            {
                'text': 'This is a test,'
                'http://www.测试.com/path/file.html?param1=value1&param2=value2',
                'target': 'This is a test,'
            },
        ]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_no_link_text(self):

        samples = [
            {
                'text': 'This is a sample for test',
                'target': 'This is a sample for test',
            },
            {
                'text': '这是一个测试',
                'target': '这是一个测试',
            },
            {
                'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►',
                'target': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►',
            },
        ]
        op = CleanLinksMapper()
        self._run_clean_links(op, samples)

    def test_replace_links_text(self):

        samples = [
            {
                'text': 'ftp://user:password@ftp.example.com:21/',
                'target': '<LINKS>'
            },
            {
                'text': 'This is a sample for test',
                'target': 'This is a sample for test',
            },
            {
                'text': 'abcd://ef is a sample for test',
                'target': '<LINKS> is a sample for test',
            },
            {
                'text':
                'HTTP://example.com/my-page.html?param1=value1&param2=value2',
                'target': '<LINKS>'
            },
        ]
        op = CleanLinksMapper(repl='<LINKS>')
        self._run_clean_links(op, samples)


if __name__ == '__main__':
    unittest.main()
