import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_html_mapper import CleanHtmlMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CleanHtmlMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.op = CleanHtmlMapper()

    def _run_helper(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_complete_html_text(self):

        samples = [
            {
                'text':
                '<header><nav><ul>'
                '<tile>测试</title>'
                '<li><a href="#">Home</a></li>'
                '<li><a href="#">About</a></li>'
                '<li><a href="#">Services</a></li>'
                '<li><a href="#">Contact</a></li></ul></nav></header>'
                '<main><h1>Welcome to My Website</h1>'
                '<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
                '<button>Learn More</button></main>'
                '<footer><p>&copy; '
                '2021 My Website. All Rights Reserved.</p></footer>',
                'target':
                '测试\n*Home\n*About\n*Services\n*Contact'
                'Welcome to My WebsiteLorem ipsum dolor sit amet, '
                'consectetur adipiscing elit.'
                'Learn More© 2021 My Website. All Rights Reserved.'
            },
        ]
        self._run_helper(samples)

    def test_html_en_text(self):

        samples = [
            {
                'text': '<p>This is a test</p>',
                'target': 'This is a test'
            },
            {
                'text':
                '<a href=\'https://www.example.com/file.html?;name=Test\''
                ' rel=\'noopener noreferrer\' target=\'_blank\'>Test</a>',
                'target': 'Test'
            },
            {
                'text':
                '<p>This is a test</p>'
                '<div class=\"article-content\"><div><p>This is a test</p>'
                '<p>This is a test</p>'
                '<p><span>Test</span>：This is a test</p>'
                '<p><span>Test</span>：This is a test</p></div></div>'
                '<p></p><div></div>',
                'target':
                'This is a test'
                'This is a test'
                'This is a test'
                'Test：This is a test'
                'Test：This is a test'
            },
        ]

        self._run_helper(samples)

    def test_html_zh_text(self):

        samples = [
            {
                'text': '<p>这是个测试</p>',
                'target': '这是个测试'
            },
            {
                'text':
                '<a href=\'https://www.example.com/file.html?;name=Test\''
                ' rel=\'noopener noreferrer\' target=\'_blank\'>测试</a>',
                'target': '测试'
            },
            {
                'text':
                '<p>这是1个测试。</p>'
                '<div class=\"article-content\"><div><p>这是2个测试。</p>'
                '<p>这是3个测试。</p>'
                '<p><span>测试</span>：这是4个测试。</p>'
                '<p><span>测试</span>：这是5个测试。</p></div></div>'
                '<p></p><div></div>',
                'target':
                '这是1个测试。'
                '这是2个测试。'
                '这是3个测试。'
                '测试：这是4个测试。'
                '测试：这是5个测试。'
            },
        ]
        self._run_helper(samples)

    def test_no_html_text(self):

        samples = [
            {
                'text': 'This is a test',
                'target': 'This is a test'
            },
            {
                'text': '这是个测试',
                'target': '这是个测试'
            },
            {
                'text': '12345678',
                'target': '12345678'
            },
        ]
        self._run_helper(samples)

    def test_fake_html_text(self):

        samples = [
            {
                'text': 'This is a test</p>',
                'target': 'This is a test'
            },
            {
                'text': '<p>这是个测试',
                'target': '这是个测试'
            },
            {
                'text': '<kkkeabcd>hello</kkkeabcd>',
                'target': 'hello'
            },
            {
                'text': '<a1bcd>这是个测试</a1bcd>',
                'target': '这是个测试'
            },
            {
                'text': '<测试>这是个测试</测试>',
                'target': '<测试>这是个测试'
            },
            {
                'text':
                'abc="https://www.example.com/file.html?name=Test\" 测试',
                'target':
                'abc="https://www.example.com/file.html?name=Test" 测试'
            },
            {
                'text':
                'href="https://www.example.com/file.html;name=Test">测试',
                'target':
                'href="https://www.example.com/file.html;name=Test">测试'
            },
            {
                'text':
                '<abc="https://www.example.com/file.html?name=Test">测试',
                'target': '测试'
            },
        ]
        self._run_helper(samples)

    def test_whitespace_text(self):

        samples = [
            {
                'text': '    ',
                'target': ''
            },
            {
                'text': '',
                'target': ''
            },
            {
                'text': '    This is a test',
                'target': 'This is a test'
            },
            {
                'text': '    This is a test  ',
                'target': 'This is a test  '
            },
        ]
        self._run_helper(samples)

    def test_only_list_text(self):

        samples = [
            {
                'text': '<li>Apple</li>',
                'target': '*Apple'
            },
            {
                'text': '<li>苹果</li>',
                'target': '*苹果'
            },
            {
                'text': '<ol>Apple</ol>',
                'target': '*Apple'
            },
            {
                'text': '<ol>苹果</ol>',
                'target': '*苹果'
            },
        ]

        self._run_helper(samples)


if __name__ == '__main__':
    unittest.main()
