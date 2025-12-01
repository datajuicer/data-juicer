import unittest

from data_juicer.ops.mapper.extract_tables_from_html_mapper import ExtractTablesFromHtmlMapper
from data_juicer.utils.constant import Fields, MetaKeys

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from loguru import logger


class ExtractTablesFromHtmlMapperTest(DataJuicerTestCaseBase):
    raw_html = """
    <!DOCTYPE html>
            <html lang="zh">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>表格示例</title>
            </head>
            <body>
                <h1>表格示例</h1>
                <table border="1">
                    <thead>
                        <tr>
                            <th>姓名</th>
                            <th>年龄</th>
                            <th>城市</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>张三</td>
                            <td>25</td>
                            <td>北京</td>
                        </tr>
                        <tr>
                            <td>李四</td>
                            <td>30</td>
                            <td>上海</td>
                        </tr>
                        <tr>
                            <td>王五</td>
                            <td>28</td>
                            <td>广州</td>
                        </tr>
                    </tbody>
                </table>
            </body>
            </html>
    """

    def _run_mapper(self, op, source_list):
        dataset = Dataset.from_list(source_list)
        dataset = op.run(dataset)
        sample = dataset[0]
        self.assertIn(MetaKeys.html_tables, sample[Fields.meta])
        self.assertNotEqual(len(sample[Fields.meta][MetaKeys.html_tables]), 0)
        logger.info(f"Tables: {sample[Fields.meta][MetaKeys.html_tables]}")

    def test_retain_html_tags(self):
        ds_list = [{
            'text': self.raw_html
        }]

        op = ExtractTablesFromHtmlMapper(retain_html_tags=True)
        self._run_mapper(op, ds_list)

    def test_extract_tables_include_header(self):
        ds_list = [{
            'text': self.raw_html
        }]

        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
        self._run_mapper(op, ds_list)

    def test_extract_tables_without_header(self):
        ds_list = [{
            'text': self.raw_html
        }]

        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=False)
        self._run_mapper(op, ds_list)

    def test_multiple_tables(self):
        ds_list = [{
            'text': self.raw_html + self.raw_html
        }]

        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
        self._run_mapper(op, ds_list)

    def test_large_html_content(self):
        large_html = "<html>" + "".join(
            f"<table><tr><td>Row {i}</td></tr></table>" for i in range(1000)
        ) + "</html>"

        ds_list = [{
            'text': large_html
        }]

        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
        self._run_mapper(op, ds_list)

    def test_no_tables(self):
        ds_list = [{
            'text': "<html><body>New testCase - No tables here!</body></html>"
        }]

        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        sample = dataset[0]
        self.assertIn(MetaKeys.html_tables, sample[Fields.meta])
        self.assertEqual(len(sample[Fields.meta][MetaKeys.html_tables]), 0)
        logger.info(f"Tables: {sample[Fields.meta][MetaKeys.html_tables]}")


if __name__ == '__main__':
    unittest.main()
