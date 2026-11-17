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

    def test_without_header_keeps_rows_with_row_headers(self):
        html = ("<table>"
                "<tr><th>Country</th><th>Capital</th></tr>"
                "<tr><th>France</th><td>Paris</td></tr>"
                "<tr><th>Japan</th><td>Tokyo</td></tr>"
                "</table>")
        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=False)
        sample = op.process_single({'text': html, Fields.meta: {}})
        self.assertEqual(sample[Fields.meta][MetaKeys.html_tables],
                         [[['France', 'Paris'], ['Japan', 'Tokyo']]])

        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=True)
        sample = op.process_single({'text': html, Fields.meta: {}})
        self.assertEqual(sample[Fields.meta][MetaKeys.html_tables],
                         [[['Country', 'Capital'], ['France', 'Paris'], ['Japan', 'Tokyo']]])

    def test_without_header_skips_header_row_with_empty_corner_cell(self):
        html = ("<table>"
                "<tr><td></td><th>2020</th><th>2021</th></tr>"
                "<tr><th>Sales</th><td>1</td><td>2</td></tr>"
                "</table>")
        op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=False)
        sample = op.process_single({'text': html, Fields.meta: {}})
        self.assertEqual(sample[Fields.meta][MetaKeys.html_tables], [[['Sales', '1', '2']]])

    def test_without_header_skips_thead_rows_and_keeps_row_headers(self):
        html = ("<table>"
                "<thead><tr><td>Metric</td><th>2024</th></tr></thead>"
                "<tbody><tr><th>Revenue</th><td>10</td></tr></tbody>"
                "</table>")
        for include_header, expected in [(False, [[['Revenue', '10']]]),
                                         (True, [[['Metric', '2024'], ['Revenue', '10']]])]:
            op = ExtractTablesFromHtmlMapper(retain_html_tags=False, include_header=include_header)
            dataset = op.run(Dataset.from_list([{'text': html}]))
            self.assertEqual(dataset[0][Fields.meta][MetaKeys.html_tables], expected)

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
