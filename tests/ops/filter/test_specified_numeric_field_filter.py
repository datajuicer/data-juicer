import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.specified_numeric_field_filter import \
    SpecifiedNumericFieldFilter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields


class SpecifiedNumericFieldFilterTest(DataJuicerTestCaseBase):

    def _run_specified_numeric_field_filter(self, dataset: Dataset,
                                            target_list, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf',
                'star': 50
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'meta': {
                'suffix': '.docx',
                'star': 6
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'meta': {
                'suffix': '.txt',
                'star': 100
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'meta': {
                'suffix': '.html',
                'star': 12.51
            }
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            'meta': {
                'suffix': None
            }
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf',
                'star': 50
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'meta': {
                'suffix': '.html',
                'star': 12.51
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = SpecifiedNumericFieldFilter(field_key='meta.star',
                                         min_value=10,
                                         max_value=70)
        self._run_specified_numeric_field_filter(dataset, tgt_list, op)

    def test_multi_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 34
                    },
                    'count': 5
                }
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'count': 16,
            'meta': {
                'suffix': '.docx',
                'key1': {
                    'key2': {
                        'count': 243
                    },
                    'count': 63
                }
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'count': 162,
            'meta': {
                'suffix': '.txt',
                'key1': {
                    'key2': {
                        'count': None
                    },
                    'count': 23
                }
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'count': None,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 34
                    },
                    'count': 5
                }
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'count': None,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = SpecifiedNumericFieldFilter(field_key='meta.key1.key2.count',
                                         min_value=10,
                                         max_value=70)
        self._run_specified_numeric_field_filter(dataset, tgt_list, op)

    def test_str_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf',
                'star': '36'
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'meta': {
                'suffix': '.docx',
                'star': '13.5'
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'meta': {
                'suffix': '.txt',
                'star': 'asdkc'
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'meta': {
                'suffix': '.html',
                'star': '441'
            }
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            'meta': {
                'suffix': None
            }
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf',
                'star': '36'
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'meta': {
                'suffix': '.docx',
                'star': '13.5'
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = SpecifiedNumericFieldFilter(field_key='meta.star',
                                         min_value=10,
                                         max_value=70)
        self._run_specified_numeric_field_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
