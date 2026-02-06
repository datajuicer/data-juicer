import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.format.formatter import load_dataset, unify_format
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class UnifyFormatTest(DataJuicerTestCaseBase):

    def run_test(self, sample, args=None):
        if args is None:
            args = {}
        ds = Dataset.from_list(sample['source'])
        ds = unify_format(ds, **args)
        self.assertEqual(ds.to_list(), sample['target'])

    def test_text_key(self):
        samples = [
            {
                'source': [{
                    'text': 'This is a test text',
                    'outer_key': 1,
                }],
                'target': [{
                    'text': 'This is a test text',
                    'outer_key': 1,
                }]
            },
            {
                'source': [{
                    'content': 'This is a test text',
                    'outer_key': 1,
                }],
                'target': [{
                    'content': 'This is a test text',
                    'outer_key': 1,
                }]
            },
            {
                'source': [{
                    'input': 'This is a test text, input part',
                    'instruction': 'This is a test text, instruction part',
                    'outer_key': 1,
                }],
                'target': [{
                    'input': 'This is a test text, input part',
                    'instruction': 'This is a test text, instruction part',
                    'outer_key': 1,
                }]
            },
        ]
        self.run_test(samples[0])
        self.run_test(samples[1], args={'text_keys': ['content']})
        self.run_test(samples[2], args={'text_keys': ['input', 'instruction']})

    def test_empty_text(self):
        # filter out samples containing None field, but '' is OK
        samples = [
            {
                'source': [{
                    'text': '',
                    'outer_key': 1,
                }],
                'target': [{
                    'text': '',
                    'outer_key': 1,
                }],
            },
            {
                'source': [{
                    'text': None,
                    'outer_key': 1,
                }],
                'target': [],
            },
        ]
        for sample in samples:
            self.run_test(sample)

    def test_no_extra_fields(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'lang': 'en'
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'lang': 'en'
                },
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
            }],
            'target': [{
                'text': 'This is a test text.',
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_no_extra_fields_except_meta(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
                Fields.stats: {
                    'lang': 'en'
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
                Fields.stats: {
                    'lang': 'en'
                },
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_invalid_stats(self):
        # non-dict stats will be unified into stats
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'stats': 'nice',
            }],
            'target': [{
                'text': 'This is a test text.',
                'stats': 'nice'
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'version': 1
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'version': 1
                },
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_outer_fields(self):
        samples = [
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice',
                    },
                    'outer_field': 'value',
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    'stats': 'en'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    'stats': 'en'
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    'stats': 'en'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    'stats': 'en'
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    'stats': 'en',
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    'stats': 'en'
                }],
            },
        ]
        for sample in samples:
            self.run_test(sample)

    def test_recursive_meta(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'outer_field': {
                    'rec1': {
                        'rec2': 'value'
                    }
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                'outer_field': {
                    'rec1': {
                        'rec2': 'value'
                    }
                },
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_hetero_meta(self):
        cur_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'data', 'structured')
        file_path = os.path.join(cur_dir, 'demo-dataset.jsonl')
        ds = load_dataset('json', data_files=file_path)
        ds = unify_format(ds)
        # import datetime

        # the 'None' fields are missing fields after merging
        # sample = [{
        #     'text': "Today is Sunday and it's a happy day!",
        #     'meta': {
        #         'src': 'Arxiv',
        #         'date': datetime.datetime(2023, 4, 27, 0, 0),
        #         'version': '1.0',
        #         'author': None
        #     }
        # }, {
        #     'text': 'Do you need a cup of coffee?',
        #     'meta': {
        #         'src': 'code',
        #         'date': None,
        #         'version': None,
        #         'author': 'xxx'
        #     }
        # }]
        # test nested and missing field for the following cases:
        # 1. first row, then column
        unified_sample_first = ds[0]
        unified_sample_second = ds[1]
        self.assertEqual(unified_sample_first['meta.src'], 'Arxiv')
        self.assertEqual(unified_sample_first['meta.author'], None)
        self.assertEqual(unified_sample_second['meta.date'], None)
        # 2. first column, then row
        self.assertEqual(ds['meta.src'][0], 'Arxiv')
        self.assertEqual(ds['meta.src'][1], 'code')
        self.assertEqual(ds['meta.author'][0], None)
        self.assertEqual(ds['meta.date'][1], None)
        # 3. first partial rows, then column, final row
        unified_ds_first = ds.select([0])
        unified_ds_second = ds.select([1])
        self.assertEqual(unified_ds_first['meta.src'][0], 'Arxiv')
        self.assertEqual(unified_ds_first['meta.author'][0], None)
        self.assertEqual(unified_ds_second['meta.date'][0], None)

    def test_empty_meta(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {},
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_empty_stats(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {},
                Fields.stats: {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {},
                Fields.stats: {},
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_empty_outer_fields(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {},
                'out_field': {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {},
                'out_field': {},
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
                'out_field': {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'out_field': {},
            }],
        }]
        for sample in samples:
            self.run_test(sample)


if __name__ == '__main__':
    unittest.main()
