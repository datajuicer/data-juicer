import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset

from data_juicer.config import init_configs
from data_juicer.ops.load import load_ops
from data_juicer.ops.selector.random_selector import RandomSelector
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RandomSelectorTest(DataJuicerTestCaseBase):

    def _seed_dataset(self):
        return Dataset.from_dict({"id": list(range(100)), "text": [f"sample {i}" for i in range(100)]})

    def test_default_preserves_previous_seed_and_order(self):
        dataset = self._seed_dataset()
        expected = dataset.shuffle(seed=42).select(range(17)).to_list()
        self.assertEqual(RandomSelector(select_num=17).process(dataset).to_list(), expected)

    def test_custom_seed_is_reproducible(self):
        dataset = self._seed_dataset()
        for seed in (0, 7, 42):
            for view_name, view in (("original", dataset), ("reversed", dataset.select(range(99, -1, -1)))):
                with self.subTest(seed=seed, view=view_name):
                    expected = view.shuffle(seed=seed).select(range(17)).to_list()
                    op = RandomSelector(select_num=17, seed=seed)
                    self.assertEqual(op.process(view).to_list(), expected)
                    self.assertEqual(op.process(view).to_list(), expected)

    def test_recipe_seed_changes_selected_rows(self):
        dataset = self._seed_dataset()
        recipe = yaml.safe_load("process:\n  - random_selector:\n      select_num: 17\n      seed: 7\n")
        selected = load_ops(recipe["process"])[0].process(dataset).to_list()
        self.assertEqual(selected, dataset.shuffle(seed=7).select(range(17)).to_list())
        self.assertNotEqual(selected, RandomSelector(select_num=17).process(dataset).to_list())

    def test_cli_overrides_recipe_seed(self):
        dataset = self._seed_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "input.jsonl"
            source.write_text('{"text": "sample"}\n', encoding="utf-8")
            config = root / "recipe.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "dataset_path": source.as_posix(),
                        "export_path": (root / "output.jsonl").as_posix(),
                        "process": [{"random_selector": {"select_num": 17, "seed": 42}}],
                    }
                ),
                encoding="utf-8",
            )
            cfg = init_configs(args=["--config", str(config), "--random_selector.seed", "7"], load_configs_only=True)
            selected = load_ops(cfg.process)[0].process(dataset).to_list()
        self.assertEqual(selected, dataset.shuffle(seed=7).select(range(17)).to_list())

    def test_ratio_and_count_use_custom_seed(self):
        dataset = self._seed_dataset()
        for config, count in (({"select_ratio": 0.25}, 25), ({"select_ratio": 0.25, "select_num": 7}, 7)):
            with self.subTest(**config):
                expected = dataset.shuffle(seed=7).select(range(count)).to_list()
                self.assertEqual(RandomSelector(seed=7, **config).process(dataset).to_list(), expected)

    def test_seeded_oversampling(self):
        dataset = self._seed_dataset()
        expected = dataset.shuffle(seed=7).select(list(range(100)) + list(range(25))).to_list()
        self.assertEqual(RandomSelector(select_num=125, seed=7).process(dataset).to_list(), expected)

    def test_seed_keeps_noop_and_empty_selection_behavior(self):
        dataset = self._seed_dataset()
        for config in ({}, {"select_num": 100}, {"select_ratio": 0.0}):
            with self.subTest(**config):
                self.assertEqual(
                    RandomSelector(seed=7, **config).process(dataset).to_list(),
                    RandomSelector(**config).process(dataset).to_list(),
                )

    def test_custom_seed_preserves_numpy_global_rng(self):
        dataset = self._seed_dataset()
        before = np.random.get_state()
        RandomSelector(select_num=17, seed=7).process(dataset)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_zero_selection_preserves_schema(self):
        dataset = Dataset.from_dict({'text': ['first', 'second'], 'score': [1, 2]})
        for kwargs in [
            {'select_ratio': 0.0},
            {'select_ratio': 0.0, 'select_num': 1},
            {'select_ratio': 0.1},
            {'select_ratio': 0.1, 'select_num': 1},
        ]:
            with self.subTest(**kwargs):
                result = RandomSelector(**kwargs).process(dataset)
                self.assertEqual(len(result), 0)
                self.assertEqual(result.features, dataset.features)

    def test_unspecified_ratio_uses_sample_number(self):
        dataset = Dataset.from_dict({'text': ['first', 'second', 'third']})
        result = RandomSelector(select_num=2).process(dataset)
        self.assertEqual(len(result), 2)

    def test_unspecified_selection_returns_original_dataset(self):
        dataset = Dataset.from_dict({'text': ['first', 'second']})
        self.assertEqual(RandomSelector().process(dataset).to_list(), dataset.to_list())

    def test_empty_dataset_returns_original_dataset(self):
        dataset = Dataset.from_dict({'text': [], 'score': []})
        result = RandomSelector(select_ratio=0.0).process(dataset)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.features, dataset.features)

    def test_single_row_returns_original_dataset(self):
        dataset = Dataset.from_dict({'text': ['first']})
        self.assertEqual(RandomSelector(select_ratio=0.0).process(dataset).to_list(), dataset.to_list())

    def _run_random_selector(self, dataset: Dataset, target_num, op):
        dataset = op.process(dataset)
        res_list = dataset.to_list()
        self.assertEqual(len(res_list), target_num)

    def test_ratio_select(self):
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
        }, {
            'text': '他的英文名字叫Harry Potter',
            'count': 88,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 551
                    },
                    'count': 78
                }
            }
        }, {
            'text': '这是一个测试',
            'count': None,
            'meta': {
                'suffix': '.py',
                'key1': {
                    'key2': {
                        'count': 89
                    },
                    'count': 3
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 33
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }]
        tgt_num = 2
        dataset = Dataset.from_list(ds_list)
        op = RandomSelector(select_ratio=0.2,
                             select_num=5)
        self._run_random_selector(dataset, tgt_num, op)

    def test_num_select(self):
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
        }, {
            'text': '他的英文名字叫Harry Potter',
            'count': 88,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 551
                    },
                    'count': 78
                }
            }
        }, {
            'text': '这是一个测试',
            'count': None,
            'meta': {
                'suffix': '.py',
                'key1': {
                    'key2': {
                        'count': 89
                    },
                    'count': 3
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 33
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }]
        tgt_num = 4
        dataset = Dataset.from_list(ds_list)
        op = RandomSelector(select_ratio=0.5,
                             select_num=4)
        self._run_random_selector(dataset, tgt_num, op)


if __name__ == '__main__':
    unittest.main()
