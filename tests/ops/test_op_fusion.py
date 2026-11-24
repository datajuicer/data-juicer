import unittest

from data_juicer.core import NestedDataset
from data_juicer.ops.base_op import Mapper, OP
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import fuse_operators, GeneralFusedOP
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class OpFusionTest(DataJuicerTestCaseBase):

    def _run_equal_config(self, original_process_list):
        dataset = NestedDataset.from_list([
            {'text': 'This is a test.'},
            {'text': 'This is a test. This is a test. This is a test.'},
            {'text': 'aaaaaaaaaaaaaaabbbbbbbbbbbbcccccccccccccc'},
            {'text': 'punc test。'}
        ])
        unfused_op = load_ops(original_process_list)
        fused_ops = fuse_operators(unfused_op)
        res1 = dataset.process(fused_ops)
        res2 = dataset.process(unfused_op)
        self.assertDatasetEqual(res1, res2)

    def _run_op_fusion(self, original_process_list, target_process_list, probe_res=None):
        ops = load_ops(original_process_list)
        ops = fuse_operators(ops, probe_res)
        new_process_list = [op._op_cfg for op in ops]
        self.assertEqual(new_process_list, target_process_list)

    def test_regular_config(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process)
        self._run_equal_config(original_process)

    def test_only_mapper(self):
        original_process = [{
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_only_deduplicator(self):

        original_process = [{
            'document_deduplicator': {
                'ignore_non_character': True,
                'lowercase': True,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }, {
            'document_minhash_deduplicator': {
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6,
                'num_permutations': 256,
                'jaccard_threshold': 0.7
            }
        }]
        target_process = [{
            'document_deduplicator': {
                'ignore_non_character': True,
                'lowercase': True,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }, {
            'document_minhash_deduplicator': {
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6,
                'num_permutations': 256,
                'jaccard_threshold': 0.7
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_non_fusible_filters(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'alphanumeric_filter': {
                'min_ratio': 0.25,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'alphanumeric_filter': {
                'min_ratio': 0.25,
                'text_key': 'text'
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_not_enough_fusible_ops_to_fuse(self):
        # still apply reordering:
        # - ordinary ops
        # - ops with InterVars.lines
        # - ops with InterVars.words
        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_multiple_groups(self):

        original_process = [{
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(stopwords_filter,flagged_words_filter)': [{
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }, {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }]
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process)

    def test_only_fusible_ops(self):

        original_process = [{
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
            [
                {
                    'words_num_filter': {
                        'lang': 'en',
                        'max_num': 100000,
                        'min_num': 20,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'word_repetition_filter': {
                        'lang': 'en',
                        'max_ratio': 0.19,
                        'min_ratio': 0.0,
                        'rep_len': 5,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'perplexity_filter': {
                        'lang': 'en',
                        'max_ppl': 1500,
                        'text_key': 'text'
                    }
                }
            ]
        }]
        self._run_op_fusion(original_process, target_process)

    def test_different_intermediate_vars(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'maximum_line_length_filter': {
                'min_len': 20,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(average_line_length_filter,maximum_line_length_filter)':  # noqa: E501
                [
                    {
                        'average_line_length_filter': {
                            'min_len': 10,
                            'text_key': 'text',
                        }
                    },
                    {
                        'maximum_line_length_filter': {
                            'min_len': 20,
                            'text_key': 'text',
                        }
                    }
                ]
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process)

    def test_regular_config_with_probe_res(self):
        probed_speeds = [
            # single filter
            {'speed': 100},

            # mappers
            {'speed': 2},
            {'speed': 1},
            {'speed': 4},
            {'speed': 5},
            {'speed': 3},

            # filter groups
            # fused OPs: ~2.56
            # single OP 1: 1 (slowest)
            # single OP 2: 3 (fastest)
            {'speed': 15},  # fusible
            {'speed': 1},
            {'speed': 14},  # fusible
            {'speed': 3},
            {'speed': 13},  # fusible
            {'speed': 12},  # fusible
            {'speed': 11},  # fusible

            # deduplicator
            {'speed': 0.1},
        ]

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process, probed_speeds)

    def test_not_enough_fusible_ops_to_fuse_with_probe_res(self):
        # still apply reordering:
        # - ordinary ops
        # - ops with InterVars.lines
        # - ops with InterVars.words
        probe_res_list = [
            {'speed': 3},
            {'speed': 1},
            {'speed': 4},
            {'speed': 2},
        ]

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }]
        self._run_op_fusion(original_process, target_process, probe_res_list)

    def test_multiple_groups_with_probe_res(self):
        probe_res_list = [
            # group 1
            # fused filter will be put before the single filter
            {'speed': 10},
            {'speed': 10},
            {'speed': 1},

            # mappers
            {'speed': 4},
            {'speed': 2},
            {'speed': 5},
            {'speed': 3},
            {'speed': 1},

            # group 2
            # fused filter will be put after those two single filters
            {'speed': 1},  # fusible
            {'speed': 8},
            {'speed': 1},  # fusible
            {'speed': 10},
            {'speed': 1},  # fusible

            # deduplicator
            {'speed': 1},
        ]

        original_process = [{
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'OpFusion:(stopwords_filter,flagged_words_filter)': [{
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }, {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }]
            },
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process, probe_res_list)

    def test_only_fusible_ops_with_probe_res(self):
        probe_res_list = [
            {'speed': 1},
            {'speed': 1},
            {'speed': 1},
            {'speed': 1},
            {'speed': 1},
        ]

        original_process = [{
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
            [
                {
                    'words_num_filter': {
                        'lang': 'en',
                        'max_num': 100000,
                        'min_num': 20,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'word_repetition_filter': {
                        'lang': 'en',
                        'max_ratio': 0.19,
                        'min_ratio': 0.0,
                        'rep_len': 5,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'perplexity_filter': {
                        'lang': 'en',
                        'max_ppl': 1500,
                        'text_key': 'text'
                    }
                }
            ]
        }]
        self._run_op_fusion(original_process, target_process, probe_res_list)

    def test_different_intermediate_vars_with_probe_res(self):
        probe_res_list = [
            # single filter
            {'speed': 1},

            # mappers
            {'speed': 5},
            {'speed': 3},
            {'speed': 1},
            {'speed': 2},
            {'speed': 4},

            # filter group
            # single 1: 1 (2)
            # single 2: 0.5 (3)
            # group 1: 0.04 (4)
            # group 2: 1.5 (1)
            {'speed': 0.1},  # group 1
            {'speed': 1},
            {'speed': 3},  # group 2
            {'speed': 0.2},  # group 1
            {'speed': 0.5},
            {'speed': 0.3},  # group 1
            {'speed': 0.4},  # group 1
            {'speed': 3},  # group 2
            {'speed': 0.5},  # group 1

            # deduplicator
            {'speed': 1},
        ]

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'maximum_line_length_filter': {
                'min_len': 20,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(average_line_length_filter,maximum_line_length_filter)':  # noqa: E501
                [
                    {
                        'average_line_length_filter': {
                            'min_len': 10,
                            'text_key': 'text',
                        }
                    },
                    {
                        'maximum_line_length_filter': {
                            'min_len': 20,
                            'text_key': 'text',
                        }
                    }
                ]
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process, probe_res_list)


class GeneralFusedOPTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.raw_data = [
            {'text': 'This is a test.'},
            {'text': 'This is a test. This is a test. This is a test.'},
            {'text': 'aaaaaaaaaaaaaaabbbbbbbbbbbbcccccccccccccc'},
            {'text': 'punc test。'}
        ]

    def _get_fresh_dataset(self):
        """Get a fresh dataset instance to avoid state pollution between tests."""
        return NestedDataset.from_list(self.raw_data)

    def _run_equal_config(self, fused_process, unfused_process):
        fused_op = load_ops(fused_process)
        self.assertEqual(len(fused_op), 1)
        fused_op = fused_op[0]
        unfused_op = load_ops(unfused_process)
        self.assertIsInstance(fused_op, GeneralFusedOP)
        self.assertEqual(len(fused_op.fused_ops), len(unfused_process))
        
        # Use fresh datasets for each operation to avoid state pollution
        dataset1 = self._get_fresh_dataset()
        dataset2 = self._get_fresh_dataset()
        res1 = dataset1.process(fused_op)
        res2 = dataset2.process(unfused_op)
        self.assertDatasetEqual(res1, res2)

    def test_regular_config(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }]
        fused_process = [{
            'general_fused_op': {
                'batch_size': 2,
                'fused_op_list': original_process,
            }
        }]
        self._run_equal_config(fused_process, original_process)

    def test_border_cases(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }]
        empty_fused_process = [{
            'general_fused_op': {
                'batch_size': 2,
                'fused_op_list': None,
            }
        }]
        fused_process = [{
            'general_fused_op': {
                'batch_size': 2,
                'fused_op_list': original_process,
            }
        }]
        # empty fused process
        fused_op = load_ops(empty_fused_process)[0]
        self.assertEqual(len(fused_op.fused_ops), 0)
        dataset = self._get_fresh_dataset()
        res = fused_op.run(dataset)
        self.assertDatasetEqual(res, dataset)
        # unsupported fused op
        dataset2 = self._get_fresh_dataset()
        with self.assertRaises(NotImplementedError):
            fused_op = load_ops([{
                'general_fused_op': {
                    'batch_size': 2,
                    'fused_op_list': [{
                        'document_deduplicator': {}
                    }],
                }
            }])[0]
            fused_op.process_batched(dataset2.to_dict())


class FusedFilterFingerprintTest(DataJuicerTestCaseBase):
    """Tests that FusedFilter fingerprints exclude child OP work_dirs."""

    def test_fused_filter_stable_across_work_dirs(self):
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.filter.words_num_filter import WordsNumFilter
        from data_juicer.ops.op_fusion import FusedFilter
        from data_juicer.utils.fingerprint_utils import Hasher

        f1a = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/a')
        f2a = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/a')
        fused_a = FusedFilter('fused', [f1a, f2a])

        f1b = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/b')
        f2b = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/b')
        fused_b = FusedFilter('fused', [f1b, f2b])

        self.assertEqual(Hasher.hash(fused_a), Hasher.hash(fused_b))

    def test_fused_filter_differs_when_child_params_change(self):
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.filter.words_num_filter import WordsNumFilter
        from data_juicer.ops.op_fusion import FusedFilter
        from data_juicer.utils.fingerprint_utils import Hasher

        f1a = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/a')
        f2a = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/a')
        fused_a = FusedFilter('fused', [f1a, f2a])

        f1b = TextLengthFilter(min_len=50, max_len=10000, work_dir='/tmp/a')
        f2b = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/a')
        fused_b = FusedFilter('fused', [f1b, f2b])

        self.assertNotEqual(Hasher.hash(fused_a), Hasher.hash(fused_b))


class _MockUpperCaseMapper(Mapper):
    """Mapper that uppercases text and returns a NEW dict."""
    _batched_op = True

    def __init__(self, text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self._name = 'mock_upper_case_mapper'

    def process_batched(self, samples, **kwargs):
        new_samples = samples.copy()
        new_samples[self.text_key] = [t.upper() for t in samples[self.text_key]]
        return new_samples


class _MockSuffixMapper(Mapper):
    """Mapper that appends a suffix and returns a NEW dict."""
    _batched_op = True

    def __init__(self, suffix='_DONE', text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self.suffix = suffix
        self._name = 'mock_suffix_mapper'

    def process_batched(self, samples, **kwargs):
        new_samples = samples.copy()
        new_samples[self.text_key] = [t + self.suffix for t in samples[self.text_key]]
        return new_samples


class _MockInPlaceMapper(Mapper):
    """Mapper that mutates in-place (masks the new-dict bug)."""
    _batched_op = True

    def __init__(self, text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self._name = 'mock_inplace_mapper'

    def process_batched(self, samples, **kwargs):
        samples[self.text_key] = [t.lower() for t in samples[self.text_key]]
        return samples


class GeneralFusedOPMapperBugTest(DataJuicerTestCaseBase):
    """Regression: mapper results must chain correctly in GeneralFusedOP."""

    def _make_fused_op(self, ops):
        fused = GeneralFusedOP.__new__(GeneralFusedOP)
        fused._name = 'GeneralFusedOP:test'
        fused.fused_ops = ops
        fused.accelerator = 'cpu'
        fused.batch_size = 10
        fused.num_proc = 1
        return fused

    def test_two_new_dict_mappers_results_chained(self):
        fused_op = self._make_fused_op([
            _MockUpperCaseMapper(), _MockSuffixMapper(suffix='_DONE')])
        samples = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }
        result = fused_op.process_batched(samples)
        self.assertEqual(result['text'], ['HELLO_DONE', 'WORLD_DONE'])

    def test_single_mapper_result_returned(self):
        fused_op = self._make_fused_op([_MockUpperCaseMapper()])
        samples = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }
        result = fused_op.process_batched(samples)
        self.assertEqual(result['text'], ['HELLO', 'WORLD'])

    def test_inplace_then_newdict_mapper(self):
        fused_op = self._make_fused_op([
            _MockInPlaceMapper(), _MockSuffixMapper(suffix='_END')])
        samples = {
            'text': ['HELLO', 'WORLD'],
            Fields.stats: [{}, {}],
        }
        result = fused_op.process_batched(samples)
        self.assertEqual(result['text'], ['hello_END', 'world_END'])


class FusedContextKeyIsolationTest(DataJuicerTestCaseBase):
    """Fusion must not change any statistic.

    Ops in a fused group share one `Fields.context` dict. The cache key for
    each intermediate variable therefore has to identify everything the
    cached value depends on -- the source column (`text_key`) and the
    tokenizer (`model_key`). If two ops in the same group disagree on either
    one but build the same key, the second op silently reads the first op's
    value and reports a statistic for a column it never looked at.
    """

    def _fused_vs_unfused(self, process_list, row):
        """Compute stats and the keep/drop decision, fused and unfused."""

        def make_batch():
            batch = {key: [value] for key, value in row.items()}
            batch[Fields.stats] = [{}]
            batch[Fields.context] = [{}]
            return batch

        fused_ops = fuse_operators(load_ops(process_list))
        self.assertEqual(len(fused_ops), 1,
                         'ops under test are expected to fuse into one op')

        fused_batch = make_batch()
        fused_ops[0].compute_stats_batched(fused_batch)
        fused_stats = fused_batch[Fields.stats][0]
        fused_keep = bool(list(fused_ops[0].process_batched(fused_batch))[0])

        # Every unfused op gets its own batch and therefore its own context,
        # so it cannot read another op's intermediate value. That makes the
        # unfused run the reference behaviour the fused run has to match.
        unfused_stats, unfused_keep = {}, True
        for op in load_ops(process_list):
            batch = make_batch()
            op.compute_stats_batched(batch)
            unfused_stats.update(batch[Fields.stats][0])
            unfused_keep &= bool(list(op.process_batched(batch))[0])

        return fused_stats, unfused_stats, fused_keep, unfused_keep

    def _assert_fusion_transparent(self, process_list, row, expected_stats,
                                   expected_keep):
        fused_stats, unfused_stats, fused_keep, unfused_keep = \
            self._fused_vs_unfused(process_list, row)

        self.assertEqual(sorted(unfused_stats), sorted(expected_stats))
        self.assertEqual(sorted(fused_stats), sorted(expected_stats))
        for key, expected in expected_stats.items():
            self.assertAlmostEqual(unfused_stats[key], expected, places=6)
            self.assertAlmostEqual(fused_stats[key], expected, places=6)
        self.assertEqual(unfused_keep, expected_keep)
        self.assertEqual(fused_keep, expected_keep)

    def test_inter_words_ops_differing_in_text_key(self):
        process_list = [{
            'words_num_filter': {
                'lang': 'en',
                'min_num': 1,
                'max_num': 10000,
                'tokenization': False,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'rep_len': 2,
                'min_ratio': 0.0,
                'max_ratio': 1.0,
                'tokenization': False,
                'text_key': 'translation'
            }
        }]
        # 'text' is all-identical 2-grams, 'translation' has no repetition, so
        # reading the wrong column flips word_rep_ratio from 0.0 to 1.0, which
        # also pushes the sample past max_ratio and drops it.
        row = {
            'text': 'ha ha ha ha ha ha ha ha',
            'translation': 'alpha beta gamma delta epsilon zeta eta theta'
        }
        self._assert_fusion_transparent(process_list,
                                        row,
                                        expected_stats={
                                            'num_words': 8,
                                            'word_rep_ratio': 0.0
                                        },
                                        expected_keep=True)

    def test_inter_lines_ops_differing_in_text_key(self):
        process_list = [{
            'average_line_length_filter': {
                'min_len': 1,
                'max_len': 100000,
                'text_key': 'text'
            }
        }, {
            'maximum_line_length_filter': {
                'min_len': 10,
                'max_len': 100000,
                'text_key': 'translation'
            }
        }]
        # 'text' splits into three 1-char lines, 'translation' is one long
        # line, so reading the wrong column reports max_line_length 1 not 30,
        # which falls below min_len and drops the sample.
        row = {'text': 'a\nb\nc', 'translation': 'x' * 30}
        self._assert_fusion_transparent(process_list,
                                        row,
                                        expected_stats={
                                            'avg_line_length': 5 / 3,
                                            'max_line_length': 30
                                        },
                                        expected_keep=True)

    def test_inter_words_ops_differing_in_tokenizer(self):
        # Tokenized word_repetition_filter followed by untokenized
        # stopwords_filter on the same field. The tokenized version uses
        # SentencePiece which produces different word tokens than simple
        # whitespace splitting. With a broken cache key (missing model_key),
        # the second op would read the first op's tokenized words and report
        # the wrong stopwords_ratio.
        process_list = [{
            'word_repetition_filter': {
                'lang': 'en',
                'rep_len': 2,
                'tokenization': True,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'tokenization': False,
                'text_key': 'text'
            }
        }]
        # 3 of the 9 whitespace-split words are English stopwords ('the',
        # 'over', 'the'), so stopwords_ratio is 1/3 and the sample is kept.
        # SentencePiece splits the same sentence into more, partly sub-word
        # tokens that the stopwords list does not match; reading those from the
        # shared context yields 0.0, below min_ratio, and drops the sample.
        row = {'text': 'the quick brown fox jumps over the lazy dog'}
        self._assert_fusion_transparent(process_list,
                                        row,
                                        expected_stats={
                                            'word_rep_ratio': 0.0,
                                            'stopwords_ratio': 1 / 3
                                        },
                                        expected_keep=True)


if __name__ == '__main__':
    unittest.main()
