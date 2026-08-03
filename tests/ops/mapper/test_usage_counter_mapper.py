# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.usage_counter_mapper import UsageCounterMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestUsageCounterMapper(DataJuicerTestCaseBase):
    def test_dedupes_identical_top_level_and_choice_usage(self):
        """Same usage in response_usage and choices[0] must not double-count."""
        op = UsageCounterMapper(
            choices_key="response_choices",
            usage_key="response_usage",
            dedupe_identical_usage=True,
        )
        sample = {
            "response_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "response_choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                }
            ],
            Fields.meta: {},
        }
        out = op.process_single(sample)
        meta = out[Fields.meta]
        self.assertEqual(meta[MetaKeys.prompt_tokens], 100)
        self.assertEqual(meta[MetaKeys.completion_tokens], 50)
        self.assertEqual(meta[MetaKeys.total_tokens], 150)

    def test_sums_distinct_usage_blocks(self):
        op = UsageCounterMapper(
            choices_key="choices",
            usage_key="usage",
            dedupe_identical_usage=True,
        )
        sample = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [
                {
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                }
            ],
            Fields.meta: {},
        }
        out = op.process_single(sample)
        meta = out[Fields.meta]
        self.assertEqual(meta[MetaKeys.prompt_tokens], 30)
        self.assertEqual(meta[MetaKeys.completion_tokens], 13)

    def test_dedupe_off_preserves_double_count(self):
        op = UsageCounterMapper(
            choices_key="response_choices",
            usage_key="response_usage",
            dedupe_identical_usage=False,
        )
        u = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        sample = {
            "response_usage": dict(u),
            "response_choices": [{"usage": dict(u)}],
            Fields.meta: {},
        }
        out = op.process_single(sample)
        meta = out[Fields.meta]
        self.assertEqual(meta[MetaKeys.prompt_tokens], 200)
        self.assertEqual(meta[MetaKeys.completion_tokens], 100)


if __name__ == "__main__":
    unittest.main()
