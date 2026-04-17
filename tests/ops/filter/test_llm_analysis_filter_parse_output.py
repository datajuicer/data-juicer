# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for LLMAnalysisFilter.parse_output (no API)."""

import unittest

from data_juicer.ops.filter.llm_analysis_filter import LLMAnalysisFilter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LLMAnalysisFilterParseOutputTest(DataJuicerTestCaseBase):
    def setUp(self):
        self.op = object.__new__(LLMAnalysisFilter)
        self.op.dim_required_keys = list(LLMAnalysisFilter.DEFAULT_DIM_REQUIRED_KEYS)

    def test_recommendation_scalar_string_becomes_list(self):
        raw = """{
  "dimension_scores": {"clarity": 4, "relevance": 5, "usefulness": 3, "fluency": 4},
  "flags": [],
  "rationale": "ok",
  "recommendation": "review"
}"""
        score, record, _tags = self.op.parse_output(raw)
        self.assertIsNotNone(record)
        self.assertEqual(record["recommendation"], ["review"])
        self.assertIsInstance(score, float)

    def test_recommendation_list_preserved(self):
        raw = """{
  "dimension_scores": {"clarity": 4, "relevance": 5, "usefulness": 3, "fluency": 4},
  "flags": [],
  "rationale": "ok",
  "recommendation": ["keep", "review"]
}"""
        _score, record, _tags = self.op.parse_output(raw)
        self.assertEqual(record["recommendation"], ["keep", "review"])

    def test_recommendation_missing_defaults_to_empty_list(self):
        raw = """{
  "dimension_scores": {"clarity": 4, "relevance": 5, "usefulness": 3, "fluency": 4},
  "flags": [],
  "rationale": "ok"
}"""
        _score, record, _tags = self.op.parse_output(raw)
        self.assertEqual(record["recommendation"], [])


if __name__ == "__main__":
    unittest.main()
