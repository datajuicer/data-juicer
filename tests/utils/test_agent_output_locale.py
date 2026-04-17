# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.utils.agent_output_locale import (
    agent_skill_insight_system_prompt,
    dialog_score_json_instruction,
    llm_filter_free_text_language_appendix,
    normalize_preferred_output_lang,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestAgentOutputLocale(DataJuicerTestCaseBase):
    def test_normalize(self):
        self.assertEqual(normalize_preferred_output_lang("zh-CN"), "zh")
        self.assertEqual(normalize_preferred_output_lang("EN"), "en")
        self.assertEqual(normalize_preferred_output_lang(None), "en")

    def test_json_instruction_zh_has_score(self):
        s = dialog_score_json_instruction("zh")
        self.assertIn("score", s)
        self.assertIn("reason", s)

    def test_filter_appendix_empty_when_none(self):
        self.assertEqual(llm_filter_free_text_language_appendix(None), "")

    def test_skill_insight_prompt_zh_concrete_length(self):
        s = agent_skill_insight_system_prompt("zh")
        self.assertIn("8～12", s)
        self.assertIn("禁止", s)

    def test_skill_insight_prompt_en_concrete_length(self):
        s = agent_skill_insight_system_prompt("en")
        self.assertIn("4–8 words", s)
        self.assertIn("Forbidden", s)


if __name__ == "__main__":
    unittest.main()
