# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.utils.agent_output_locale import (
    agent_insight_system_prompt,
    agent_skill_insight_system_prompt,
    dialog_detection_output_language_note,
    dialog_score_json_instruction,
    llm_filter_free_text_language_appendix,
    normalize_preferred_output_lang,
    rubric_reason_language_clause,
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


    def test_normalize_zh_variants(self):
        """All Chinese locale variants should normalize to 'zh'."""
        for val in ("zh", "zh-CN", "zh-TW", "zh_cn", "ZH"):
            self.assertEqual(normalize_preferred_output_lang(val), "zh",
                             f"Failed for {val!r}")

    def test_normalize_en_variants(self):
        for val in ("en", "EN", "english", "eng", "English"):
            self.assertEqual(normalize_preferred_output_lang(val), "en",
                             f"Failed for {val!r}")

    def test_normalize_empty_string(self):
        self.assertEqual(normalize_preferred_output_lang(""), "en")

    def test_normalize_unknown_defaults_en(self):
        self.assertEqual(normalize_preferred_output_lang("fr"), "en")
        self.assertEqual(normalize_preferred_output_lang("ja"), "en")

    def test_json_instruction_en(self):
        s = dialog_score_json_instruction("en")
        self.assertIn("score", s)
        self.assertIn("JSON", s)

    def test_rubric_reason_zh(self):
        s = rubric_reason_language_clause("zh")
        self.assertIn("简体中文", s)

    def test_rubric_reason_en(self):
        s = rubric_reason_language_clause("en")
        self.assertIn("English", s)

    def test_filter_appendix_zh(self):
        s = llm_filter_free_text_language_appendix("zh")
        self.assertIn("简体中文", s)
        self.assertGreater(len(s), 0)

    def test_filter_appendix_en(self):
        s = llm_filter_free_text_language_appendix("en")
        self.assertIn("English", s)

    def test_filter_appendix_empty_string(self):
        self.assertEqual(llm_filter_free_text_language_appendix(""), "")

    def test_agent_insight_prompt_zh(self):
        s = agent_insight_system_prompt("zh")
        self.assertIn("分析员", s)

    def test_agent_insight_prompt_en(self):
        s = agent_insight_system_prompt("en")
        self.assertIn("analyst", s.lower())

    def test_dialog_detection_note_intent_zh(self):
        s = dialog_detection_output_language_note("zh", "intent")
        self.assertIn("意图分析", s)
        self.assertIn("简体中文", s)

    def test_dialog_detection_note_intent_en(self):
        s = dialog_detection_output_language_note("en", "intent")
        self.assertIn("English", s)

    def test_dialog_detection_note_topic_zh(self):
        s = dialog_detection_output_language_note("zh", "topic")
        self.assertIn("话题", s)

    def test_dialog_detection_note_sentiment_zh(self):
        s = dialog_detection_output_language_note("zh", "sentiment")
        self.assertIn("情感", s)

    def test_dialog_detection_note_intensity_zh(self):
        s = dialog_detection_output_language_note("zh", "intensity")
        self.assertIn("情绪", s)

    def test_dialog_detection_note_unknown_mode(self):
        """Unknown mode should return empty string."""
        self.assertEqual(dialog_detection_output_language_note("zh", "unknown"), "")
        self.assertEqual(dialog_detection_output_language_note("en", "foobar"), "")


if __name__ == "__main__":
    unittest.main()
