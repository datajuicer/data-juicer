# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.ops.mapper.pii_llm_suspect_mapper import (
    DEFAULT_REDACTION_PLACEHOLDER,
    PiiLlmSuspectMapper,
    _extract_json_object,
    _heuristic_trigger,
    _name_like_rule_trigger,
    _normalize_spacy_ner_model_names,
    _resolve_spacy_auto_download_flag,
    ensure_spacy_pipeline_installed,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestPiiLlmSuspectHelpers(DataJuicerTestCaseBase):
    def test_heuristic_long_digits(self):
        self.assertTrue(_heuristic_trigger("call me 13812345678"))

    def test_heuristic_email_at(self):
        self.assertTrue(_heuristic_trigger("not redacted a@b.co"))

    def test_heuristic_secret_keyword(self):
        self.assertTrue(_heuristic_trigger("api_key: something"))

    def test_heuristic_negative(self):
        self.assertFalse(_heuristic_trigger("hello world only"))

    def test_name_rule_zh_intro(self):
        self.assertTrue(_name_like_rule_trigger("我叫张三想咨询流程"))

    def test_name_rule_zh_suffix(self):
        self.assertTrue(_name_like_rule_trigger("经办人李四"))

    def test_name_rule_en_title(self):
        self.assertTrue(_name_like_rule_trigger("Seen with Dr. Watson yesterday."))

    def test_name_rule_negative_plain(self):
        self.assertFalse(_name_like_rule_trigger("hello world only generic text"))

    def test_normalize_spacy_models_list_and_legacy(self):
        self.assertEqual(
            _normalize_spacy_ner_model_names("en_core_web_sm", ["zh_core_web_sm"]),
            ["zh_core_web_sm", "en_core_web_sm"],
        )

    def test_normalize_spacy_models_dedup(self):
        self.assertEqual(
            _normalize_spacy_ner_model_names(
                "zh_core_web_sm",
                ["zh_core_web_sm", "en_core_web_sm"],
            ),
            ["zh_core_web_sm", "en_core_web_sm"],
        )

    def test_resolve_spacy_auto_download_env_off(self):
        with patch.dict(os.environ, {"PII_SPACY_AUTO_DOWNLOAD": "0"}):
            self.assertFalse(_resolve_spacy_auto_download_flag(True))

    def test_resolve_spacy_auto_download_env_on(self):
        with patch.dict(os.environ, {"PII_SPACY_AUTO_DOWNLOAD": "1"}):
            self.assertTrue(_resolve_spacy_auto_download_flag(False))

    @patch(
        "spacy.util.is_package",
        return_value=False,
    )
    @patch("spacy.cli.download")
    def test_ensure_spacy_calls_download(self, mock_download, _mock_pkg):
        ensure_spacy_pipeline_installed("en_core_web_sm", auto_download=True)
        mock_download.assert_called_once_with("en_core_web_sm")

    @patch("spacy.util.is_package", return_value=True)
    @patch("spacy.cli.download")
    def test_ensure_spacy_skips_when_installed(self, mock_download, _mock_pkg):
        ensure_spacy_pipeline_installed("en_core_web_sm", auto_download=True)
        mock_download.assert_not_called()

    def test_extract_json_object(self):
        raw = 'prefix {"suspected": [], "likely_clean": true} suffix'
        obj = _extract_json_object(raw)
        self.assertIsNotNone(obj)
        self.assertEqual(obj["suspected"], [])
        self.assertTrue(obj["likely_clean"])

    def test_extract_json_fenced(self):
        raw = '```json\n{"a": 1}\n```'
        obj = _extract_json_object(raw)
        self.assertEqual(obj, {"a": 1})


class TestPiiLlmSuspectMapper(DataJuicerTestCaseBase):
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_heuristic_gate_skips(self, mock_get, mock_prepare):
        mock_prepare.return_value = lambda device="cpu": None

        m = PiiLlmSuspectMapper(api_model="qwen-turbo", gate_mode="heuristic")
        sample = {Fields.meta: {}, "text": "short"}
        out = m.process_single(sample)
        self.assertFalse(mock_get.called)
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertTrue(meta.get("skipped"))
        self.assertEqual(meta.get("reason"), "heuristic_gate")

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_heuristic_name_rules_opens_gate(self, mock_get, mock_prepare):
        """Person-only prose should call LLM when heuristic_name_rules is on (default)."""

        def fake_client(messages, **kwargs):
            return '{"suspected": [], "likely_clean": true}'

        mock_prepare.return_value = lambda device="cpu": None
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(
            api_model="qwen-turbo",
            gate_mode="heuristic",
            heuristic_name_rules=True,
        )
        sample = {
            Fields.meta: {},
            "query": "我叫王五彩想询问订单状态，没有任何数字电话",
        }
        out = m.process_single(sample)
        self.assertTrue(mock_get.called)
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertFalse(meta.get("skipped"))

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_heuristic_name_rules_disabled_skips(self, mock_get, mock_prepare):
        mock_prepare.return_value = lambda device="cpu": None

        m = PiiLlmSuspectMapper(
            api_model="qwen-turbo",
            gate_mode="heuristic",
            heuristic_name_rules=False,
        )
        sample = {
            Fields.meta: {},
            "query": "我叫王五彩想询问订单状态，没有任何数字电话",
        }
        out = m.process_single(sample)
        self.assertFalse(mock_get.called)
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertTrue(meta.get("skipped"))
        self.assertEqual(meta.get("reason"), "heuristic_gate")

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_spacy_ner_models_zh_then_en(self, mock_get, mock_prepare):
        """Both pipelines run order; English model can open gate after Chinese miss."""

        class Ent:
            label_ = "PERSON"

        class DocHit:
            ents = [Ent()]

        class DocMiss:
            ents = []

        def make_nlp(hit: bool):
            class Nlp:
                def __call__(self, text):
                    return DocHit() if hit else DocMiss()

            return Nlp()

        loads = {
            "zh_core_web_sm": make_nlp(False),
            "en_core_web_sm": make_nlp(True),
        }

        def fake_load(name):
            return loads[name]

        fake_spacy = MagicMock()
        fake_spacy.load.side_effect = fake_load
        with patch.dict(sys.modules, {"spacy": fake_spacy}):

            def fake_client(messages, **kwargs):
                return '{"suspected": [], "likely_clean": true}'

            mock_prepare.return_value = lambda device="cpu": None
            mock_get.return_value = fake_client

            m = PiiLlmSuspectMapper(
                api_model="qwen-turbo",
                gate_mode="heuristic",
                heuristic_name_rules=False,
                spacy_ner_models=["zh_core_web_sm", "en_core_web_sm"],
            )
            sample = {
                Fields.meta: {},
                "query": "John Smith only no zh cue no digits at all xxxx",
            }
            out = m.process_single(sample)
            self.assertTrue(mock_get.called)
            meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
            self.assertFalse(meta.get("skipped"))
            self.assertEqual(fake_spacy.load.call_count, 2)

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_llm_writes_meta(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return (
                '{"suspected":[{"field":"query","category":"phone",'
                '"evidence":"138****"}], "likely_clean": false}'
            )

        mock_prepare.return_value = lambda device="cpu": fake_client
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(api_model="qwen-turbo", gate_mode="always")
        sample = {
            Fields.meta: {},
            "query": "phone 13812345678 here",
        }
        out = m.process_single(sample)
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertEqual(len(meta.get("suspected", [])), 1)
        self.assertEqual(meta["suspected"][0]["field"], "query")
        self.assertFalse(meta.get("likely_clean", True))

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_no_overwrite(self, mock_get, mock_prepare):
        mock_prepare.return_value = lambda device="cpu": None
        existing = {"suspected": [], "likely_clean": True}
        m = PiiLlmSuspectMapper(api_model="qwen-turbo", overwrite=False)
        sample = {
            Fields.meta: {MetaKeys.pii_llm_suspect: existing},
            "query": "13812345678",
        }
        out = m.process_single(sample)
        self.assertIs(out[Fields.meta][MetaKeys.pii_llm_suspect], existing)
        self.assertFalse(mock_get.called)

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_redaction_evidence_substrings(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return (
                '{"suspected":[{"field":"query","category":"phone",'
                '"evidence":"13812345678"}], "likely_clean": false}'
            )

        mock_prepare.return_value = lambda device="cpu": fake_client
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(
            api_model="qwen-turbo",
            gate_mode="always",
            redaction_mode="evidence",
        )
        sample = {
            Fields.meta: {},
            "query": "phone 13812345678 here",
        }
        out = m.process_single(sample)
        self.assertIn(DEFAULT_REDACTION_PLACEHOLDER, out["query"])
        self.assertNotIn("13812345678", out["query"])
        meta = out[Fields.meta][MetaKeys.pii_llm_suspect]
        self.assertEqual(meta.get("redaction_mode"), "evidence")
        self.assertTrue(meta.get("redaction_applied"))

    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.prepare_model")
    @patch("data_juicer.ops.mapper.pii_llm_suspect_mapper.get_model")
    def test_redaction_whole_field(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return (
                '{"suspected":[{"field":"query","category":"risk",'
                '"evidence":""}], "likely_clean": false}'
            )

        mock_prepare.return_value = lambda device="cpu": fake_client
        mock_get.return_value = fake_client

        m = PiiLlmSuspectMapper(
            api_model="qwen-turbo",
            gate_mode="always",
            redaction_mode="whole_field",
        )
        sample = {
            Fields.meta: {},
            "query": "sensitive entire field",
        }
        out = m.process_single(sample)
        self.assertEqual(out["query"], DEFAULT_REDACTION_PLACEHOLDER)


if __name__ == "__main__":
    unittest.main()
