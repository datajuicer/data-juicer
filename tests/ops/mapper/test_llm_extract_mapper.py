"""Unit tests for LLMExtractMapper (llm_semantic_ops)."""

import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.llm_extract_mapper import LLMExtractMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK


@unittest.skipIf(
    FROM_FORK, "Skipping API-based test because running from a fork repo"
)
class TestLLMExtractMapper(DataJuicerTestCaseBase):
    api_or_hf_model = "gpt-4o"

    def test_extract_default(self):
        """Extract topic and sentiment; check meta and optional cost stats."""
        ds_list = [
            {"text": "The stock market rose today. Investors are optimistic."},
            {"text": "Bad weather caused delays. Many people were upset."},
        ]
        dataset = Dataset.from_list(ds_list)
        op = LLMExtractMapper(
            input_keys=["text"],
            output_schema={
                "topic": "One short phrase: main topic.",
                "sentiment": "One word: positive, negative, or neutral.",
            },
            api_or_hf_model=self.api_or_hf_model,
            meta_output_key=MetaKeys.llm_extract,
            try_num=2,
        )
        result = dataset.map(op.process, batch_size=1)
        out_list = result.to_list()
        self.assertEqual(len(out_list), 2)
        for sample in out_list:
            self.assertIn(Fields.meta, sample)
            meta = sample[Fields.meta]
            self.assertIn(MetaKeys.llm_extract, meta)
            extracted = meta[MetaKeys.llm_extract]
            self.assertIsInstance(extracted, dict)
            self.assertIn("topic", extracted)
            self.assertIn("sentiment", extracted)
            # Cost stats may be present when API returns usage
            if MetaKeys.llm_semantic_usage in meta:
                usage = meta[MetaKeys.llm_semantic_usage]
                self.assertIn("prompt_tokens", usage)
                self.assertIn("completion_tokens", usage)
                self.assertIn("total_tokens", usage)

    def test_extract_empty_input(self):
        """Empty input text yields nulls in output schema."""
        ds_list = [{"text": ""}]
        dataset = Dataset.from_list(ds_list)
        op = LLMExtractMapper(
            input_keys=["text"],
            output_schema={"topic": "Main topic.", "sentiment": "Sentiment."},
            api_or_hf_model=self.api_or_hf_model,
            meta_output_key=MetaKeys.llm_extract,
        )
        result = dataset.map(op.process, batch_size=1)
        out_list = result.to_list()
        self.assertEqual(len(out_list), 1)
        extracted = out_list[0][Fields.meta][MetaKeys.llm_extract]
        self.assertEqual(extracted["topic"], None)
        self.assertEqual(extracted["sentiment"], None)

    def test_extract_empty_input_flatten_to_meta(self):
        """When meta_output_key is None, fields are flattened into meta."""
        ds_list = [{"text": ""}]
        dataset = Dataset.from_list(ds_list)
        op = LLMExtractMapper(
            input_keys=["text"],
            output_schema={"topic": "Main topic.", "sentiment": "Sentiment."},
            api_or_hf_model=self.api_or_hf_model,
            meta_output_key=None,
        )
        result = dataset.map(op.process, batch_size=1)
        out_list = result.to_list()
        self.assertEqual(len(out_list), 1)
        meta = out_list[0][Fields.meta]
        self.assertIn("topic", meta)
        self.assertIn("sentiment", meta)
        self.assertEqual(meta["topic"], None)
        self.assertEqual(meta["sentiment"], None)
        self.assertNotIn(MetaKeys.llm_extract, meta)
