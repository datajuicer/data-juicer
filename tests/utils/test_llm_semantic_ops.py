"""Unit tests for llm_semantic_ops (RecordRow, RecordBatch, LLMCallUsage, prompts)."""

import unittest

from data_juicer.utils.llm_semantic_ops import (
    LLMCallUsage,
    RecordRow,
    get_condition_prompt,
    get_extract_prompt,
    record_batch_from_dicts,
    record_batch_to_dicts,
    InferenceStrategy,
)


class TestLLMCallUsage(unittest.TestCase):
    def test_to_dict(self):
        u = LLMCallUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        d = u.to_dict()
        self.assertEqual(d["prompt_tokens"], 10)
        self.assertEqual(d["completion_tokens"], 5)
        self.assertEqual(d["total_tokens"], 15)
        self.assertNotIn("cost_estimate", d)

    def test_to_dict_with_cost(self):
        u = LLMCallUsage(
            prompt_tokens=10, completion_tokens=5, cost_estimate=0.001
        )
        d = u.to_dict()
        self.assertEqual(d["cost_estimate"], 0.001)

    def test_from_dict(self):
        d = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        u = LLMCallUsage.from_dict(d)
        self.assertEqual(u.prompt_tokens, 20)
        self.assertEqual(u.completion_tokens, 8)
        self.assertEqual(u.total_tokens, 28)

    def test_from_dict_partial(self):
        u = LLMCallUsage.from_dict({})
        self.assertEqual(u.prompt_tokens, 0)
        self.assertEqual(u.completion_tokens, 0)
        self.assertEqual(u.total_tokens, 0)


class TestRecordRowAndBatch(unittest.TestCase):
    def test_record_row_from_dict(self):
        d = {"topic": "sports", "sentiment": "positive"}
        row = RecordRow.from_schema_dict(d)
        self.assertEqual(row.topic, "sports")
        self.assertEqual(row.sentiment, "positive")
        self.assertEqual(row.to_dict(), d)

    def test_record_row_from_schema_dict_restricts_keys(self):
        d = {"topic": "tech", "sentiment": "neutral", "extra": "ignored"}
        row = RecordRow.from_schema_dict(
            d, schema_keys=["topic", "sentiment"]
        )
        out = row.to_dict()
        self.assertIn("topic", out)
        self.assertIn("sentiment", out)
        self.assertNotIn("extra", out)

    def test_record_batch_roundtrip(self):
        items = [
            {"topic": "a", "sentiment": "pos"},
            {"topic": "b", "sentiment": "neg"},
        ]
        batch = record_batch_from_dicts(
            items, schema_keys=["topic", "sentiment"]
        )
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], RecordRow)
        back = record_batch_to_dicts(batch)
        self.assertEqual(back, items)


class TestPrompts(unittest.TestCase):
    def test_get_extract_prompt_direct(self):
        text = "Hello world."
        schema = {"topic": "Main topic.", "label": "One label."}
        prompt = get_extract_prompt(
            text, schema, strategy=InferenceStrategy.DIRECT
        )
        self.assertIn("Hello world.", prompt)
        self.assertIn("topic", prompt)
        self.assertIn("label", prompt)
        self.assertIn("Main topic.", prompt)

    def test_get_extract_prompt_with_grounding(self):
        text = "Content."
        schema = {"k": "Instruction."}
        prompt = get_extract_prompt(
            text,
            schema,
            knowledge_grounding="Background info.",
        )
        self.assertIn("Background", prompt)
        self.assertIn("Content.", prompt)

    def test_get_condition_prompt(self):
        prompt = get_condition_prompt("Some text.", "Contains a question.")
        self.assertIn("Some text.", prompt)
        self.assertIn("Contains a question.", prompt)
        self.assertIn("yes or no", prompt.lower())


if __name__ == "__main__":
    unittest.main()
