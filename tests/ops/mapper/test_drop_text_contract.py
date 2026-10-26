"""Regression tests for source-text removal in LLM extraction mappers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from data_juicer.core.data.dj_dataset import NestedDataset, nested_obj_factory
from data_juicer.ops.mapper import extract_event_mapper, extract_keyword_mapper, extract_support_text_mapper
from data_juicer.ops.mapper.drop_text import remove_text_from_dataset, remove_text_from_sample
from data_juicer.utils.constant import Fields


class DropTextContractTest(unittest.TestCase):
    def setUp(self):
        model_patch = patch.object(
            extract_keyword_mapper,
            "get_model",
            return_value=lambda *args, **kwargs: '("content_keywords" "Alice, Beijing")',
        )
        model_patch.start()
        self.addCleanup(model_patch.stop)

    def make_keyword_op(self):
        with patch.object(extract_keyword_mapper, "prepare_model", return_value="fake"):
            return extract_keyword_mapper.ExtractKeywordMapper(api_model="fake", drop_text=True, try_num=1)

    def check_direct_and_dataset(self, text_key, existing):
        op = self.make_keyword_op()
        op.text_key = text_key
        row = {"other": "keep", Fields.meta: {}}
        if text_key == "text":
            row["text"] = "Alice visited Beijing"
        else:
            row["payload"] = {"text": "Alice visited Beijing", "other": "nested keep"}
        if existing:
            row[Fields.meta]["keyword"] = ["already extracted"]

        direct = op.process_single(nested_obj_factory(row))
        self.assertEqual(direct["other"], "keep")
        self.assertNotIn("text", direct if text_key == "text" else direct["payload"])
        self.assertTrue(direct[Fields.meta]["keyword"])

        processed = op.run(NestedDataset(Dataset.from_list([row])))
        result = processed[0]
        self.assertEqual(result["other"], "keep")
        self.assertNotIn("text", result if text_key == "text" else result["payload"])
        if text_key != "text":
            self.assertEqual(result["payload"]["other"], "nested keep")
        self.assertTrue(result[Fields.meta]["keyword"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.jsonl"
            processed.to_json(str(output))
            self.assertNotIn("Alice visited Beijing", output.read_text())

    def test_top_level_text(self):
        self.check_direct_and_dataset("text", False)

    def test_existing_top_level_text(self):
        self.check_direct_and_dataset("text", True)

    def test_nested_text(self):
        self.check_direct_and_dataset("payload.text", False)

    def test_existing_nested_text(self):
        self.check_direct_and_dataset("payload.text", True)

    def test_drop_text_false_keeps_source(self):
        op = self.make_keyword_op()
        op.drop_text = False
        row = {"text": "Alice visited Beijing", Fields.meta: {}}
        result = op.run(NestedDataset(Dataset.from_list([row])))[0]
        self.assertEqual(result["text"], row["text"])

    def check_failed_model(self, text_key):
        op = self.make_keyword_op()
        op.text_key = text_key
        with patch.object(extract_keyword_mapper, "get_model", return_value=lambda *args, **kwargs: "invalid response"):
            row = {Fields.meta: {}, "text": "secret"} if text_key == "text" else {
                Fields.meta: {}, "payload": {"text": "secret", "other": "keep"}
            }
            result = op.run(NestedDataset(Dataset.from_list([row])))[0]
        self.assertNotIn("text", result if text_key == "text" else result["payload"])
        self.assertIn("keyword", result[Fields.meta])

    def test_failed_model_top_level(self):
        self.check_failed_model("text")

    def test_failed_model_nested(self):
        self.check_failed_model("payload.text")

    def test_literal_key_takes_precedence(self):
        row = {"payload.text": "literal", "payload": {"text": "nested", "other": "keep"}}
        self.assertEqual(remove_text_from_sample(row.copy(), "payload.text")["payload"]["text"], "nested")
        result = remove_text_from_dataset(NestedDataset(Dataset.from_list([row])), "payload.text")[0]
        self.assertEqual(result["payload"], {"text": "nested", "other": "keep"})
        self.assertNotIn("payload.text", result)

    def test_competing_dotted_prefixes(self):
        row = {"a.b": {"c": "keep"}, "a": {"b": {"c": "secret", "other": "keep"}}}
        self.assertEqual(nested_obj_factory(row)["a.b.c"], "secret")
        sample = remove_text_from_sample(row, "a.b.c")
        self.assertEqual(sample["a.b"], {"c": "keep"})
        self.assertEqual(sample["a"]["b"], {"other": "keep"})
        result = remove_text_from_dataset(NestedDataset(Dataset.from_list([row])), "a.b.c")[0]
        self.assertEqual(result["a.b"], {"c": "keep"})
        self.assertEqual(result["a"]["b"], {"other": "keep"})

    def test_multiple_rows_and_nested_siblings(self):
        op = self.make_keyword_op()
        op.text_key = "payload.text"
        rows = [
            {"payload": {"text": "Alice visited Beijing", "other": "first"}, Fields.meta: {}},
            {"payload": {"text": "Alice visited Beijing", "other": "second"}, Fields.meta: {}},
        ]
        result = op.run(NestedDataset(Dataset.from_list(rows)))
        self.assertEqual([row["payload"] for row in result], [{"other": "first"}, {"other": "second"}])
        self.assertTrue(all(row[Fields.meta]["keyword"] for row in result))

    def test_invalid_input_support_mapper(self):
        with patch.object(extract_support_text_mapper, "prepare_model", return_value="fake"):
            op = extract_support_text_mapper.ExtractSupportTextMapper(api_model="fake", drop_text=True)
        with patch.object(extract_support_text_mapper, "get_model", return_value=lambda *args, **kwargs: ""):
            result = op.process_single({"text": "secret", Fields.meta: {}})
        self.assertNotIn("text", result)

    def make_event_op(self, drop_text=True, text_key="text"):
        with patch.object(extract_event_mapper, "prepare_model", return_value="fake"):
            op = extract_event_mapper.ExtractEventMapper(api_model="fake", drop_text=drop_text, text_key=text_key)
        op._process_single_sample = lambda text, rank=None: (["event"], [["A"]])
        return op

    def test_one_to_many_event_mapper(self):
        op = self.make_event_op()
        op._process_single_sample = lambda text, rank=None: (["event one", "event two"], ["A", "B"])
        rows = [{"text": "secret", "other": "keep", Fields.meta: {}}]
        result = op.run(NestedDataset(Dataset.from_list(rows)))
        self.assertEqual(len(result), 2)
        self.assertNotIn("text", result.column_names)
        self.assertEqual([row["other"] for row in result], ["keep", "keep"])

    def test_event_mapper_nested_text(self):
        op = self.make_event_op(text_key="payload.text")
        rows = [{"payload": {"text": "secret", "other": "keep"}, Fields.meta: {}}]
        result = op.run(NestedDataset(Dataset.from_list(rows)))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["payload"], {"other": "keep"})

    def test_event_mapper_nested_lists_keep_text(self):
        op = self.make_event_op(drop_text=False, text_key="payload.text")
        seen = []

        def extract(text, rank=None):
            seen.append(text)
            return ["event"], [["A"]]

        op._process_single_sample = extract
        batch = nested_obj_factory({"payload": [[{"text": "hello", "other": "keep"}]], Fields.meta: [{}]})
        result = op.process_batched(batch)
        self.assertEqual(seen, [["hello"]])
        self.assertEqual(result["payload"], [[{"text": "hello", "other": "keep"}]])

    def test_nested_nullable_parent(self):
        rows = [{"payload": {"text": "secret", "other": "keep"}}, {"payload": None}]
        result = remove_text_from_dataset(NestedDataset(Dataset.from_list(rows)), "payload.text")
        self.assertEqual([row["payload"] for row in result], [{"other": "keep"}, None])

    def test_list_of_structs(self):
        op = self.make_keyword_op()
        op.text_key = "payload.text"
        row = {"payload": [{"text": "secret", "other": "keep"}], Fields.meta: {"keyword": ["existing"]}}
        direct = op.process_single(nested_obj_factory(row))
        self.assertEqual(direct["payload"], [{"other": "keep"}])
        result = op.run(NestedDataset(Dataset.from_list([row])))[0]
        self.assertEqual(result["payload"], [{"other": "keep"}])
