"""Tests for source-text field lookup and removal."""

import unittest
from copy import deepcopy

import pyarrow as pa
from datasets import Dataset, Features, LargeList, List, Value
from datasets.formatting.formatting import LazyBatch, PythonFormatter

from data_juicer.core.data.dj_dataset import (
    NestedDataset,
    NestedQueryDict,
    nested_obj_factory,
    nested_query,
)
from data_juicer.utils.text_field_utils import (
    read_text_from_sample,
    remove_text_from_dataset,
    remove_text_from_sample,
)


class TextFieldUtilsTest(unittest.TestCase):
    def test_sample_cleanup_after_removing_last_nested_field(self):
        cases = [
            ("payload.text", {"payload": {"text": "source"}}, {}),
            ("payload.text", {"payload": {"text": "source", "lang": "en"}}, {"payload": {"lang": "en"}}),
            ("payload.parts.text", {"payload": {"parts": [{"text": "source"}]}}, {}),
            ("payload.parts.text", {"payload": {"parts": [{"text": "source"}, None]}}, {}),
            ("payload.parts.text", {"payload": {"parts": []}}, {"payload": {"parts": []}}),
            ("payload.parts.text", {"payload": {"parts": None}}, {"payload": {"parts": None}}),
            ("payload.text", {"payload": {"text": "source", "other": {}}}, {"payload": {"other": {}}}),
        ]
        for text_key, row, expected in cases:
            with self.subTest(text_key=text_key, row=row):
                source = deepcopy(row)
                result = remove_text_from_sample(deepcopy(row), text_key)
                self.assertEqual(result, expected)
                self.assertEqual(remove_text_from_sample(result, text_key), expected)
                self.assertEqual(row, source)

    def test_literal_key_takes_precedence(self):
        cases = [
            (
                "payload.text",
                {"payload.text": "literal", "payload": {"text": "nested", "other": "keep"}},
                {"payload": {"text": "nested", "other": "keep"}},
            ),
            (
                "a.b.c",
                {"a": {"b.c": None, "b": {"c": "keep"}}},
                {"a": {"b": {"c": "keep"}}},
            ),
        ]
        for text_key, row, expected in cases:
            with self.subTest(text_key=text_key):
                self.assertEqual(remove_text_from_sample(deepcopy(row), text_key), expected)
                result = remove_text_from_dataset(NestedDataset(Dataset.from_list([deepcopy(row)])), text_key)
                self.assertEqual(list(result), [expected])
                self.assertEqual(result.features, Dataset.from_list([expected]).features)

    def test_competing_dotted_prefixes(self):
        row = {"a.b": {"c": "keep"}, "a": {"b": {"c": "secret", "other": "keep"}}}
        pristine = deepcopy(row)
        self.assertEqual(nested_obj_factory(deepcopy(row))["a.b.c"], "secret")
        sample = remove_text_from_sample(deepcopy(row), "a.b.c")
        self.assertEqual(sample["a.b"], {"c": "keep"})
        self.assertEqual(sample["a"]["b"], {"other": "keep"})
        self.assertEqual(row, pristine)
        result = remove_text_from_dataset(NestedDataset(Dataset.from_list([deepcopy(row)])), "a.b.c")[0]
        self.assertEqual(result["a.b"], {"c": "keep"})
        self.assertEqual(result["a"]["b"], {"other": "keep"})

    def test_nullable_and_absent_paths(self):
        cases = [
            ("text", {"text": None, "other": "keep"}, {"other": "keep"}),
            ("payload.text", {"payload": {"text": "secret", "other": "keep"}}, {"payload": {"other": "keep"}}),
            ("payload.text", {"payload": None}, {"payload": None}),
            ("payload.items.text", {"payload": {"items": None}}, {"payload": {"items": None}}),
            ("a.b.c", {"a.b": {"c": "keep"}}, {"a.b": {"c": "keep"}}),
            ("a.b.c", {"a.b": [{"c": "keep"}]}, {"a.b": [{"c": "keep"}]}),
            ("a.b.c.text", {"a": {"b.c": {"text": "keep"}}}, {"a": {"b.c": {"text": "keep"}}}),
        ]
        for text_key, row, expected in cases:
            with self.subTest(text_key=text_key, row=row):
                direct = remove_text_from_sample(deepcopy(row), text_key)
                self.assertEqual(direct, expected)
                self.assertEqual(remove_text_from_sample(direct, text_key), expected)
                dataset = NestedDataset(Dataset.from_list([deepcopy(row)]))
                features = deepcopy(dataset.features)
                result = remove_text_from_dataset(dataset, text_key)
                self.assertEqual(list(result), [expected])
                self.assertEqual(result.features, Dataset.from_list([expected]).features)
                repeated = remove_text_from_dataset(result, text_key)
                self.assertEqual(list(repeated), [expected])
                self.assertEqual(repeated.features, result.features)
                self.assertEqual(list(dataset), [row])
                self.assertEqual(dataset.features, features)

    def test_read_matches_nested_query_for_resolved_paths(self):
        cases = [
            ("text", {"text": "source"}),
            ("text", {"text": None}),
            ("payload.text", {"payload": {"text": "source", "other": "keep"}}),
            ("payload.text", {"payload": None}),
            ("payload.text", {"payload": []}),
            ("payload.text", {"payload": [[{"text": "source"}, None], [], None]}),
            ("payload.items.text", {"payload": {"items": [{"text": "source"}, None, {"text": None}, {}]}}),
            ("payload.text", {"payload.text": "literal", "payload": {"text": "nested"}}),
            ("payload.text", {"payload.text": None, "payload": {"text": "nested"}, "text": "sibling"}),
            ("a.b.c", {"a.b": {"c": "wrong"}, "a": {"b": {"c": "source"}}}),
            ("a.b.c", {"a": {"b.c": "literal", "b": {"c": "nested"}}}),
        ]
        for text_key, row in cases:
            for kind in (dict, NestedQueryDict, LazyBatch):
                with self.subTest(text_key=text_key, row=row, kind=kind.__name__):
                    source = deepcopy(row)
                    if kind is LazyBatch:
                        source = {key: [value] for key, value in source.items()}
                    expected = nested_query(deepcopy(source), text_key)
                    sample = (
                        LazyBatch(pa.Table.from_pydict(source), PythonFormatter())
                        if kind is LazyBatch
                        else kind(source)
                    )
                    self.assertEqual(read_text_from_sample(sample, text_key), expected)

    def test_struct_list_schema_contract(self):
        rows = [
            {
                "payload": {
                    "items": [{"text": "secret", "other": "item"}, None, {"text": None, "other": "null"}],
                    "other": "parent",
                }
            },
            {"payload": {"items": [], "other": "empty"}},
            {"payload": {"items": None, "other": "null list"}},
            {"payload": None},
        ]
        pruned = [
            {"payload": {"items": [{"other": "item"}, None, {"other": "null"}], "other": "parent"}},
            *deepcopy(rows[1:]),
        ]
        for list_type in (List, LargeList):
            features = Features(
                {
                    "payload": {
                        "items": list_type({"text": Value("string"), "other": Value("string")}),
                        "other": Value("string"),
                    }
                }
            )
            for text_key in ("payload.items.text", "payload.items.missing", "payload.missing.text", "missing.text"):
                with self.subTest(list_type=list_type.__name__, text_key=text_key):
                    expected = pruned if text_key == "payload.items.text" else rows
                    expected_features = deepcopy(features)
                    if text_key == "payload.items.text":
                        del expected_features["payload"]["items"].feature["text"]
                    for row, expected_row in zip(rows, expected):
                        sample = remove_text_from_sample(deepcopy(row), text_key)
                        self.assertEqual(sample, expected_row)
                        self.assertEqual(remove_text_from_sample(sample, text_key), expected_row)
                    batch = LazyBatch(
                        pa.Table.from_pylist(deepcopy(rows), schema=features.arrow_schema), PythonFormatter()
                    )
                    expected_batch = {"payload": [row["payload"] for row in expected]}
                    self.assertEqual(dict(remove_text_from_sample(batch, text_key)), expected_batch)
                    dataset = NestedDataset(Dataset.from_list(deepcopy(rows), features=deepcopy(features)))
                    result = remove_text_from_dataset(dataset, text_key)
                    self.assertEqual(list(result), expected)
                    self.assertEqual(result.features, expected_features)
                    repeated = remove_text_from_dataset(result, text_key)
                    self.assertEqual(list(repeated), expected)
                    self.assertEqual(repeated.features, expected_features)
                    self.assertEqual(list(dataset), rows)
                    self.assertEqual(dataset.features, features)
