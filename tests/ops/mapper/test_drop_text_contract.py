"""Regression tests for source-text removal in LLM extraction mappers."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
from datasets import Dataset, Features, LargeList, List, Value
from datasets.formatting.formatting import LazyBatch, PythonFormatter

from data_juicer.core.data.dj_dataset import NestedDataset, NestedQueryDict, nested_obj_factory, nested_query
from data_juicer.ops.mapper import extract_event_mapper, extract_keyword_mapper, extract_support_text_mapper
from data_juicer.ops.mapper.drop_text import read_text_from_sample, remove_text_from_dataset, remove_text_from_sample
from data_juicer.utils.constant import Fields, MetaKeys


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
            return extract_keyword_mapper.ExtractKeywordMapper(
                api_model="fake", drop_text=True, try_num=1, num_proc=1, auto_op_parallelism=False
            )

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

        pristine = deepcopy(row)
        direct = op.process_single(nested_obj_factory(deepcopy(row)))
        self.assertEqual(direct["other"], "keep")
        self.assertNotIn("text", direct if text_key == "text" else direct["payload"])
        self.assertTrue(direct[Fields.meta]["keyword"])
        self.assertEqual(row, pristine)

        processed = op.run(NestedDataset(Dataset.from_list([deepcopy(row)])))
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

    def test_event_cleanup_preserves_original_literal_path(self):
        cases = [
            (
                "payload.text",
                {"payload.text": "secret", "payload": {"text": "keep", "other": "keep"}},
                {"payload": {"text": "keep", "other": "keep"}},
            ),
            (
                "payload.a.b",
                {"payload": {"a.b": "secret", "a": {"b": "keep"}, "other": "keep"}},
                {"payload": {"a": {"b": "keep"}, "other": "keep"}},
            ),
        ]
        for text_key, source, expected in cases:
            for existing in (False, True):
                with self.subTest(text_key=text_key, existing=existing):
                    op = self.make_event_op(text_key=text_key)
                    op._process_single_sample = Mock(return_value=(["event"], [["A"]]))
                    meta = {MetaKeys.event_description: "event", MetaKeys.relevant_characters: ["A"]}
                    row = {**deepcopy(source), Fields.meta: deepcopy(meta) if existing else {}}
                    dataset = NestedDataset(Dataset.from_list([row]))
                    result = op.run(dataset)
                    self.assertEqual(list(result), [{**expected, Fields.meta: meta}])
                    repeated = remove_text_from_dataset(result, text_key, dataset.features)
                    self.assertEqual(list(repeated), list(result))
                    self.assertEqual(repeated.features, result.features)

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
            return extract_event_mapper.ExtractEventMapper(
                api_model="fake", drop_text=drop_text, text_key=text_key, num_proc=1, auto_op_parallelism=False
            )

    def test_event_routes_contract(self):
        cases = [
            ("text", ["first", "second"], None, ["first", "second"]),
            (
                "payload.text",
                [{"text": "first", "sibling": "one"}, {"text": "second", "sibling": "two"}],
                [{"sibling": "one"}, {"sibling": "two"}],
                ["first", "second"],
            ),
            (
                "payload.text",
                [[{"text": "first", "sibling": "one"}, None], [{"text": "second", "sibling": "two"}]],
                [[{"sibling": "one"}, None], [{"sibling": "two"}]],
                [["first", None], ["second"]],
            ),
            (
                "payload.items.text",
                [
                    {"items": [{"text": "first", "sibling": "one"}, None], "sibling": "parent one"},
                    {"items": [None, {"text": "second", "sibling": "two"}], "sibling": "parent two"},
                ],
                [
                    {"items": [{"sibling": "one"}, None], "sibling": "parent one"},
                    {"items": [None, {"sibling": "two"}], "sibling": "parent two"},
                ],
                [["first", None], [None, "second"]],
            ),
        ]
        for text_key, values, pruned, texts in cases:
            column = text_key.split(".")[0]
            rows = [
                {column: value, "other": f"row {i}", Fields.meta: {"origin": f"meta {i}"}}
                for i, value in enumerate(values)
            ]
            pristine = deepcopy(rows)
            for drop_text in (False, True):
                for route in ("batch", "arrow", "dataset"):
                    with self.subTest(text_key=text_key, texts=texts, drop_text=drop_text, route=route):
                        op = self.make_event_op(drop_text=drop_text, text_key=text_key)

                        def extract(text, rank=None):
                            if text == texts[0]:
                                return ["event one", "event two"], [["A"], ["B"]]
                            if text == texts[1]:
                                return ["event three"], [["C"]]
                            raise AssertionError(f"Unexpected model input: {text!r}")

                        op._process_single_sample = Mock(side_effect=extract)
                        expected = []
                        for i, event, characters in [
                            (0, "event one", ["A"]), (0, "event two", ["B"]), (1, "event three", ["C"])
                        ]:
                            row = deepcopy(rows[i])
                            if drop_text:
                                if pruned is None:
                                    del row[column]
                                else:
                                    row[column] = deepcopy(pruned[i])
                            row[Fields.meta].update(
                                {MetaKeys.event_description: event, MetaKeys.relevant_characters: characters}
                            )
                            expected.append(row)
                        fresh_rows = deepcopy(rows)
                        if route == "dataset":
                            result = op.run(NestedDataset(Dataset.from_list(fresh_rows)))
                            actual = list(result)
                            self.assertEqual(result.features, Dataset.from_list(expected).features)
                            with tempfile.TemporaryDirectory() as directory:
                                output = Path(directory) / "output.jsonl"
                                result.to_json(str(output))
                                exported = [json.loads(line) for line in output.read_text().splitlines()]
                                self.assertEqual(exported, expected)
                        else:
                            batch = {key: [row[key] for row in fresh_rows] for key in fresh_rows[0]}
                            result = (
                                op.process_batched(nested_obj_factory(batch))
                                if route == "batch"
                                else op.process(pa.Table.from_pylist(fresh_rows))
                            )
                            self.assertEqual(result, {key: [row[key] for row in expected] for key in expected[0]})
                            actual = [dict(zip(result, values)) for values in zip(*result.values())]
                            self.assertEqual([call.args[0] for call in op._process_single_sample.call_args_list], texts)
                        self.assertEqual(len(actual), 3)
                        self.assertEqual(actual, expected)
                        self.assertEqual(rows, pristine)

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

    def test_list_of_structs(self):
        op = self.make_keyword_op()
        op.text_key = "payload.text"
        row = {"payload": [{"text": "secret", "other": "keep"}], Fields.meta: {"keyword": ["existing"]}}
        pristine = deepcopy(row)
        direct = op.process_single(nested_obj_factory(deepcopy(row)))
        self.assertEqual(direct["payload"], [{"other": "keep"}])
        self.assertEqual(row, pristine)
        result = op.run(NestedDataset(Dataset.from_list([deepcopy(row)])))[0]
        self.assertEqual(result["payload"], [{"other": "keep"}])

    def test_read_matches_nested_query(self):
        cases = [
            ("text", {"text": "source"}),
            ("text", {"text": None}),
            ("missing.text", {"other": "keep"}),
            ("payload.text", {"payload": {"text": "source", "other": "keep"}}),
            ("payload.text", {"payload": None}),
            ("payload.text", {"payload": []}),
            ("payload.text", {"payload": [[{"text": "source"}, None], [], None]}),
            ("payload.items.text", {"payload": {"items": [{"text": "source"}, None, {"text": None}, {}]}}),
            ("payload.text", {"payload.text": "literal", "payload": {"text": "nested"}}),
            ("payload.text", {"payload.text": None, "payload": {"text": "nested"}, "text": "sibling"}),
            ("a.b.c", {"a.b": {"c": "wrong"}, "a": {"b": {"c": "source"}}}),
            ("a.b.c", {"a": {"b.c": "literal", "b": {"c": "nested"}}}),
            ("a.b.c", {"a.b": {"c": "not a fallback"}}),
            ("a.b.c", {"a.b": [{"c": "not a fallback"}]}),
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

    def test_keyword_struct_list_contract(self):
        row = {
            "payload": {
                "items": [{"text": "source", "other": "keep"}, None, {"text": None, "other": "nullable"}],
                "other": "parent",
            },
            "other": "row",
            Fields.meta: {"origin": "keep"},
        }
        pristine = deepcopy(row)
        for drop_text in (False, True):
            for kind in (dict, NestedQueryDict):
                with self.subTest(drop_text=drop_text, kind=kind.__name__):
                    op = self.make_keyword_op()
                    op.text_key = "payload.items.text"
                    op.drop_text = drop_text
                    op.prompt_template = "{input_text}"
                    client = Mock(return_value='("content_keywords" "Alice, Beijing")')
                    with patch.object(extract_keyword_mapper, "get_model", return_value=client):
                        result = op.process_single(kind(deepcopy(row)))
                    client.assert_called_once_with([{"role": "user", "content": "['source', None, None]"}])
                    expected = deepcopy(row)
                    if drop_text:
                        expected["payload"]["items"] = [{"other": "keep"}, None, {"other": "nullable"}]
                    expected[Fields.meta][MetaKeys.keyword] = ["Alice", "Beijing"]
                    self.assertEqual(result, expected)
                    self.assertEqual(row, pristine)

    def test_struct_list_schema_contract(self):
        rows = [
            {"payload": {
                "items": [{"text": "secret", "other": "item"}, None, {"text": None, "other": "null"}],
                "other": "parent",
            }},
            {"payload": {"items": [], "other": "empty"}},
            {"payload": {"items": None, "other": "null list"}},
            {"payload": None},
        ]
        pruned = [
            {"payload": {"items": [{"other": "item"}, None, {"other": "null"}], "other": "parent"}},
            *deepcopy(rows[1:]),
        ]
        for list_type in (List, LargeList):
            features = Features({
                "payload": {
                    "items": list_type({"text": Value("string"), "other": Value("string")}),
                    "other": Value("string"),
                }
            })
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
