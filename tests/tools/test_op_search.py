import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.tools.op_search import OPSearcher


class OPRecordTest(DataJuicerTestCaseBase):
    """Tests for OPRecord metadata extraction."""

    def setUp(self):
        # Use a small fixed op list to keep tests fast
        self.searcher = OPSearcher(
            specified_op_list=["text_length_filter", "language_id_score_filter"]
        )

    def test_record_has_required_fields(self):
        record = self.searcher.op_records[0]
        for field in ["name", "type", "desc", "tags", "sig", "source_path", "test_path"]:
            self.assertIsNotNone(record[field], f"Field '{field}' should not be None")

    def test_record_type_is_valid(self):
        from data_juicer.tools.op_search import op_type_list
        for record in self.searcher.op_records:
            self.assertIn(record.type, op_type_list + ["unknown"])

    def test_record_tags_is_list(self):
        for record in self.searcher.op_records:
            self.assertIsInstance(record.tags, list)

    def test_record_to_dict_keys(self):
        record = self.searcher.op_records[0]
        record_dict = record.to_dict()
        expected_keys = {
            "type", "name", "desc", "tags", "sig",
            "param_desc", "param_desc_map", "source_path", "test_path",
        }
        self.assertTrue(expected_keys.issubset(set(record_dict.keys())))

    def test_record_getitem_and_setitem(self):
        record = self.searcher.op_records[0]
        original_name = record["name"]
        record["name"] = "test_name"
        self.assertEqual(record["name"], "test_name")
        record["name"] = original_name

    def test_record_getitem_missing_key_raises(self):
        record = self.searcher.op_records[0]
        with self.assertRaises(KeyError):
            _ = record["nonexistent_field"]


class OPSearcherBasicSearchTest(DataJuicerTestCaseBase):
    """Tests for OPSearcher.search() — tag and type filtering."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.searcher = OPSearcher()

    def test_search_all_returns_nonempty(self):
        results = self.searcher.search()
        self.assertGreater(len(results), 0)

    def test_search_by_op_type_filter(self):
        results = self.searcher.search(op_type="filter")
        self.assertGreater(len(results), 0)
        for op in results:
            self.assertEqual(op["type"], "filter")

    def test_search_by_op_type_mapper(self):
        results = self.searcher.search(op_type="mapper")
        self.assertGreater(len(results), 0)
        for op in results:
            self.assertEqual(op["type"], "mapper")

    def test_search_by_tag_cpu(self):
        results = self.searcher.search(tags=["cpu"])
        self.assertGreater(len(results), 0)
        for op in results:
            self.assertIn("cpu", op["tags"])

    def test_search_match_all_tags(self):
        results = self.searcher.search(tags=["text", "cpu"], match_all=True)
        for op in results:
            self.assertIn("text", op["tags"])
            self.assertIn("cpu", op["tags"])

    def test_search_match_any_tag(self):
        results_any = self.searcher.search(tags=["text", "cpu"], match_all=False)
        results_all = self.searcher.search(tags=["text", "cpu"], match_all=True)
        # match_any should return at least as many results as match_all
        self.assertGreaterEqual(len(results_any), len(results_all))

    def test_search_by_type_and_tag(self):
        results = self.searcher.search(op_type="filter", tags=["text"])
        for op in results:
            self.assertEqual(op["type"], "filter")
            self.assertIn("text", op["tags"])

    def test_search_invalid_type_returns_empty(self):
        results = self.searcher.search(op_type="nonexistent_type")
        self.assertEqual(results, [])

    def test_search_result_is_list_of_dicts(self):
        results = self.searcher.search(op_type="filter")
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, dict)


class OPSearcherRegexSearchTest(DataJuicerTestCaseBase):
    """Tests for OPSearcher.search_by_regex()."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.searcher = OPSearcher()

    def test_regex_search_by_name(self):
        results = self.searcher.search_by_regex(query="text_length")
        self.assertGreater(len(results), 0)
        names = [op["name"] for op in results]
        self.assertTrue(any("text_length" in name for name in names))

    def test_regex_search_case_insensitive(self):
        results_lower = self.searcher.search_by_regex(query="filter")
        results_upper = self.searcher.search_by_regex(query="FILTER")
        self.assertEqual(len(results_lower), len(results_upper))

    def test_regex_search_with_op_type_filter(self):
        results = self.searcher.search_by_regex(query="text", op_type="filter")
        for op in results:
            self.assertEqual(op["type"], "filter")
            

    def test_regex_search_with_tag_filter(self):
        results = self.searcher.search_by_regex(query="language", tags=["cpu"])
        for op in results:
            self.assertIn("cpu", op["tags"])

    def test_regex_search_custom_fields(self):
        results = self.searcher.search_by_regex(query="filter", fields=["name"])
        self.assertGreater(len(results), 0)
        for op in results:
            self.assertIn("filter", op["name"])

    def test_regex_search_invalid_pattern_returns_empty(self):
        results = self.searcher.search_by_regex(query="[invalid(regex")
        self.assertEqual(results, [])

    def test_regex_search_no_match_returns_empty(self):
        results = self.searcher.search_by_regex(query="xyzzy_no_such_op_12345")
        self.assertEqual(results, [])

    def test_regex_search_result_is_list_of_dicts(self):
        results = self.searcher.search_by_regex(query="text")
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, dict)


class OPSearcherBM25SearchTest(DataJuicerTestCaseBase):
    """Tests for OPSearcher.search_by_bm25()."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.searcher = OPSearcher()

    def test_bm25_search_returns_results(self):
        results = self.searcher.search_by_bm25(query="filter text by length")
        self.assertGreater(len(results), 0)

    def test_bm25_search_top_k(self):
        results = self.searcher.search_by_bm25(query="text", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_bm25_search_score_threshold(self):
        # With a very high threshold, should return fewer or no results
        results_high = self.searcher.search_by_bm25(query="text", score_threshold=999.0)
        results_low = self.searcher.search_by_bm25(query="text", score_threshold=0.0)
        self.assertLessEqual(len(results_high), len(results_low))

    def test_bm25_search_with_op_type_filter(self):
        results = self.searcher.search_by_bm25(
            query="language detection", op_type="filter"
        )
        for op in results:
            self.assertEqual(op["type"], "filter")

    def test_bm25_search_with_tag_filter(self):
        results = self.searcher.search_by_bm25(query="text processing", tags=["cpu"])
        for op in results:
            self.assertIn("cpu", op["tags"])

    def test_bm25_search_custom_fields(self):
        results = self.searcher.search_by_bm25(query="length", fields=["name", "desc"])
        self.assertIsInstance(results, list)

    def test_bm25_search_empty_candidates_returns_empty(self):
        results = self.searcher.search_by_bm25(
            query="text", op_type="nonexistent_type"
        )
        self.assertEqual(results, [])

    def test_bm25_search_result_is_list_of_dicts(self):
        results = self.searcher.search_by_bm25(query="image quality")
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, dict)

    def test_bm25_index_rebuilt_on_different_filters(self):
        # Ensure index is correctly rebuilt when filter conditions change
        results_filter = self.searcher.search_by_bm25(query="text", op_type="filter")
        results_mapper = self.searcher.search_by_bm25(query="text", op_type="mapper")
        # Results should differ since the candidate pool is different
        filter_names = {op["name"] for op in results_filter}
        mapper_names = {op["name"] for op in results_mapper}
        self.assertTrue(filter_names.isdisjoint(mapper_names))

    def test_bm25_relevance_ordering(self):
        # "text_length_filter" should appear in results for a length-related query
        results = self.searcher.search_by_bm25(
            query="text length filter", top_k=5
        )
        names = [op["name"] for op in results]
        self.assertTrue(any("length" in name for name in names))


class OPSearcherSpecifiedOpsTest(DataJuicerTestCaseBase):
    """Tests for OPSearcher initialized with a specified op list."""

    def test_specified_ops_only_scans_given_ops(self):
        op_list = ["text_length_filter", "language_id_score_filter"]
        searcher = OPSearcher(specified_op_list=op_list)
        self.assertEqual(len(searcher.op_records), len(op_list))
        names = [r.name for r in searcher.op_records]
        for op_name in op_list:
            self.assertIn(op_name, names)

    def test_all_ops_dict_populated(self):
        op_list = ["text_length_filter"]
        searcher = OPSearcher(specified_op_list=op_list)
        self.assertIn("text_length_filter", searcher.all_ops)

    def test_records_map_deprecated_still_returns_all_ops(self):
        searcher = OPSearcher(specified_op_list=["text_length_filter"])
        # records_map should still return all_ops (with a deprecation warning logged)
        records_map = searcher.records_map
        self.assertEqual(records_map, searcher.all_ops)


if __name__ == '__main__':
    unittest.main()
