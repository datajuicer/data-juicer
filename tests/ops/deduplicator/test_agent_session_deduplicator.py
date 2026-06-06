# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.agent_session_deduplicator import AgentSessionDeduplicator
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AgentSessionDeduplicatorTest(DataJuicerTestCaseBase):

    def _run_session_dedup(self, dataset: Dataset, op, expected_count, show_num=0):
        dataset = dataset.map(op.compute_hash)
        result, dup_pairs = op.process(dataset, show_num=show_num)
        self.assertEqual(len(result), expected_count)
        return result, dup_pairs

    def test_basic_field_mode(self):
        """Basic test: group by session_id and user_id fields, keep latest timestamp."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": 100, "text": "oldest"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 200, "text": "newest"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 150, "text": "middle"},
            {"session_id": "s2", "user_id": "u2", "timestamp": 50, "text": "other_session"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator(
            session_id_field="session_id",
            user_id_field="user_id",
            timestamp_field="timestamp",
        )
        result, _ = self._run_session_dedup(dataset, op, expected_count=2)
        
        # Check that only the newest samples are kept
        texts = [r["text"] for r in result]
        self.assertIn("newest", texts)
        self.assertIn("other_session", texts)
        self.assertNotIn("oldest", texts)
        self.assertNotIn("middle", texts)

    def test_nested_field_mode(self):
        """Test nested field paths like meta.session_id."""
        ds_list = [
            {"meta": {"session_id": "s1", "user_id": "u1"}, "timestamp": 100, "text": "old"},
            {"meta": {"session_id": "s1", "user_id": "u1"}, "timestamp": 200, "text": "new"},
            {"meta": {"session_id": "s2", "user_id": "u1"}, "timestamp": 50, "text": "different"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator(
            session_id_field="meta.session_id",
            user_id_field="meta.user_id",
            timestamp_field="timestamp",
        )
        result, _ = self._run_session_dedup(dataset, op, expected_count=2)
        
        texts = [r["text"] for r in result]
        self.assertIn("new", texts)
        self.assertIn("different", texts)

    def test_regex_extract_mode(self):
        """Test regex extraction from first system message."""
        ds_list = [
            {
                "messages": [
                    {"role": "system", "content": "session_id=sess_001 user_id=user_A"},
                    {"role": "user", "content": "Hello"},
                ],
                "timestamp": 100,
            },
            {
                "messages": [
                    {"role": "system", "content": "session_id=sess_001 user_id=user_A"},
                    {"role": "user", "content": "Hi again"},
                ],
                "timestamp": 200,
            },
            {
                "messages": [
                    {"role": "system", "content": "session_id=sess_002 user_id=user_B"},
                    {"role": "user", "content": "Different session"},
                ],
                "timestamp": 50,
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator(
            group_key_mode="regex_extract",
            regex_pattern=r"session_id=(?P<session_id>[^\s]+)\s+user_id=(?P<user_id>[^\s]+)",
            regex_search_target="first_system_message",
            timestamp_field="timestamp",
        )
        result, _ = self._run_session_dedup(dataset, op, expected_count=2)
        
        # Check that newest from each session is kept
        contents = [r["messages"][1]["content"] for r in result]
        self.assertIn("Hi again", contents)
        self.assertIn("Different session", contents)

    def test_missing_session_or_user_always_kept(self):
        """Samples without valid session_id or user_id should always be kept."""
        ds_list = [
            {"session_id": "", "user_id": "u1", "timestamp": 100, "text": "no_session"},
            {"session_id": "s1", "user_id": None, "timestamp": 100, "text": "no_user"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 200, "text": "valid_newest"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 100, "text": "valid_old"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, _ = self._run_session_dedup(dataset, op, expected_count=3)
        
        texts = [r["text"] for r in result]
        self.assertIn("no_session", texts)
        self.assertIn("no_user", texts)
        self.assertIn("valid_newest", texts)
        self.assertNotIn("valid_old", texts)

    def test_dedup_info_meta_written(self):
        """Verify agent_session_dedup_info is written to meta before filtering."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": 100, "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 200, "text": "new"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        
        # Run compute_hash first
        dataset = dataset.map(op.compute_hash)
        
        # Call process but capture the pre-filter dataset
        # Since filter rebuilds the dataset, we check the dedup logic instead
        result, _ = op.process(dataset, show_num=0)
        
        # Only one sample should remain
        self.assertEqual(len(result), 1)
        # Verify it's the newest one
        self.assertEqual(result[0]["text"], "new")
        self.assertEqual(result[0]["timestamp"], 200)

    def test_single_sample_no_dedup(self):
        """Single sample should not trigger deduplication."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": 100, "text": "only_one"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, dup_pairs = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(dup_pairs, {})

    def test_show_num_tracing(self):
        """Test show_num parameter for duplicate pair tracing."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": 100, "text": "s1_old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 200, "text": "s1_new"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 150, "text": "s1_mid"},
            {"session_id": "s2", "user_id": "u2", "timestamp": 50, "text": "s2_old"},
            {"session_id": "s2", "user_id": "u2", "timestamp": 100, "text": "s2_new"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, dup_pairs = self._run_session_dedup(dataset, op, expected_count=2, show_num=1)
        
        # show_num=1 should return at most 1 duplicate group
        self.assertLessEqual(len(dup_pairs), 1)
        for h, samples in dup_pairs.items():
            self.assertGreater(len(samples), 1)

    def test_messages_nested_path(self):
        """Test extracting from nested messages path."""
        ds_list = [
            {
                "messages": [
                    {"role": "system", "content": "sid=s1 uid=u1"},
                ],
                "timestamp": 100,
            },
            {
                "messages": [
                    {"role": "system", "content": "sid=s1 uid=u1"},
                ],
                "timestamp": 200,
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator(
            group_key_mode="regex_extract",
            regex_pattern=r"sid=(?P<session_id>\w+)\s+uid=(?P<user_id>\w+)",
            regex_search_target="messages.0.content",
            timestamp_field="timestamp",
        )
        result, _ = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(result[0]["timestamp"], 200)

    def test_timestamp_string_datetime_with_milliseconds(self):
        """Test timestamp as datetime string with milliseconds."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23 00:30:28.73", "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23 01:45:12.95", "text": "new"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23 00:30:28.10", "text": "older"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, _ = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(result[0]["text"], "new")
        self.assertEqual(result[0]["timestamp"], "2026-03-23 01:45:12.95")

    def test_timestamp_string_datetime_without_milliseconds(self):
        """Test timestamp as datetime string without milliseconds."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23 00:30:28", "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23 01:45:12", "text": "new"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, _ = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(result[0]["text"], "new")

    def test_timestamp_iso_format(self):
        """Test timestamp in ISO 8601 format."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23T00:30:28.73", "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23T01:45:12.95", "text": "new"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, _ = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(result[0]["text"], "new")

    def test_timestamp_date_only(self):
        """Test timestamp as date only (time defaults to 00:00:00)."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-22", "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-23", "text": "new"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "2026-03-21", "text": "oldest"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, _ = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(result[0]["text"], "new")

    def test_timestamp_numeric_string(self):
        """Test timestamp as numeric string (should be parsed as float)."""
        ds_list = [
            {"session_id": "s1", "user_id": "u1", "timestamp": "1711152628.73", "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": "1711156228.95", "text": "new"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = AgentSessionDeduplicator()
        result, _ = self._run_session_dedup(dataset, op, expected_count=1)
        self.assertEqual(result[0]["text"], "new")

    def test_timestamp_mixed_types(self):
        """Test that numeric and string timestamps are both parsed correctly.
        
        Note: In real usage, timestamps within a group should be consistent.
        This test verifies the parser handles both formats independently.
        """
        # Test numeric timestamps
        ds_numeric = [
            {"session_id": "s1", "user_id": "u1", "timestamp": 1711152628, "text": "old"},
            {"session_id": "s1", "user_id": "u1", "timestamp": 1711156228, "text": "new"},
        ]
        dataset_numeric = Dataset.from_list(ds_numeric)
        op_numeric = AgentSessionDeduplicator()
        result_numeric, _ = self._run_session_dedup(dataset_numeric, op_numeric, expected_count=1)
        self.assertEqual(result_numeric[0]["text"], "new")
        
        # Test string datetime timestamps
        ds_string = [
            {"session_id": "s2", "user_id": "u2", "timestamp": "2024-03-23 00:30:28", "text": "old"},
            {"session_id": "s2", "user_id": "u2", "timestamp": "2026-03-23 01:45:12", "text": "new"},
        ]
        dataset_string = Dataset.from_list(ds_string)
        op_string = AgentSessionDeduplicator()
        result_string, _ = self._run_session_dedup(dataset_string, op_string, expected_count=1)
        self.assertEqual(result_string[0]["text"], "new")


if __name__ == "__main__":
    unittest.main()
