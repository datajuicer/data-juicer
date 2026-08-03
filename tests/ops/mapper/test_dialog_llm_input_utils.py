# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.dialog_llm_input_utils import build_dialog_turns_for_prompt
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestBuildDialogTurnsForPrompt(DataJuicerTestCaseBase):
    def test_dedupes_last_turn_when_same_as_query_response(self):
        sample = {
            "dialog_history": [("u1", "a1"), ("u2", "a2")],
            "query": "u2",
            "response": "a2",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1"), ("u2", "a2")])

    def test_does_not_mutate_sample_dialog_history(self):
        hist = [("u1", "a1")]
        sample = {
            "dialog_history": hist,
            "query": "q",
            "response": "r",
        }
        build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(hist, [("u1", "a1")])

    def test_appends_when_query_differs_from_last_user(self):
        sample = {
            "dialog_history": [("u1", "a1")],
            "query": "u2",
            "response": "a2",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1"), ("u2", "a2")])


if __name__ == "__main__":
    unittest.main()
