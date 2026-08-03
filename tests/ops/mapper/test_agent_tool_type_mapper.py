# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.agent_tool_type_mapper import AgentToolTypeMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AgentToolTypeMapperTest(DataJuicerTestCaseBase):
    def test_sets_empty_fields_when_meta_or_tool_list_is_missing(self):
        op = AgentToolTypeMapper()

        out = op.process_single({})
        self.assertIn(Fields.meta, out)
        self.assertIsNone(out[Fields.meta][MetaKeys.primary_tool_type])
        self.assertEqual(out[Fields.meta][MetaKeys.dominant_tool_types], [])

        out = op.process_single({Fields.meta: {MetaKeys.agent_tool_types: "search"}})
        self.assertIsNone(out[Fields.meta][MetaKeys.primary_tool_type])
        self.assertEqual(out[Fields.meta][MetaKeys.dominant_tool_types], [])

    def test_counts_tools_and_strips_blank_values(self):
        op = AgentToolTypeMapper(top_k_dominant=2)
        sample = {
            Fields.meta: {
                MetaKeys.agent_tool_types: [
                    " search ",
                    "python",
                    "search",
                    "",
                    None,
                    "browser",
                    "python",
                    "search",
                ],
            },
        }

        out = op.process_single(sample)

        self.assertEqual(out[Fields.meta][MetaKeys.primary_tool_type], "search")
        self.assertEqual(
            out[Fields.meta][MetaKeys.dominant_tool_types],
            ["search", "python"],
        )


if __name__ == "__main__":
    unittest.main()
