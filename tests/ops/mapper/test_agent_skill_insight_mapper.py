import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.agent_skill_insight_mapper import AgentSkillInsightMapper
from data_juicer.utils.constant import DEFAULT_API_MODEL, Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork


class AgentSkillInsightMapperEdgeCaseTest(DataJuicerTestCaseBase):

    def test_empty_tools_and_skills(self):
        op = AgentSkillInsightMapper(api_model='fake')
        sample = {
            'text': 'test',
            Fields.meta: {
                MetaKeys.agent_tool_types: [],
                MetaKeys.agent_skill_types: [],
            }
        }
        result = op.process_single(sample)
        # When both tools and skills are empty, insights should be empty
        self.assertEqual(
            result[Fields.meta][MetaKeys.agent_skill_insights], [])

    def test_missing_tool_types(self):
        op = AgentSkillInsightMapper(api_model='fake')
        sample = {
            'text': 'test',
            Fields.meta: {}
        }
        result = op.process_single(sample)
        # meta.get() returns None, which falls through to empty list
        self.assertEqual(
            result[Fields.meta][MetaKeys.agent_skill_insights], [])

    def test_missing_meta(self):
        op = AgentSkillInsightMapper(api_model='fake')
        sample = {'text': 'test'}
        result = op.process_single(sample)
        # No meta dict at all -> sample returned unchanged, no insights key
        self.assertNotIn(Fields.meta, result)

    def test_non_dict_meta(self):
        op = AgentSkillInsightMapper(api_model='fake')
        sample = {'text': 'test', Fields.meta: 'not_a_dict'}
        result = op.process_single(sample)
        # Non-dict meta -> sample returned unchanged
        self.assertEqual(result[Fields.meta], 'not_a_dict')

    def test_already_generated_skip(self):
        op = AgentSkillInsightMapper(api_model='fake')
        existing = ['existing_insight_1', 'existing_insight_2']
        sample = {
            'text': 'test',
            Fields.meta: {
                MetaKeys.agent_skill_insights: existing,
                MetaKeys.agent_tool_types: ['code_interpreter'],
                MetaKeys.agent_skill_types: ['problem_solving'],
            }
        }
        result = op.process_single(sample)
        # insights_key already in meta -> skip, keep existing value
        self.assertEqual(
            result[Fields.meta][MetaKeys.agent_skill_insights], existing)

    def test_string_tools_and_skills_coerced(self):
        """Non-list tool/skill values should be coerced to single-element lists."""
        op = AgentSkillInsightMapper(api_model='fake')
        sample = {
            'text': 'test',
            Fields.meta: {
                MetaKeys.agent_tool_types: 'web_search',
                MetaKeys.agent_skill_types: 'reasoning',
            }
        }
        # Even with non-empty string values, the code coerces them to lists.
        # But calling LLM with 'fake' model will fail, so insights will be [].
        result = op.process_single(sample)
        self.assertIn(MetaKeys.agent_skill_insights, result[Fields.meta])

    def test_custom_keys(self):
        op = AgentSkillInsightMapper(
            api_model='fake',
            tool_types_key='my_tools',
            skill_types_key='my_skills',
            insights_key='my_insights',
        )
        sample = {
            'text': 'test',
            Fields.meta: {
                'my_tools': [],
                'my_skills': [],
            }
        }
        result = op.process_single(sample)
        # Custom insights key should be used
        self.assertEqual(result[Fields.meta]['my_insights'], [])


@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class AgentSkillInsightMapperTest(DataJuicerTestCaseBase):

    def _run_op(self, op, tools, skills):
        samples = [{
            'text': 'test conversation',
            Fields.meta: {
                MetaKeys.agent_tool_types: tools,
                MetaKeys.agent_skill_types: skills,
            }
        }]
        dataset = Dataset.from_list(samples)
        result = op.run(dataset)
        return result[0]

    def test_basic(self):
        op = AgentSkillInsightMapper(
            api_model=DEFAULT_API_MODEL,
            sampling_params={'enable_thinking': False},
        )
        sample = self._run_op(
            op,
            ['code_interpreter', 'web_search'],
            ['problem_solving'],
        )
        self.assertIn(MetaKeys.agent_skill_insights, sample[Fields.meta])
        self.assertGreater(
            len(sample[Fields.meta][MetaKeys.agent_skill_insights]), 0)
        logger.info(
            f"insights: {sample[Fields.meta][MetaKeys.agent_skill_insights]}")

    def test_custom_key(self):
        op = AgentSkillInsightMapper(
            api_model=DEFAULT_API_MODEL,
            insights_key='my_insights',
            sampling_params={'enable_thinking': False},
        )
        sample = self._run_op(op, ['calculator'], ['math'])
        self.assertIn('my_insights', sample[Fields.meta])
        logger.info(f"custom insights: {sample[Fields.meta]['my_insights']}")


if __name__ == '__main__':
    unittest.main()
