import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional

from data_juicer.ops.mapper.annotation.annotation_mapper import BaseAnnotationMapper
from data_juicer.ops.mapper.annotation.human_preference_annotation_mapper import HumanPreferenceAnnotationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

import tempfile
from contextlib import contextmanager

@contextmanager
def temp_label_config():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml') as f:
        f.write(HumanPreferenceAnnotationMapper.DEFAULT_LABEL_CONFIG)
        f.flush()
        yield f.name

class MockLabelStudioClientClass:
    def __init__(self, project_id=999, base_task_id=3000):
        self.project_id = project_id
        self.base_task_id = base_task_id
        self.tasks = []
        self.task_ids = []

    def _create_mock_project(self):
        project = MagicMock()
        project.id = self.project_id
        project.get_tasks.return_value = self.tasks
        project.get_task.side_effect = self._get_task
        project.import_tasks.side_effect = self._import_tasks
        return project

    def get_project(self, *args, **kwargs):
        return self._create_mock_project()

    def create_project(self, *args, **kwargs):
        return self._create_mock_project()

    def _get_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                return task

    def _import_tasks(self, tasks):
        task_ids = []
        if self.task_ids is not None:
            task_ids = self.task_ids
        else:
            for i in range(len(tasks)):
                task_id = self.base_task_id + i
                task_ids.append(task_id)
            self.tasks_ids = task_ids
        return task_ids

    def add_annotation(self, task_id, selected="left"):
        """Add a mock annotation for a task"""
        task = {"id": task_id, "annotations": [{"result": [{"type": "pairwise", "value": {"selected": selected}}]}]}
        self.tasks.append(task)

    def setup_for_batch(self, selected_list, task_ids: Optional[List] = None):
        """Setup mock data for batch processing"""
        self.tasks = []
        self.task_ids = task_ids
        for i, selected in enumerate(selected_list):
            if task_ids is not None:
                task_id = task_ids[i]
            else:
                task_id = self.base_task_id + i
            self.add_annotation(task_id, selected)


class HumanPreferenceAnnotationMapperTest(DataJuicerTestCaseBase):
    """Test cases for HumanPreferenceAnnotationMapper"""

    def setUp(self):
        super().setUp()

        # Create samples for testing human preference
        self.samples = [
            {
                "prompt": "Which response is more helpful?",
                "answer1": "The capital of France is Paris.",
                "answer2": "Paris is the capital and largest city of France, located on the Seine River.",
                "id": "pref_sample1",
            },
            {
                "prompt": "Which explanation is clearer?",
                "answer1": "To create a list in Python, use square brackets.",
                "answer2": "In Python, you can create a list using square brackets []. Lists can contain items of different types.",
                "id": "pref_sample2",
            },
        ]

        # Create a dictionary version of samples (column-oriented)
        self.samples_dict = {
            "prompt": [s["prompt"] for s in self.samples],
            "answer1": [s["answer1"] for s in self.samples],
            "answer2": [s["answer2"] for s in self.samples],
            "id": [s["id"] for s in self.samples],
        }

    @patch("label_studio_sdk.Client")
    def test_init_parameters(self, MockLabelStudioClient):
        """Test initialization with custom parameters"""
        mock_client = MockLabelStudioClientClass()
        MockLabelStudioClient.return_value = mock_client
        
        with temp_label_config() as config_path:
            mapper = HumanPreferenceAnnotationMapper(
                label_config_file=config_path,
                answer1_key="response_a",
                answer2_key="response_b",
                prompt_key="question",
                chosen_key="chosen",
                rejected_key="rejected",
            )

        self.assertEqual(mapper.answer1_key, "response_a")
        self.assertEqual(mapper.answer2_key, "response_b")
        self.assertEqual(mapper.prompt_key, "question")
        self.assertEqual(mapper.chosen_key, "chosen")
        self.assertEqual(mapper.rejected_key, "rejected")
        self.assertEqual(mapper.text_key, "question")  # text_key should match prompt_key

    @patch("label_studio_sdk.Client")
    def test_format_task(self, MockLabelStudioClient):
        """Test task formatting for human preference"""
        mock_client = MockLabelStudioClientClass()
        MockLabelStudioClient.return_value = mock_client

        mapper = HumanPreferenceAnnotationMapper()

        # Format a task from the first sample
        formatted_task = mapper._format_task(self.samples)

        # Verify the formatting
        self.assertIn("data", formatted_task)
        self.assertEqual(formatted_task["data"]["prompt"], self.samples[0]["prompt"])
        self.assertEqual(formatted_task["data"]["answer1"], self.samples[0]["answer1"])
        self.assertEqual(formatted_task["data"]["answer2"], self.samples[0]["answer2"])
        self.assertEqual(formatted_task["data"]["meta:id"], self.samples[0]["id"])

    @patch("label_studio_sdk.Client")
    def test_format_task_missing_keys(self, MockLabelStudioClient):
        """Test task formatting with missing keys"""
        MockLabelStudioClient.return_value = MockLabelStudioClientClass()

        mapper = HumanPreferenceAnnotationMapper(
            answer1_key="response_a",
            answer2_key="response_b",
            prompt_key="question",
        )
        # Format a task from the first sample
        formatted_task = mapper._format_task([self.samples[0]])
        # Verify the formatting
        self.assertIn("data", formatted_task)
        self.assertEqual(formatted_task["data"]["prompt"], "No prompt provided")
        self.assertEqual(formatted_task["data"]["answer1"], "No answer 1 provided")
        self.assertEqual(formatted_task["data"]["answer2"], "No answer 2 provided")

    @patch("label_studio_sdk.Client")
    def test_process_annotation_result_left_preference(self, MockLabelStudioClient):
        """Test processing annotation result when left option is preferred"""
        mock_client = MockLabelStudioClientClass()
        MockLabelStudioClient.return_value = mock_client

        mapper = HumanPreferenceAnnotationMapper()

        # Create a sample
        sample = self.samples[0].copy()

        # Create an annotation with preference for the left option (answer1)
        annotation = {"id": "annotation_1", "result": [{"type": "pairwise", "value": {"selected": "left"}}]}

        # Process the annotation
        processed_sample = mapper._process_annotation_result(annotation, sample)

        # Verify the result
        self.assertEqual(processed_sample["chosen"], processed_sample["answer1"])

    @patch("label_studio_sdk.Client")
    def test_process_annotation_result_right_preference(self, MockLabelStudioClient):
        """Test processing annotation result when right option is preferred"""
        mock_client = MockLabelStudioClientClass()
        MockLabelStudioClient.return_value = mock_client

        mapper = HumanPreferenceAnnotationMapper()
        # Create a sample
        sample = self.samples[0].copy()

        # Create an annotation with preference for the right option (answer2)
        annotation = {"id": "annotation_1", "result": [{"type": "pairwise", "value": {"selected": "right"}}]}

        # Process the annotation
        processed_sample = mapper._process_annotation_result(annotation, sample)

        # Verify the result
        self.assertEqual(processed_sample["chosen"], processed_sample["answer2"])

    @patch("label_studio_sdk.Client")
    def test_process_batched(self, MockLabelStudioClient):
        """Test processing a batch of samples with HumanPreferenceAnnotationMapper"""

        # Create and setup mock client
        mock_client = MockLabelStudioClientClass()
        mock_client.setup_for_batch(["left" if i % 2 == 0 else "right" for i in range(len(self.samples))])
        MockLabelStudioClient.return_value = mock_client

        mapper = HumanPreferenceAnnotationMapper(wait_for_annotations=True)
        # Process the samples
        result = mapper.process_batched(self.samples_dict)

        # Verify results
        self.assertEqual(len(result["prompt"]), len(self.samples_dict["prompt"]))
        self.assertEqual(len(result["id"]), len(self.samples_dict["id"]))

        # Verify that results are properly added
        self.assertIn("chosen", result)

        # First sample should prefer answer1 (left choice)
        self.assertEqual(result["chosen"][0], result["answer1"][0])

        # Second sample should prefer answer2 (right choice)
        self.assertEqual(result["chosen"][1], result["answer2"][1])

    @patch("label_studio_sdk.Client")
    def test_custom_keys(self, MockLabelStudioClient):
        """Test using custom keys for answers and prompt"""
        # Create a sample with custom keys
        custom_sample = {
            "question": "Which explanation is better?",
            "response_a": "Explanation A",
            "response_b": "Explanation B",
            "id": "custom_sample",
        }

        # Create a dictionary version of the sample
        custom_dict = {
            "question": [custom_sample["question"]],
            "response_a": [custom_sample["response_a"]],
            "response_b": [custom_sample["response_b"]],
            "id": [custom_sample["id"]],
        }

        # Create and setup mock client
        mock_client = MockLabelStudioClientClass()
        mock_client.setup_for_batch(["left"], task_ids=custom_dict["id"])
        MockLabelStudioClient.return_value = mock_client

        # Create mapper with custom keys
        mapper = HumanPreferenceAnnotationMapper(
            answer1_key="response_a",
            answer2_key="response_b",
            prompt_key="question",
            chosen_key="chosen",
            rejected_key="rejected",
            wait_for_annotations=True,
        )
        # Process the sample
        result = mapper.process_batched(custom_dict)

        # Verify the results
        self.assertIn("chosen", result)
        self.assertEqual(result["chosen"][0], result["response_a"][0])

    @patch("label_studio_sdk.Client")
    def test_process_uses_existing_ids(self, MockLabelStudioClient):
        """Test that the Human Preference mapper uses existing IDs in samples instead of generating new ones"""
        # Create samples with predefined IDs
        samples_with_ids = {
            "prompt": ["Which is better? A or B", "Which is clearer? X or Y"],
            "answer1": ["Option A", "Option X"],
            "answer2": ["Option B", "Option Y"],
            "id": ["preference_id_1", "preference_id_2"],
        }

        mock_client = MockLabelStudioClientClass()
        mock_client.setup_for_batch(
            ["left" if i % 2 == 0 else "right" for i in range(len(samples_with_ids["id"]))],
            task_ids=samples_with_ids["id"],
        )
        MockLabelStudioClient.return_value = mock_client

        # First pass: process without waiting for annotations
        mapper = HumanPreferenceAnnotationMapper(wait_for_annotations=False)

        # Process the samples
        result = mapper.process_batched(samples_with_ids)

        # Verify that the predefined IDs were used in the mapping
        for i, sample_id in enumerate(samples_with_ids["id"]):
            # Check if each predefined ID is in the sample-to-task mapping
            self.assertIn(sample_id, mapper.sample_to_task_id)

            # Get the task ID for this sample
            task_id = mapper.sample_to_task_id[sample_id]

            # Verify that the sample ID is in the task's sample list
            self.assertIn(sample_id, mapper.task_to_samples[task_id])

        # Second pass: process with waiting for annotations
        mapper.wait_for_annotations = True
        result = mapper.process_batched(samples_with_ids)

        # Verify results include preference annotations
        self.assertIn("chosen", result)

        # First sample should prefer answer1 (left choice)
        self.assertEqual(result["chosen"][0], result["answer1"][0])

        # Second sample should prefer answer2 (right choice)
        self.assertEqual(result["chosen"][1], result["answer2"][1])

        # Verify the original IDs were preserved in the result
        self.assertEqual(result["id"], samples_with_ids["id"])


if __name__ == "__main__":
    unittest.main()
