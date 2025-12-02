import os
from typing import Dict, List, Optional

from loguru import logger

from data_juicer.ops.mapper.annotation.annotation_mapper import (
    LabelStudioAnnotationMapper,
)

from ...base_op import OPERATORS


@OPERATORS.register_module("human_preference_annotation_mapper")
class HumanPreferenceAnnotationMapper(LabelStudioAnnotationMapper):
    """Operator for human preference annotation using Label Studio.

    This operator formats and presents pairs of answers to a prompt for human evaluation. It
    uses a default or custom Label Studio configuration to display the prompt and answer
    options. The operator processes the annotations to determine the preferred answer,
    updating the sample with the chosen and rejected answers. The operator requires specific
    keys in the samples for the prompt and answer options. If these keys are missing, it
    logs warnings and uses placeholder text. The annotated results are processed to update
    the sample with the chosen and rejected answers."""

    DEFAULT_LABEL_CONFIG = """
    <View className="root">
      <Style>
        .root {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
          font-family: 'Roboto',
            sans-serif;
          line-height: 1.6;
          background-color: #f0f0f0;
        }

        .container {
          margin: 0 auto;
          padding: 20px;
          background-color: #ffffff;
          border-radius: 5px;
          box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.1);
        }

        .prompt {
          padding: 20px;
          background-color: #0084ff;
          color: #ffffff;
          border-radius: 5px;
          margin-bottom: 20px;
          box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 3px 10px 0 rgba(0, 0, 0, 0.1);
        }

        .answers {
          display: flex;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 20px;
        }

        .answer-box {
          flex-basis: 49%;
          padding: 20px;
          background-color: rgba(44, 62, 80, 0.9);
          color: #ffffff;
          border-radius: 5px;
          box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 3px 10px 0 rgba(0, 0, 0, 0.1);
        }

        .answer-box p {
          word-wrap: break-word;
        }

        .answer-box:hover {
          background-color: rgba(52, 73, 94, 0.9);
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .lsf-richtext__line:hover {
          background: unset;
        }

        .answer-box .lsf-object {
          padding: 20px
        }
      </Style>
      <View className="container">
        <View className="prompt">
          <Text name="prompt" value="$prompt" />
        </View>
        <View className="answers">
          <Pairwise name="comparison" toName="answer1,answer2"
                    selectionStyle="background-color: #27ae60; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.2); border: 2px solid #2ecc71; cursor: pointer; transition: all 0.3s ease;"
                    leftChoiceValue="answer1" rightChoiceValue="answer2" />
          <View className="answer-box">
            <Text name="answer1" value="$answer1" />
          </View>
          <View className="answer-box">
            <Text name="answer2" value="$answer2" />
          </View>
        </View>
      </View>
    </View>
    """  # noqa: E501

    def __init__(
        self,
        label_config_file: str = None,
        answer1_key: str = "answer1",
        answer2_key: str = "answer2",
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        **kwargs,
    ):
        """
        Initialize the human preference annotation operator.

        :param label_config_file: Path to the label config file
        :param answer1_key: Key for the first answer
        :param answer2_key: Key for the second answer
        :param prompt_key: Key for the prompt/question
        :param chosen_key: Key for the chosen answer
        :param rejected_key: Key for the rejected answer
        """
        # Ensure text_key is set to prompt_key if not explicitly provided
        if "text_key" not in kwargs:
            kwargs["text_key"] = prompt_key

        super().__init__(**kwargs)
        # Store our class-specific attributes
        self.answer1_key = answer1_key
        self.answer2_key = answer2_key
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        # Prepare the label_config parameter
        if label_config_file and os.path.exists(label_config_file):
            with open(label_config_file, "r") as f:
                kwargs["label_config"] = f.read().strip()
                logger.info(f"Loaded label config from {label_config_file}")
        else:
            kwargs["label_config"] = self.DEFAULT_LABEL_CONFIG.strip()
            logger.info("Using default UI config for human preference annotation")

    def _format_task(self, samples: List[Dict]) -> Dict:
        """Format samples as a Label Studio task for human preference.

        Args:
            samples: List of samples to include in the task

        Returns:
            Dict: Formatted task data
        """
        # For human preference, we need a special format
        if len(samples) != 1:
            logger.warning("Human preference requires exactly one sample per task")

        sample = samples[0]
        task = {"data": {}}

        # Add the prompt/question
        if self.prompt_key in sample:
            task["data"]["prompt"] = sample[self.prompt_key]
        else:
            logger.warning(f"Sample missing required field '{self.prompt_key}'")
            task["data"]["prompt"] = "No prompt provided"

        # Add the answer options
        if self.answer1_key in sample:
            task["data"]["answer1"] = sample[self.answer1_key]
        else:
            logger.warning(f"Sample missing required field '{self.answer1_key}'")
            task["data"]["answer1"] = "No answer 1 provided"

        if self.answer2_key in sample:
            task["data"]["answer2"] = sample[self.answer2_key]
        else:
            logger.warning(f"Sample missing required field '{self.answer2_key}'")
            task["data"]["answer2"] = "No answer 2 provided"

        # Add any other metadata as string values only
        for key, value in sample.items():
            if key not in [self.prompt_key, self.answer1_key, self.answer2_key]:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    # Convert to string to ensure compatibility
                    task["data"][f"meta:{key}"] = str(value) if value is not None else ""

        # Log the task for debugging
        logger.debug(f"Formatted task: {task}")

        return task

    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available"""
        annotation = super()._get_task_annotation(task_id)

        # Process the annotation if available
        if annotation and "result" in annotation:
            # Extract the preference information
            for item in annotation["result"]:
                if item.get("type") == "pairwise":
                    # Get the selected option (from_id or to_id)
                    selected = item.get("value", {}).get("selected")
                    if selected:
                        # Add the preference to the annotation
                        annotation["preference"] = selected

        return annotation

    def _process_annotation_result(self, annotation: Dict, sample: Dict) -> Dict:
        """Process human preference annotation result and update the sample

        Args:
            annotation: The annotation result from Label Studio
            sample: The original sample that was annotated

        Returns:
            Dict: The updated sample with preference results
        """
        # Extract the preference information
        logger.debug(f"Processing annotation result: {annotation}")

        all_keys = f"{self.answer1_key}{self.answer2_key}"
        preference = None
        for item in annotation["result"]:
            if item.get("type") == "pairwise":
                # Get the selected option
                selected = item.get("value", {}).get("selected")
                if selected:
                    # Map 'left'/'right' to 'answer1'/'answer2'
                    if selected == "left":
                        preference = self.answer1_key
                    elif selected == "right":
                        preference = self.answer2_key
                    else:
                        # In case it's already 'answer1'/'answer2'
                        preference = selected
                    break

        # Store the preference result directly in the sample
        chosen = preference if preference else "Unanswered"
        rejected = all_keys.replace(preference, "") if preference else "Unanswered"
        sample[self.chosen_key] = sample[chosen]
        sample[self.rejected_key] = sample[rejected]

        logger.debug(f"Updated sample: {sample}")
        return sample
