from pathlib import Path

from data_juicer.utils.robot_utils import episode_root

from ..base_op import OPERATORS, Mapper

OP_NAME = "robot_gripper_anchor_mapper"


@OPERATORS.register_module(OP_NAME)
class RobotGripperAnchorMapper(Mapper):
    """Extract an anchor on the gripper-state timeline."""

    def __init__(self, output_dir: str = "output/episodes", *args, **kwargs):
        """Initialize the mapper.

        :param output_dir: Directory for episode artifacts.
        :param args: Additional positional arguments for Mapper.
        :param kwargs: Additional keyword arguments for Mapper.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir

    def process_single(self, sample):
        from data_juicer.utils.robot_utils import (
            extract_gripper_anchor,
            read_instruction,
        )

        root = episode_root(sample, self.output_dir)
        if sample["source"] != "droid":
            raise ValueError("Gripper signal anchors require a DROID sample")
        (instruction, column) = read_instruction(Path(sample["metadata_path"]))
        extract_gripper_anchor(Path(sample["state_path"]), int(sample["episode_index"]), root / "ir/phase_anchor.json")
        sample["instruction"] = instruction
        sample["instruction_source_column"] = column
        return sample
