from data_juicer.utils.robot_utils import (
    episode_root,
    read_json,
    vlm_settings,
    write_json,
)

from ..base_op import OPERATORS, Mapper

OP_NAME = "robot_destination_grounding_mapper"


@OPERATORS.register_module(OP_NAME)
class RobotDestinationGroundingMapper(Mapper):
    """Select a visible final view and ground the demonstrated target location."""

    def __init__(self, model: str = None, endpoint: str = None, output_dir: str = "output/episodes", *args, **kwargs):
        """Initialize the mapper.

        :param model: Served VLM model name. Defaults to VLM_MODEL.
        :param endpoint: OpenAI-compatible VLM endpoint. Defaults to VLM_URL.
        :param output_dir: Directory for episode artifacts.
        :param args: Additional positional arguments for the Mapper.
        :param kwargs: Additional keyword arguments for the Mapper.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        (self.model, self.endpoint) = (model, endpoint)

    def process_single(self, sample):
        from data_juicer.utils.robot_perception_utils import ground_final_target

        root = episode_root(sample, self.output_dir)
        target = read_json(sample["object_outputs"][0])["vlm_target_object"]["target_object"]
        views = read_json(root / "ir/sampling.json")["views"]
        (model, endpoint) = vlm_settings(self.model, self.endpoint)
        grounding = ground_final_target(root, views, target, model, endpoint)
        write_json(root / "ir/destination.json", grounding)
        return sample
