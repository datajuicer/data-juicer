from data_juicer.utils.robot_utils import episode_root, read_json, write_json

from ..base_op import OPERATORS, Mapper

OP_NAME = "robot_interaction_geometry_mapper"


@OPERATORS.register_module(OP_NAME)
class RobotInteractionGeometryMapper(Mapper):
    """Compose tracked geometry, state anchors and synchronized views into episode IR."""

    def __init__(self, output_dir: str = "output/episodes", *args, **kwargs):
        """Initialize the mapper.

        :param output_dir: Directory for episode artifacts.
        :param args: Additional positional arguments for Mapper.
        :param kwargs: Additional keyword arguments for Mapper.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir

    def process_single(self, sample):
        from data_juicer.utils.robot_utils import overlap_ratio

        root = episode_root(sample, self.output_dir)
        views = read_json(root / "ir/sampling.json")["views"]
        primary = sample["primary_view"]
        if any((v["frame_indices"] != views[primary]["frame_indices"] for v in views.values())):
            raise ValueError("Cross-view frame indices are not synchronized")
        obj = read_json(sample["object_outputs"][1])
        gripper = read_json(sample["gripper_output"])
        objects = {int(f["frame_index"]): f["bbox_xyxy"] for f in obj["frames"] if f.get("bbox_xyxy")}
        grippers = {int(f["frame_index"]): f["bbox_xyxy"] for f in gripper["tracking"]["frames"] if f.get("bbox_xyxy")}
        joint = sorted(set(objects) & set(grippers))
        if len(joint) < 3:
            raise ValueError("Insufficient joint tracks")
        contact = max(joint, key=lambda i: overlap_ratio(objects[i], grippers[i]))
        ratio = overlap_ratio(objects[contact], grippers[contact])
        if ratio < 0.05:
            raise ValueError("No geometric contact frame")
        destination = read_json(root / "ir/destination.json")
        geometry = {
            "contact_sample_index": contact,
            "contact_overlap_ratio": ratio,
            "destination_sample_index": destination["sample_index"],
            "final_target_grounding": destination,
            "secondary_view": next((v for v in sorted(views) if v != primary)),
        }
        ir = {
            "episode_id": sample["episode_id"],
            "instruction": sample["instruction"],
            "instruction_source_column": sample["instruction_source_column"],
            "phase": read_json(root / "ir/phase_anchor.json"),
            "primary_view": primary,
            "sampled_views": views,
            "object_operator_outputs": sample["object_outputs"],
            "gripper_operator_output": sample["gripper_output"],
            "geometry": geometry,
        }
        if sample.get("depth_ir"):
            from data_juicer.utils.robot_depth_utils import tracked_depth

            depth = read_json(sample["depth_ir"])
            ir["depth"] = depth
            geometry["tracked_depth"] = tracked_depth(
                root, depth, primary, views[primary], {"object": objects, "gripper": grippers}
            )
        write_json(root / "ir/episode_ir.json", ir)
        return sample
