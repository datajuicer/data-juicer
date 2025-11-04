import os
from typing import Dict, Optional, Sequence, Union

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

mmpose = LazyLoader("mmpose")
mmdet = LazyLoader("mmdet==3.2.0")

OP_NAME = "image_mmpose_mapper"


@OPERATORS.register_module(OP_NAME)
class ImageMMPoseMapper(Mapper):
    """Mapper to perform human keypoint detection inference using MMPose models.
    It requires three essential components for model initialization:
    - deploy_cfg (str): Path to the deployment configuration file (defines inference settings)
    - model_cfg (str): Path to the model configuration file (specifies model architecture)
    - model_files (List[str]): Model weight files including pre-trained weights and parameters

    The implementation follows the official MMPose deployment guidelines from MMDeploy.
    For detailed configuration requirements and usage examples, refer to:
    https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md

    """

    _accelerator = "cuda"

    def __init__(
        self,
        deploy_cfg: str = None,
        model_cfg: str = None,
        model_files: Optional[Union[str, Sequence[str]]] = None,
        pose_key: str = MetaKeys.pose_info,
        visualization_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param deploy_cfg: MMPose deployment config file.
        :param model_cfg: MMPose model config file.
        :param model_files: Path to the model weight files.
        :param pose_key: Key to store pose information.
        :param visualization_dir: Directory to save visualization results.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pose_key = pose_key

        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        if isinstance(model_files, str):
            self.model_files = [model_files]
        else:
            self.model_files = model_files

        self.visualization_dir = visualization_dir

        import mmdet  # noqa: F401

        self.model_key = prepare_model(
            "mmlab",
            model_cfg=self.model_cfg,
            deploy_cfg=self.deploy_cfg,
            model_files=self.model_files,
        )

    def parse_and_filter(self, data_sample: mmpose.structures.PoseDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`PoseDataSample`): Predictions of the model.

        Returns:
            dict: Prediction results.
        """
        result = {
            "keypoints": [],
            "keypoint_scores": [],
            "bboxes": [],
            "bbox_scores": [],
        }
        if "pred_instances" in data_sample:
            if "keypoints" in data_sample.pred_instances:
                result["keypoints"] = data_sample.pred_instances.keypoints
            if "keypoint_scores" in data_sample.pred_instances:
                result["keypoint_scores"] = data_sample.pred_instances.keypoint_scores
            if "bboxes" in data_sample.pred_instances:
                result["bboxes"] = data_sample.pred_instances.bboxes
            if "bbox_scores" in data_sample.pred_instances:
                result["bbox_scores"] = data_sample.pred_instances.bbox_scores

        return result

    def visualize_results(self, image, model, result, output_file):
        model.task_processor.visualize(
            image=image, model=model, result=result[0], window_name="visualize", output_file=output_file
        )

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.pose_key in sample[Fields.meta]:
            return sample

        model = get_model(self.model_key, rank, self.use_cuda())
        images = sample[self.image_key]

        from mmpose.apis.inference import dataset_meta_from_config

        dataset_meta = dataset_meta_from_config(model.task_processor.model_cfg, dataset_mode="test")
        keypoint_names = [dataset_meta["keypoint_id2name"][i] for i in range(dataset_meta["num_keypoints"])]

        results = [model(img) for img in images]
        pose_info = [self.parse_and_filter(res[0]) for res in results]
        for pinfo in pose_info:
            pinfo["keypoint_names"] = keypoint_names

        sample[Fields.meta][self.pose_key] = pose_info

        if self.visualization_dir:
            os.makedirs(self.visualization_dir, exist_ok=True)
            for i, img in enumerate(images):
                img_name = os.path.splitext(os.path.basename(img))[0]
                output_file = f"{self.visualization_dir}/{img_name}.png"
                self.visualize_results(img, model, results[i], output_file)

        return sample
