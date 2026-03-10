import json
import os

import numpy as np
from pydantic import PositiveInt

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_calibration_static_deepcalib_mapper"

cv2 = LazyLoader("cv2", "opencv-python")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraCalibrationStaticDeepcalibMapper(Mapper):
    """Compute the camera intrinsics and field of view (FOV)
    for a static camera using DeepCalib."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "weights_10_0.02.h5",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        tag_field_name: str = MetaKeys.static_camera_calibration_deepcalib_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_output_info: bool = True,
        output_info_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the DeepCalib Regression model.
        :param frame_num: The number of frames to be extracted uniformly from
            the video. If it's 1, only the middle frame will be extracted. If
            it's 2, only the first and the last frames will be extracted. If
            it's larger than 2, in addition to the first and the last frames,
            other frames will be extracted uniformly within the video duration.
            If "duration" > 0, frame_num is the number of frames per segment.
        :param duration: The duration of each segment in seconds.
            If 0, frames are extracted from the entire video.
            If duration > 0, the video is segmented into multiple segments
            based on duration, and frames are extracted from each segment.
        :param tag_field_name: The field name to store the tags. It's
            "static_camera_calibration_deepcalib_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_output_info: Whether to save the camera parameters results
            to an JSON file.
        :param output_info_dir: Output directory for saving camera parameters.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        LazyLoader.check_packages(["tensorflow"])
        import keras
        from keras.applications.imagenet_utils import preprocess_input

        self.keras = keras
        self.preprocess_input = preprocess_input

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "frame_key": MetaKeys.video_frames,
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])
        self.model_key = prepare_model(model_type="deepcalib", model_path=model_path)

        self.frame_num = frame_num
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.if_output_info = if_output_info
        self.output_info_dir = output_info_dir
        self.INPUT_SIZE = 299
        self.focal_start = 40
        self.focal_end = 500

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # load videos
        ds_list = [{"text": SpecialTokens.video, "videos": sample[self.video_key]}]

        dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
        dataset = self.fused_ops[0].run(dataset)

        frames_root = os.path.join(self.frame_dir, os.path.splitext(os.path.basename(sample[self.video_key][0]))[0])
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])
        model = get_model(self.model_key, rank, self.use_cuda())

        final_k_list = []
        final_xi_list = []
        final_hfov_list = []
        final_vfov_list = []

        for i, path in enumerate(frames_path):
            image = cv2.imread(path)
            height, width, channels = image.shape

            image = cv2.resize(image, (self.INPUT_SIZE, self.INPUT_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = image - 0.5
            image = image * 2.0
            image = np.expand_dims(image, 0)

            image = self.preprocess_input(image)

            prediction = model.predict(image)
            prediction_focal = prediction[0]
            prediction_dist = prediction[1]

            # Scale the focal length based on the original width of the image.
            curr_focal_pred = (
                (prediction_focal[0][0] * (self.focal_end + 1.0 - self.focal_start * 1.0) + self.focal_start * 1.0)
                * (width * 1.0)
                / (self.INPUT_SIZE * 1.0)
            )
            curr_focal_pred = curr_focal_pred.item()

            # Following DeepCalib's official codes
            curr_dist_pred = prediction_dist[0][0] * 1.2
            curr_dist_pred = curr_dist_pred.item()

            temp_k = [[curr_focal_pred, 0, width / 2], [0, curr_focal_pred, height / 2], [0, 0, 1]]
            temp_xi = curr_dist_pred

            temp_hfov = 2 * np.arctan(width / 2 / curr_focal_pred)  # rad
            temp_vfov = 2 * np.arctan(height / 2 / curr_focal_pred)

            temp_hfov = temp_hfov.item()
            temp_vfov = temp_vfov.item()

            final_k_list.append(temp_k)
            final_xi_list.append(temp_xi)
            final_hfov_list.append(temp_hfov)
            final_vfov_list.append(temp_vfov)

        sample[Fields.meta][self.tag_field_name] = {
            "frames_folder": frames_root,
            "frame_names": frame_names,
            "intrinsics_list": final_k_list,
            "xi_list": final_xi_list,
            "hfov_list": final_hfov_list,
            "vfov_list": final_vfov_list,
        }

        if self.if_output_info:
            os.makedirs(self.output_info_dir, exist_ok=True)
            with open(
                os.path.join(
                    self.output_info_dir, os.path.splitext(os.path.basename(sample[self.video_key][0]))[0] + ".json"
                ),
                "w",
            ) as f:
                json.dump(sample[Fields.meta][self.tag_field_name], f)

        return sample
