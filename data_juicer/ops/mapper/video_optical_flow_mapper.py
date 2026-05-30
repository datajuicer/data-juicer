import os

import cv2
import numpy as np
from pydantic import PositiveInt

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_optical_flow_mapper"

torch = LazyLoader("torch")
torchvision = LazyLoader("torchvision")


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoOpticalFlowMapper(Mapper):
    """Generate optical flow information for videos."""

    _accelerator = "cuda"

    def __init__(
        self,
        if_save_visualization: bool = True,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        frame_num: PositiveInt = 3,
        duration: float = 0,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param if_save_visualization: Whether to save visualization results.
        :param save_visualization_dir: The path for saving visualization results.
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
        :param frame_dir: Output directory to save extracted frames.

        """

        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(["torchvision"])

        self.model_key = prepare_model(model_type="optical_flow_raft")
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_optical_flow_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "frame_key": MetaKeys.video_frames,
            "num_proc": None,  # Disable multiprocessing to avoid nested process pool issue
            "auto_op_parallelism": False,  # Disable auto parallelism to avoid nested process pool issue
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

    def raft_preprocess(self, img1_batch, img2_batch, transforms):
        img1_batch = torchvision.transforms.functional.resize(img1_batch, size=[520, 960], antialias=False)
        img2_batch = torchvision.transforms.functional.resize(img2_batch, size=[520, 960], antialias=False)
        return transforms(img1_batch, img2_batch)

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if (self.video_key not in sample or not sample[self.video_key]) and self.frame_field not in sample:
            sample[Fields.meta][self.tag_field_name] = {"pred_flow": []}
            return sample

        model, transforms = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        if rank is not None:
            device = f"cuda:{str(rank)}"
        else:
            device = "cuda"

        if self.frame_field in sample:
            frames_path = sample[self.frame_field]
            video_name = os.path.basename(os.path.dirname(frames_path[0]))
        else:
            # load videos
            ds_list = [{"text": SpecialTokens.video, "videos": sample[self.video_key]}]

            dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
            dataset = self.fused_ops[0].run(dataset)

            temp_frame_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]
            frames_root = os.path.join(self.frame_dir, temp_frame_name)
            frame_names = os.listdir(frames_root)
            frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])
            video_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]

        if len(frames_path) < 2:
            sample[Fields.meta][self.tag_field_name] = {"pred_flow": []}
            return sample

        frame_arr_list = []
        for temp_img_path_id, temp_img_path in enumerate(frames_path):
            frame_arr_list.append(cv2.imread(temp_img_path)[:, :, ::-1][None, :])

        frame_tensor = torch.from_numpy(np.concatenate(frame_arr_list, axis=0)).permute(0, 3, 1, 2)
        img1_batch = frame_tensor.clone()[:-1, :, :, :].to(device)
        img2_batch = frame_tensor.clone()[1:, :, :, :].to(device)

        img1_batch, img2_batch = self.raft_preprocess(img1_batch, img2_batch, transforms)

        with torch.no_grad():
            list_of_flows = model(img1_batch, img2_batch)
            predicted_flow = list_of_flows[-1]

        if self.if_save_visualization:
            os.makedirs(os.path.join(self.save_visualization_dir, video_name), exist_ok=True)

            flow_imgs = torchvision.utils.flow_to_image(predicted_flow).cpu().permute(0, 2, 3, 1).numpy()
            for temp_flow_img_id in range(len(flow_imgs)):
                cv2.imwrite(
                    os.path.join(self.save_visualization_dir, video_name, f"vis_{str(temp_flow_img_id)}.jpg"),
                    flow_imgs[temp_flow_img_id],
                )

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["pred_flow"] = predicted_flow.detach().cpu().numpy()

        return sample
