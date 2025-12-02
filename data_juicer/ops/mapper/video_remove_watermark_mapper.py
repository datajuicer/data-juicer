import os
from typing import List, Optional

import av
import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.logger_utils import HiddenPrints
from data_juicer.utils.mm_utils import (
    close_video,
    extract_video_frames_uniformly,
    load_data_with_context,
    load_video,
    parse_string_to_roi,
    process_each_frame,
)

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

with HiddenPrints():
    cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "video_remove_watermark_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoRemoveWatermarkMapper(Mapper):
    """Remove watermarks from videos based on specified regions.

    This operator removes watermarks from video frames by detecting and masking the
    watermark areas. It supports two detection methods: 'pixel_value' and 'pixel_diversity'.
    The regions of interest (ROIs) for watermark detection can be specified as either pixel
    coordinates or ratios of the frame dimensions. The operator extracts a set number of
    frames uniformly from the video to detect watermark pixels. A pixel is considered part
    of a watermark if it meets the detection criteria in a minimum number of frames. The
    cleaned video is saved in the specified directory or the same directory as the input
    file if no save directory is provided."""

    def __init__(
        self,
        roi_strings: List[str] = ["0,0,0.1,0.1"],
        roi_type: str = "ratio",
        roi_key: Optional[str] = None,
        frame_num: PositiveInt = 10,
        min_frame_threshold: PositiveInt = 7,
        detection_method: str = "pixel_value",
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param roi_strings: a given list of regions the watermarks locate.
            The format of each can be "x1, y1, x2, y2", "(x1, y1, x2, y2)",
            or "[x1, y1, x2, y2]".
        :param roi_type: the roi string type. When the type is 'pixel', (x1,
            y1), (x2, y2) are the locations of pixels in the top left corner
            and the bottom right corner respectively. If the roi_type is
            'ratio', the coordinates are normalized by widths and heights.
        :param roi_key: the key name of fields in samples to store roi_strings
            for each sample. It's used for set different rois for different
            samples. If it's none, use rois in parameter "roi_strings".
            It's None in default.
        :param frame_num: the number of frames to be extracted uniformly from
            the video to detect the pixels of watermark.
        :param min_frame_threshold: a coordination is considered as the
            location of a watermark pixel when it is that in no less
            min_frame_threshold frames.
        :param detection_method: the method to detect the pixels of watermark.
            If it is 'pixel_value', we consider the distribution of pixel
            value in each frame. If it is 'pixel_diversity', we will consider
            the pixel diversity in different frames. The min_frame_threshold
            is useless and frame_num must be greater than 1 in
            'pixel_diversity' mode.
        :param save_dir: The directory where generated video files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        if roi_type not in ["ratio", "pixel"]:
            raise ValueError(f"roi_type [{roi_type}]" f" is not supported. " f"Can only be one of ['ratio', 'pixel']. ")

        if detection_method not in ["pixel_value", "pixel_diversity"]:
            raise ValueError(
                f"detection_method [{detection_method}]"
                f" is not supported. "
                f"Can only be one of ['pixel_value', 'pixel_diversity']. "
            )

        if detection_method == "pixel_diversity" and frame_num < 2:
            raise ValueError("frame_num must be greater than 1 in 'pixel_diversity' mode.")

        rois = []
        if roi_key is None:
            for roi_string in roi_strings:
                roi = parse_string_to_roi(roi_string, roi_type)
                if roi is None:
                    raise ValueError(
                        "The roi in roi_strings must be four no negative"
                        ' numbers in the format of "x1, y1, x2, y2", '
                        '"(x1, y1, x2, y2)", or "[x1, y1, x2, y2]".'
                    )
                rois.append(roi)

        self.roi_type = roi_type
        self.rois = rois
        self.roi_key = roi_key
        self.frame_num = frame_num
        self.min_frame_threshold = min_frame_threshold
        self.detection_method = detection_method
        self.save_dir = save_dir

    def _detect_watermark_via_pixel_value(self, frames, rois):
        masks = []
        for frame in frames:
            frame = frame.to_ndarray(format="bgr24")
            mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
            for roi in rois:
                # dimension of ndarray frame: height x width x channel
                roi_frame = frame[roi[1] : roi[3], roi[0] : roi[2]]
                gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # assume the watermark is located in the box, so the pixel in
                # the edge must be 0, if not, reverse binary_frame
                edge_positive_num = (binary_frame[0] > 0).sum() + (binary_frame[:, 0] > 0).sum()
                total = binary_frame.shape[0] + binary_frame.shape[1]
                if edge_positive_num * 2 > total:
                    binary_frame = ~binary_frame

                mask[roi[1] : roi[3], roi[0] : roi[2]] = mask[roi[1] : roi[3], roi[0] : roi[2]] | binary_frame
            masks.append(mask)
        final_mask = sum((mask == 255).astype(np.uint8) for mask in masks)
        final_mask = np.where(final_mask >= self.min_frame_threshold, 255, 0)
        final_mask = final_mask.astype(np.uint8)
        return final_mask

    def _detect_watermark_via_pixel_diversity(self, frames, rois):
        mask = np.zeros((frames[0].height, frames[0].width), dtype=np.uint8)
        frames = [frame.to_ndarray(format="bgr24") for frame in frames]

        for roi in rois:
            roi_frames = [frame[roi[1] : roi[3], roi[0] : roi[2]] for frame in frames]
            roi_frames = np.stack(roi_frames, axis=0)
            pixel_diversity = roi_frames.std(axis=0)
            pixel_diversity = pixel_diversity.sum(-1)
            max_diversity = np.max(pixel_diversity)
            min_diversity = np.min(pixel_diversity)
            if max_diversity > min_diversity:
                scaled_diversity = 255 * (pixel_diversity - min_diversity) / (max_diversity - min_diversity)
            else:
                scaled_diversity = np.zeros_like(pixel_diversity)
            scaled_diversity = scaled_diversity.astype(np.uint8)
            _, binary_frame = cv2.threshold(scaled_diversity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # the watermark pixels have less diversity
            binary_frame = ~binary_frame
            mask[roi[1] : roi[3], roi[0] : roi[2]] = mask[roi[1] : roi[3], roi[0] : roi[2]] | binary_frame

        return mask

    def _generate_watermark_mask(self, video, sample):
        frames = extract_video_frames_uniformly(video, self.frame_num)

        if self.roi_key is not None:
            roi_strings = sample[self.roi_key]
            if isinstance(roi_strings, str):
                roi_strings = [roi_strings]
            rois = [parse_string_to_roi(roi_string, self.roi_type) for roi_string in roi_strings]
            rois = [roi for roi in rois if roi is not None]
        else:
            rois = self.rois
        if self.roi_type == "ratio":
            rois = [
                tuple(
                    [
                        int(roi[0] * frames[0].width),
                        int(roi[1] * frames[0].height),
                        int(roi[2] * frames[0].width),
                        int(roi[3] * frames[0].height),
                    ]
                )
                for roi in self.rois
            ]

        if self.detection_method == "pixel_value":
            mask = self._detect_watermark_via_pixel_value(frames, rois)
        else:
            mask = self._detect_watermark_via_pixel_diversity(frames, rois)

        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernel)

    def _clean_watermark(self, frame, watermark_mask):
        np_frame = frame.to_ndarray(format="bgr24")
        new_np_frame = cv2.inpaint(np_frame, watermark_mask, 3, cv2.INPAINT_NS)
        return av.VideoFrame.from_ndarray(new_np_frame, format="bgr24")

    def process_single(self, sample, context=False):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        for index, video_key in enumerate(loaded_video_keys):
            video = videos[video_key]
            cleaned_video_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)

            if not os.path.exists(cleaned_video_key) or cleaned_video_key not in loaded_video_keys:
                watermark_mask = self._generate_watermark_mask(video, sample)

                def process_frame_func(frame):
                    return self._clean_watermark(frame, watermark_mask)

                cleaned_video_key = process_each_frame(video, cleaned_video_key, process_frame_func)

            loaded_video_keys[index] = cleaned_video_key

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(sample[self.video_key]):
            if sample[Fields.source_file][i] != value:
                if loaded_video_keys[i] != value:
                    sample[Fields.source_file][i] = value

        sample[self.video_key] = loaded_video_keys
        return sample
