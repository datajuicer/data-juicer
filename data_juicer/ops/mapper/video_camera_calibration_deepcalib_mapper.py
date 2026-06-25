import numpy as np

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_calibration_deepcalib_mapper"

cv2 = LazyLoader("cv2", "opencv-python")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraCalibrationDeepcalibMapper(Mapper):
    """Compute the camera intrinsics and field of view (FOV)
    for a static camera using DeepCalib."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "weights_10_0.02.h5",
        frame_field: str = MetaKeys.video_frames,
        tag_field_name: str = MetaKeys.camera_calibration_deepcalib_tags,
        frame_batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the DeepCalib Regression model.
        :param frame_field: The field name where the video frames are stored.
        :param tag_field_name: The field name to store the tags. It's
            "camera_calibration_deepcalib_tags" in default.
        :param frame_batch_size: Number of frames to batch together for GPU
            inference. Larger values improve throughput but require more VRAM.
            Default: 8.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        LazyLoader.check_packages(["tensorflow==2.20.0"])
        import keras
        from keras.applications.imagenet_utils import preprocess_input

        self.keras = keras
        self.preprocess_input = preprocess_input

        self.model_key = prepare_model(model_type="deepcalib", model_path=model_path)
        self.frame_field = frame_field
        self.tag_field_name = tag_field_name
        self.frame_batch_size = frame_batch_size

        self.INPUT_SIZE = 299
        self.focal_start = 40
        self.focal_end = 500

    def _decode_and_preprocess_frame(self, frame):
        """Decode a single frame, preprocess it for DeepCalib, and return
        (preprocessed_image, original_height, original_width)."""
        if isinstance(frame, bytes):
            image_array = np.frombuffer(frame, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(frame)

        height, width, channels = image.shape

        image = cv2.resize(image, (self.INPUT_SIZE, self.INPUT_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = image - 0.5
        image = image * 2.0

        return image, height, width

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # load videos
        videos_frames = sample[self.frame_field]
        model = get_model(self.model_key, rank, self.use_cuda())

        sample[Fields.meta][self.tag_field_name] = []

        for video_idx in range(len(videos_frames)):
            # Step 1: Decode and preprocess all frames, record original dimensions
            preprocessed_images = []
            heights = []
            widths = []

            for frame in videos_frames[video_idx]:
                image, h, w = self._decode_and_preprocess_frame(frame)
                preprocessed_images.append(image)
                heights.append(h)
                widths.append(w)

            num_frames = len(preprocessed_images)

            final_k_list = []
            final_xi_list = []
            final_hfov_list = []
            final_vfov_list = []

            # Step 2: Batch inference
            # All frames are resized to INPUT_SIZE x INPUT_SIZE, so they can
            # always be stacked into batches regardless of original resolution.
            for batch_start in range(0, num_frames, self.frame_batch_size):
                batch_end = min(batch_start + self.frame_batch_size, num_frames)
                batch_images = np.array(preprocessed_images[batch_start:batch_end])  # (B, H, W, C)
                batch_images = self.preprocess_input(batch_images)

                prediction = model.predict(batch_images)
                prediction_focal = prediction[0]  # (B, 1)
                prediction_dist = prediction[1]  # (B, 1)

                for i in range(batch_end - batch_start):
                    idx = batch_start + i
                    orig_w = widths[idx]
                    orig_h = heights[idx]

                    # Scale the focal length based on the original width of the image.
                    curr_focal_pred = (
                        (
                            prediction_focal[i][0] * (self.focal_end + 1.0 - self.focal_start * 1.0)
                            + self.focal_start * 1.0
                        )
                        * (orig_w * 1.0)
                        / (self.INPUT_SIZE * 1.0)
                    )
                    curr_focal_pred = curr_focal_pred.item()

                    # Following DeepCalib's official codes
                    curr_dist_pred = prediction_dist[i][0] * 1.2
                    curr_dist_pred = curr_dist_pred.item()

                    temp_k = [[curr_focal_pred, 0, orig_w / 2], [0, curr_focal_pred, orig_h / 2], [0, 0, 1]]
                    temp_xi = curr_dist_pred

                    temp_hfov = 2 * np.arctan(orig_w / 2 / curr_focal_pred)  # rad
                    temp_vfov = 2 * np.arctan(orig_h / 2 / curr_focal_pred)

                    temp_hfov = temp_hfov.item()
                    temp_vfov = temp_vfov.item()

                    final_k_list.append(temp_k)
                    final_xi_list.append(temp_xi)
                    final_hfov_list.append(temp_hfov)
                    final_vfov_list.append(temp_vfov)

            sample[Fields.meta][self.tag_field_name].append(
                {
                    CameraCalibrationKeys.intrinsics: final_k_list,
                    CameraCalibrationKeys.xi: final_xi_list,
                    CameraCalibrationKeys.hfov: final_hfov_list,
                    CameraCalibrationKeys.vfov: final_vfov_list,
                }
            )

        return sample
