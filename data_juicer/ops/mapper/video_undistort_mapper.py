import os
import subprocess

import numpy as np

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_undistort_mapper"

ffmpeg = LazyLoader("ffmpeg", "ffmpeg-python")


def get_global_intrinsics(ks):
    fx = ks[:, 0, 0]
    fy = ks[:, 1, 1]
    cx = ks[:, 0, 2]
    cy = ks[:, 1, 2]

    global_k = np.eye(3)
    global_k[0, 0] = np.median(fx)
    global_k[1, 1] = np.median(fy)
    global_k[0, 2] = np.median(cx)
    global_k[1, 2] = np.median(cy)

    return global_k


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoUndistortMapper(Mapper):
    """Undistort raw videos with corresponding camera intrinsics
    and distortion coefficients."""

    def __init__(
        self,
        output_video_dir: str = None,
        undistorted_video_field: str = MetaKeys.undistorted_video,
        camera_calibration_field: str = "camera_calibration",
        batch_size_each_video: int = 1000,
        crf: int = 22,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param output_video_dir: Output directory to save undistorted videos.
        :param undistorted_video_field: The field name to store the tags. It's
            "undistorted_video" in default.
        :param camera_calibration_field: The field name where the camera calibration info is stored.
        :param batch_size_each_video: Number of frames to process and save per
            temporary TS file batch.
        :param crf: Constant Rate Factor (CRF) for FFmpeg encoding quality.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        import importlib.metadata

        cv2_version = importlib.metadata.version("opencv-python")
        subprocess.run(["pip", "install", f"opencv-contrib-python=={cv2_version}"], check=True)
        import cv2

        self.VideoCapture = cv2.VideoCapture
        self.CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        self.CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        self.CAP_PROP_FPS = cv2.CAP_PROP_FPS
        self.omnidir = cv2.omnidir
        self.CV_16SC2 = cv2.CV_16SC2
        self.remap = cv2.remap
        self.INTER_CUBIC = cv2.INTER_CUBIC
        self.BORDER_CONSTANT = cv2.BORDER_CONSTANT
        self.cvtColor = cv2.cvtColor
        self.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB

        self.output_video_dir = output_video_dir
        assert self.output_video_dir is not None, "output_video_dir must be specified to save the undistorted videos."
        os.makedirs(self.output_video_dir, exist_ok=True)

        self.undistorted_video_field = undistorted_video_field
        self.batch_size_each_video = batch_size_each_video
        self.crf = crf
        self.camera_calibration_field = camera_calibration_field

    def concatenate_ts_files(self, folder, video_name, batch_counts):
        """Concatenate batch TS files into final mp4."""
        inputs_path = os.path.join(folder, "inputs.txt")

        # Create a file list for ffmpeg
        with open(inputs_path, "w") as f:
            for i in range(batch_counts):
                f.write(f"file '{video_name}_b{i:04d}.ts'\n")

        # Merge using ffmpeg concat demuxer
        ffmpeg.input(inputs_path, format="concat", safe=0).output(
            os.path.join(folder, f"{video_name}.mp4"),
            c="copy",
            movflags="frag_keyframe+empty_moov",
        ).run(overwrite_output=True)

        # Cleanup temporary TS files and list file
        for i in range(batch_counts):
            os.remove(os.path.join(folder, f"{video_name}_b{i:04d}.ts"))
        os.remove(inputs_path)

    def create_ffmpeg_writer(self, output_path, width, height, fps, crf):
        """Spawn an ffmpeg async encoding process for writing raw frames."""
        return (
            ffmpeg.output(
                ffmpeg.input(
                    "pipe:0",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{width}x{height}",
                    r=fps,
                ),
                output_path,
                **{
                    "preset": "medium",
                    "pix_fmt": "yuv420p",
                    "b:v": "0",
                    "c:v": "libx264",
                    "crf": str(crf),
                    "r": fps,
                    "movflags": "frag_keyframe+empty_moov",
                },
            )
            .overwrite_output()
            .run_async(quiet=True, pipe_stdin=True)
        )

    def process_single(self, sample, context=False):
        # check if it's generated already
        if self.undistorted_video_field in sample:
            return sample

        # there is no videos in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        camera_calibration_field = self.camera_calibration_field
        intrinsics_field = CameraCalibrationKeys.intrinsics
        xi_field = CameraCalibrationKeys.xi
        dist_coeffs_field = CameraCalibrationKeys.dist_coeffs
        rotation_matrix_field = CameraCalibrationKeys.rectify_R
        new_intrinsics_field = CameraCalibrationKeys.new_intrinsics

        sample[self.undistorted_video_field] = []

        for video_idx in range(len(sample[self.video_key])):
            cur_video_calibration = sample[Fields.meta][camera_calibration_field][video_idx]
            if not cur_video_calibration.get(intrinsics_field):
                raise ValueError(
                    f"The sample must include an '{intrinsics_field}' field to store the 3x3 camera intrinsics matrix."
                )

            if not cur_video_calibration.get(xi_field):
                raise ValueError(
                    f"The sample must include an '{xi_field}' field to store the parameter xi in CMei's model."
                )

            video_path = sample[self.video_key][video_idx]
            cap = self.VideoCapture(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Get video properties
            height = int(cap.get(self.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(self.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(self.CAP_PROP_FPS)

            K = cur_video_calibration.get(intrinsics_field)  # 3x3 camera intrinsics.
            xi = cur_video_calibration.get(xi_field)  # The parameter xi for CMei's model.

            D = cur_video_calibration.get(
                dist_coeffs_field, None
            )  # Distortion coefficients (k1,k2,p1,p2). If D is None then zero distortion is used.
            R = cur_video_calibration.get(
                rotation_matrix_field, None
            )  # Rotation transform between the original and object space. If it is None, there is no rotation.
            new_K = cur_video_calibration.get(
                new_intrinsics_field, None
            )  # New camera intrinsics. if new_K is empty then identity intrinsics are used.

            K = np.array(K, dtype=np.float32)
            xi = np.array(xi, dtype=np.float32)

            # frames k
            if len(K.shape) == 3:
                K = get_global_intrinsics(K)
            if len(xi) > 1:

                xi = np.median(xi)

            xi = np.array([xi], dtype=np.float64)

            if D is None:
                D = np.array([0, 0, 0, 0], dtype=np.float32)
            else:
                D = np.array(D, dtype=np.float32)

            if R is None:
                R = np.eye(3)
            else:
                R = np.array(R, dtype=np.float32)

            if new_K is None:
                new_K = K
            else:
                new_K = np.array(new_K, dtype=np.float32)

            map1, map2 = self.omnidir.initUndistortRectifyMap(
                K, D, xi, R, new_K, (width, height), self.CV_16SC2, self.omnidir.RECTIFY_PERSPECTIVE
            )

            # Initialize the first batch ffmpeg writer
            batch_number = 0
            writer = self.create_ffmpeg_writer(
                os.path.join(self.output_video_dir, f"{video_name}_b{batch_number:04d}.ts"),
                width,
                height,
                fps,
                self.crf,
            )

            idx = 0
            # Read and process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    # End of video stream: close the last writer
                    writer.stdin.close()
                    writer.wait()
                    break

                # Undistort the frame
                undistorted_frame = self.remap(
                    frame, map1, map2, interpolation=self.INTER_CUBIC, borderMode=self.BORDER_CONSTANT
                )

                # Convert BGR to RGB before writing to ffmpeg (FFmpeg expects RGB)
                undistorted_frame = self.cvtColor(undistorted_frame, self.COLOR_BGR2RGB)

                # Write to ffmpeg stdin
                writer.stdin.write(undistorted_frame.tobytes())

                # Check if the current batch is complete (for idx + 1)
                if (idx + 1) % self.batch_size_each_video == 0:
                    # Finalize the current batch writer
                    writer.stdin.close()
                    writer.wait()

                    # Start the next batch writer
                    batch_number += 1
                    writer = self.create_ffmpeg_writer(
                        os.path.join(self.output_video_dir, f"{video_name}_b{batch_number:04d}.ts"),
                        width,
                        height,
                        fps,
                        self.crf,
                    )

                idx += 1

            cap.release()

            # Merge all temporary TS chunks into the final MP4 file
            self.concatenate_ts_files(self.output_video_dir, video_name, batch_number + 1)
            out_video = os.path.join(self.output_video_dir, f"{video_name}.mp4")

            sample[self.undistorted_video_field].append(out_video)

        return sample
