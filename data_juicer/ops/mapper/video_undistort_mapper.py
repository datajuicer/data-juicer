import os
import subprocess

import numpy as np

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_undistort_mapper"

ffmpeg = LazyLoader("ffmpeg", "ffmpeg-python")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoUndistortMapper(Mapper):
    """Undistort raw videos with corresponding camera intrinsics
    and distortion coefficients."""

    def __init__(
        self,
        output_video_dir: str = DATA_JUICER_ASSETS_CACHE,
        tag_field_name: str = MetaKeys.video_undistortion_tags,
        batch_size_each_video: int = 1000,
        crf: int = 22,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param output_video_dir: Output directory to save undistorted videos.
        :param tag_field_name: The field name to store the tags. It's
            "video_undistortion_tags" in default.
        :param batch_size_each_video: Number of frames to process and save per
            temporary TS file batch.
        :param crf: Constant Rate Factor (CRF) for FFmpeg encoding quality.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        opencv_contrib_python_exist_code = subprocess.run(["pip", "show", "opencv-contrib-python"])
        # This is written to fix version issues with Numpy
        if opencv_contrib_python_exist_code.returncode == 1:  # not exist
            LazyLoader.check_packages(["opencv-contrib-python"])
            subprocess.run(["pip", "install", "numpy==1.26.4"], check=True)

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
        self.tag_field_name = tag_field_name
        self.batch_size_each_video = batch_size_each_video
        self.crf = crf

    def concatenate_ts_files(self, folder, video_name, batch_counts):
        """Concatenate batch TS files into final mp4."""
        inputs_path = os.path.join(folder, "inputs.txt")

        # Create a file list for ffmpeg
        with open(inputs_path, "w") as f:
            for i in range(batch_counts):
                f.write(f"file '{video_name}_b{i:04d}.ts'\n")

        # Merge using ffmpeg concat demuxer
        ffmpeg.input(inputs_path, format="concat", safe=0).output(
            os.path.join(folder, f"{video_name}.mp4"), c="copy"
        ).run()

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
                },
            )
            .overwrite_output()
            .run_async(quiet=True, pipe_stdin=True)
        )

    def process_single(self, sample, context=False):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no videos in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        cap = self.VideoCapture(sample[self.video_key][0])
        video_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]

        # Get video properties
        height = int(cap.get(self.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(self.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(self.CAP_PROP_FPS)

        if "intrinsics" not in sample or sample["intrinsics"] is None:
            raise ValueError("The sample must include an 'intrinsics' field to store the 3x3 camera intrinsics matrix.")

        if "xi" not in sample or sample["xi"] is None:
            raise ValueError("The sample must include an 'xi' field to store the parameter xi in CMei's model.")

        K = sample["intrinsics"]  # 3x3 camera intrinsics.
        D = sample.get(
            "distortion_coefficients", None
        )  # Distortion coefficients (k1,k2,p1,p2). If D is None then zero distortion is used.
        xi = sample["xi"]  # The parameter xi for CMei's model.
        R = sample.get(
            "rotation_matrix", None
        )  # Rotation transform between the original and object space. If it is None, there is no rotation.
        new_K = sample.get(
            "intrinsics_new", None
        )  # New camera intrinsics. if new_K is empty then identity intrinsics are used.

        K = np.array(K, dtype=np.float32)
        xi = np.array(xi, dtype=np.float32)

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
        os.makedirs(self.output_video_dir, exist_ok=True)
        batch_number = 0
        writer = self.create_ffmpeg_writer(
            os.path.join(self.output_video_dir, f"{video_name}_b{batch_number:04d}.ts"), width, height, fps, self.crf
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

        sample[Fields.meta][self.tag_field_name] = {
            "new_video_path": os.path.join(self.output_video_dir, f"{video_name}.mp4")
        }

        return sample
