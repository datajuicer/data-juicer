import abc
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import IO, Iterator, List, Optional, Union

import attrs
import numpy as np
import numpy.typing as npt

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import close_video, cut_video_by_seconds

# TODO: support cuda

av = LazyLoader("av")
cv2 = LazyLoader("cv2")
decord = LazyLoader("decord")


@dataclass
class VideoMetadata:
    """Metadata for video content.

    This class stores essential video properties such as resolution, frame rate,
    duration.
    """

    height: int | None = None
    width: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    duration: float | None = None


@attrs.define
class Frames:
    frames: List[npt.NDArray[np.uint8]]
    indices: List[int] | None = None
    pts_time: List[float] | None = None


@attrs.define
class Clip:
    """Container for video clip data including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, span, frames and so on.
    """

    source_video: str
    span: tuple[float, float]
    id: str | None = None
    path: str | None = None
    encoded_data: bytes | None = None
    frames: List[npt.NDArray[np.uint8]] | None = None


class VideoReader(abc.ABC):
    """
    Abstract class for video processing.

    This class provides an interface for video processing tasks such as
    extracting frames, key frames, and clipping.
    """

    def __init__(self, video_source: Union[str, Path, bytes, IO[bytes]]):
        """
        Initialize video reader.

        :param video_source: Path, URL, bytes, or file-like object.
        """
        self.video_source = video_source
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is not None:
            return self._metadata

        self._metadata = self.get_metadata()
        return self._metadata

    @abc.abstractmethod
    def get_metadata(self) -> VideoMetadata:
        """Get video metadata."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_frames(self, start_time: float = 0, end_time: Optional[float] = None) -> Iterator[np.ndarray]:
        """Yield frames between [start_time, end_time) as numpy arrays.

        :param start_time: Start time in seconds (inclusive)
        :param end_time: End time in seconds (exclusive). If None, extract to end of video.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_keyframes(self, start_time: float = 0, end_time: Optional[float] = None) -> "Frames":
        """Extract keyframes and return them in a `Frames` object.

        :param start_time: Start time in seconds (inclusive)
        :param end_time: End time in seconds (exclusive). If None, extract to end of video.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_clip(
        self, start_time: float = 0, end_time: Optional[float] = None, output_path: str = None, to_numpy: bool = True
    ) -> Optional["Clip"]:
        """Extract a subclip.

        :param start_time: Start time in seconds
        :param end_time: End time in seconds. If None, extract to end of video.
        :param output_path: The path to save the output video clip. If provided, the clip is saved to a file.
        :param to_numpy: Whether to return frames as a list of numpy arrays.
        :return: A `Clip` object on success, or `None` on failure.
        """
        raise NotImplementedError

    def check_time_span(
        self,
        start_time: Optional[float] = 0.0,
        end_time: Optional[float] = None,
    ) -> None:
        if start_time < 0:
            raise ValueError("start_time cannot be negative")
        if end_time is not None and end_time <= 0:
            raise ValueError("end_time cannot be negative")
        if end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

    @abc.abstractmethod
    def close(self) -> None:
        """Release any held resources."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Check if the backend is available."""
        raise NotImplementedError


class AVReader(VideoReader):
    """Video reader using the AV library."""

    def __init__(
        self, video_source: Union[str, Path, bytes, IO[bytes]], video_stream_index: int = 0, frame_format: str = "rgb24"
    ):
        """
        Initialize AVReader.

        :param video_source: Path, URL, bytes, or file-like object.
        :param video_stream_index: Video stream index to decode, default set to 0.
        :param frame_format: Frame format to decode, default set to "rgb24".
        """
        super().__init__(video_source)

        self.frame_format = frame_format
        self.video_stream_index = video_stream_index

        if isinstance(self.video_source, bytes):
            self.container = av.open(io.BytesIO(self.video_source))
        else:
            self.container = av.open(self.video_source)

        video_streams = self.container.streams.video
        if not video_streams:
            raise ValueError("Not found video stream")
        if self.video_stream_index < 0 or self.video_stream_index >= len(video_streams):
            raise IndexError(f"index {self.video_stream_index} is out of range, valid range: 0-{len(video_streams)-1}")

        self.video_stream = self.container.streams.video[video_stream_index]

    def get_metadata(self) -> VideoMetadata:
        stream = self.video_stream
        metadata = VideoMetadata(
            duration=float(stream.duration * stream.time_base),
            fps=float(stream.average_rate),
            width=stream.width,
            height=stream.height,
            num_frames=stream.frames,
        )
        return metadata

    def extract_frames(
        self,
        start_time: Optional[float] = 0.0,
        end_time: Optional[float] = None,
    ) -> Iterator[np.ndarray]:
        """
        Get the video's frames from the container within a specified time range.

        :param start_time: Start time in seconds (default: 0.0).
        :param end_time: End time in seconds (exclusive). If None, decode until end.

        :return: Iterator of numpy objects within the specified time range.
        """
        self.check_time_span(start_time, end_time)

        if end_time is None:
            end_time = self.metadata.duration
        elif end_time and end_time > self.metadata.duration:
            end_time = self.metadata.duration

        time_base = self.video_stream.time_base
        start_pts = int(start_time / time_base)
        end_pts = int(end_time / time_base) if end_time is not None else None

        # Seek to the start position
        self.container.seek(start_pts, stream=self.video_stream)

        # Decode and filter frames
        for frame in self.container.decode(video=self.video_stream_index):
            frame_pts = frame.pts
            if frame_pts is None:
                continue  # Skip frames with invalid PTS

            frame_time = frame_pts * time_base

            # Skip frames before start_time (may occur due to keyframe seeking)
            if frame_time < start_time:
                continue

            # Break if past end_time
            if end_pts is not None and frame_pts >= end_pts:
                break

            rgb_frame = frame.reformat(format=self.frame_format).to_ndarray()

            yield rgb_frame

    def extract_keyframes(self, start_time: float = 0, end_time: Optional[float] = None):
        """Extract key frames.

        :param start_time: Start time in seconds (default: 0.0).
        :param end_time: End time in seconds (exclusive). If None, decode until end.

        :return: Iterator of numpy objects within the specified time range.
        """
        self.check_time_span(start_time, end_time)

        end_time = min(end_time, self.metadata.duration) if end_time is not None else self.metadata.duration
        time_base = self.video_stream.time_base
        stream_start_seconds = self.video_stream.start_time * time_base

        key_frames = []
        self.container.seek(0)

        for frame in self.container.decode(video=self.video_stream_index):
            # Calculate absolute time in container's timeline
            frame_abs_time = stream_start_seconds + frame.pts * time_base

            # Stop if we've passed the end time
            if frame_abs_time >= end_time:
                break

            # Collect keyframes within the target range
            if frame.key_frame and frame_abs_time >= start_time:
                key_frames.append(frame)

        # Convert frames to output format
        pts_time = [float(stream_start_seconds + f.pts * time_base) for f in key_frames]
        frame_indices = [int(t * self.metadata.fps) for t in pts_time]
        formatted_frames = [frame.reformat(format=self.frame_format).to_ndarray() for frame in key_frames]

        return Frames(frames=formatted_frames, indices=frame_indices, pts_time=pts_time)

    def extract_clip(self, start_time, end_time, output_path: str = None, to_numpy: bool = True):
        """
        Extract a clip from the video based on the start and end time.

        :param start_time: the start time in second.
        :param end_time: the end time in second. If it's None, this function
            will cut the video from the start_seconds to the end of the video.
        :param output_path: the path to output video.

        :return: Clip object.
            If output_path is not None, it will save the clip to output_path.
            If to_numpy is True, it will return clip data as numpy array and save to Clip.frames.
            If to_numpy is False, it will return clip data as bytes and save to Clip.encoded_data.
        """
        self.check_time_span(start_time, end_time)
        if end_time and end_time > self.metadata.duration:
            end_time = self.metadata.duration

        frames, encoded_data = None, None
        if (not to_numpy) or output_path:
            res = cut_video_by_seconds(
                input_video=self.container,
                output_video=output_path,
                start_seconds=start_time,
                end_seconds=end_time,
                video_stream_index=self.video_stream_index,
            )
            if output_path:
                if not res:
                    return None
                encoded_data = None
            else:
                encoded_data = res.getvalue()
        else:
            frames = list(self.extract_frames(start_time, end_time))

        return Clip(
            # id=uuid.uuid4(),
            source_video=self.video_source,
            path=output_path,
            span=(start_time, end_time),
            encoded_data=encoded_data,
            frames=frames,
        )

    @classmethod
    def is_available(cls):
        try:
            import av  # noqa: F401

            return True
        except ImportError:
            return False

    def close(self):
        close_video(self.container)


class FFmpegReader(VideoReader):
    """
    Video reader using FFmpeg.
    """

    def __init__(
        self, video_source: Union[str, Path, bytes, IO[bytes]], video_stream_index: int = 0, frame_format: str = "rgb24"
    ):
        """
        Initialize FFmpegReader.

        :param video_source: Path, URL, bytes, or file-like object.
        :param video_stream_index: Video stream index to decode, default set to 0.
        :param frame_format: Frame format, default set to "rgb24".
        """
        super().__init__(video_source)

        self.video_stream_index = video_stream_index
        self.frame_format = frame_format
        self._temp_file = None
        self._should_cleanup = False

        self.video_path = self.video_source
        # Convert Path to string if needed
        if isinstance(self.video_source, Path):
            self.video_path = str(self.video_source)

        # Handle bytes and file-like objects by creating temporary files
        if isinstance(self.video_source, bytes) or (
            hasattr(self.video_source, "read") and hasattr(self.video_source, "seek")
        ):
            self.video_path = self._create_temp_file()

    def _create_temp_file(self):
        """Create a temporary file for bytes or file-like input and set cleanup flag."""
        try:
            # Create a named temporary file that will be automatically deleted when closed
            self._temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            self._should_cleanup = True

            if isinstance(self.video_source, bytes):
                # Write bytes to temporary file
                self._temp_file.write(self.video_source)
            else:
                # File-like object - read and write its content
                self.video_source.seek(0)  # Ensure we're at the beginning
                shutil.copyfileobj(self.video_source, self._temp_file)

            self._temp_file.flush()
            self._temp_file.close()
        except Exception as e:
            # Clean up on error
            self._cleanup_temp_file()
            raise RuntimeError(f"Failed to create temporary file for video source: {e}")

        return self._temp_file.name

    def _cleanup_temp_file(self):
        """Clean up temporary file if it was created."""
        if self._should_cleanup and self._temp_file and os.path.exists(self._temp_file.name):
            temp_path = Path(self._temp_file.name)
            self._temp_file.close()
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError as e:
                from loguru import logger

                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")
            self._temp_file = None
            self._should_cleanup = False

    def get_metadata(self) -> VideoMetadata:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            f"v:{self.video_stream_index}",
            "-show_entries",
            "stream=duration,avg_frame_rate,width,height,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            self.video_path,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFprobe error: {result.stderr.strip()}")

        try:
            probe_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse output of FFprobe: {e}")

        streams = probe_data.get("streams", [])
        if not streams:
            raise RuntimeError("Not found video stream!")
        video_stream = streams[0]

        format_info = probe_data.get("format", {})

        duration = video_stream.get("duration")
        if duration:
            duration = float(duration)
        else:
            # use container format duration as a fallback
            duration = float(format_info.get("duration", 0))

        avg_frame_rate = video_stream.get("avg_frame_rate", "0/0")
        if "/" in avg_frame_rate:
            numerator, denominator = map(float, avg_frame_rate.split("/"))
            fps = numerator / denominator if denominator != 0 else 0.0
        else:
            fps = float(avg_frame_rate) if avg_frame_rate else 0.0

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        num_frames = int(video_stream.get("nb_frames", 0))

        # If the number of frames is not available, estimate it based on duration and frame rate
        if num_frames <= 0 and duration > 0 and fps > 0:
            num_frames = int(round(duration * fps))

        metadata = VideoMetadata(duration=duration, fps=fps, width=width, height=height, num_frames=num_frames)
        return metadata

    def extract_frames(
        self, start_time: Optional[float] = 0.0, end_time: Optional[float] = None
    ) -> Iterator[np.ndarray]:
        """
        Get the video's frames within a specified time range.

        :param start_time: Start time in seconds (default: 0.0).
        :param end_time: End time in seconds (exclusive). If None, decode until end.
        :param duration: Duration from start_time. Mutually exclusive with end_time.

        :return: Iterator of VideoFrame objects within the specified time range.
        """
        self.check_time_span(start_time, end_time)

        w = self.metadata.width
        h = self.metadata.height

        cmd = [
            "ffmpeg",
            "-v",
            "quiet",
            "-ss",
            str(start_time),
        ]

        if end_time is not None:
            cmd += ["-to", str(end_time)]

        cmd += [
            "-i",
            self.video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.frame_format,
            "-",
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_size = w * h * 3  # 3 bytes per pixel of RGB

        try:
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) < frame_size:
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
                yield frame
        finally:
            self._kill_process(process)

    def extract_keyframes(self, start_time: float = 0, end_time: Optional[float] = None):
        """
        Extract only true keyframes (I-frames) from video.
        """
        self.check_time_span(start_time, end_time)

        cmd = ["ffmpeg"]
        if start_time > 0 or end_time is not None:
            if not end_time:
                end_time = self.metadata.duration
            cmd.extend(["-ss", str(start_time), "-to", str(end_time)])

        cmd.extend(
            [
                "-i",
                self.video_path,
                "-vf",
                "showinfo,select=eq(pict_type\,I)",  # noqa: W605
                "-vsync",
                "vfr",
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.frame_format,
                "-",
            ]
        )

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        h, w = self.metadata.height, self.metadata.width
        frame_size = h * w * 3  # 3 bytes per pixel for RGB

        key_frames, metadata = [], []
        metadata_queue = Queue()
        stop_event = threading.Event()

        def read_stderr():
            """
            Parse metadata from stderr and put it into a queue
            """
            while not stop_event.is_set():
                line = process.stderr.readline()
                if not line:
                    break
                try:
                    line = line.decode("utf-8")
                    if "iskey:1" in line and "pts_time:" in line:
                        match = re.search(r"n:\s*(\d+).*?pts_time:([\d.]+)", line)
                        if match:
                            n = int(match.group(1))  # frame index in the original video
                            pts_time = float(match.group(2))
                            metadata_queue.put((n, pts_time))
                except (UnicodeDecodeError, ValueError, AttributeError):
                    continue

        # start the stderr thread
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        try:
            # main thread reads stdout frame data
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) < frame_size:
                    break
                try:
                    n, pts_time = metadata_queue.get(timeout=1)
                    metadata.append((n, pts_time))
                except Empty:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
                key_frames.append(frame)
        finally:
            stop_event.set()
            self._kill_process(process)
            stderr_thread.join()

        if not metadata:
            return Frames(frames=[], indices=[], pts_time=[])

        frame_indices, pts_time = zip(*metadata)
        return Frames(frames=key_frames, indices=list(frame_indices), pts_time=list(pts_time))

    def extract_clip(self, start_time, end_time, output_path: str = None, to_numpy=True, **kwargs):
        """
        Extract a clip from the video based on the start and end time.
        :param output_path: the path to output video.
        :param start_time: the start time in second.
        :param end_time: the end time in second. If it's None, this function
            will cut the video from the start_seconds to the end of the video.
        :param to_numpy: whether to return clip data as numpy array and save to Clip.frames.

        :return: Clip object.
            If output_path is not None, it will save the clip to output_path.
            If to_numpy is True, it will return clip data as numpy array and save to Clip.frames.
            If to_numpy is False, it will return clip data as bytes and save to Clip.encoded_data.
        """
        self.check_time_span(start_time, end_time)

        # allows adding extra arguments passed to ffmpeg
        import shlex

        ffmpeg_extra_args = shlex.split(kwargs.get("ffmpeg_extra_args", ""))

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file without asking
            "-ss",
            str(start_time),  # Start time
            "-i",
            self.video_path,  # Input file
        ]

        # Add end time if specified
        if end_time is not None:
            duration = end_time - start_time
            cmd.extend(["-t", str(duration)])

        # Set output options
        cmd.extend(
            [
                "-c",
                "copy",  # Stream copy (fast, no re-encoding)
                "-f",
                "mp4",
                # "-movflags", "frag_keyframe+empty_moov",  # opening when mounting oss storage may avoid unexpected errors.
            ]
        )
        cmd.extend(ffmpeg_extra_args)

        encoded_data = None
        frames = None
        if output_path is not None:
            # Output to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd.extend([output_path])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None
        elif to_numpy:
            frames = list(self.extract_frames(start_time, end_time))
        else:
            # Output to stdout
            cmd.extend(["pipe:1"])  # Output to stdout
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            encoded_data, _ = process.communicate()
            self._kill_process(process)

        return Clip(
            # id=uuid.uuid4(),
            source_video=self.video_source,
            path=output_path,
            span=(start_time, end_time),
            encoded_data=encoded_data,
            frames=frames,
        )

    def close(self):
        """Clean up resources, including temporary files."""
        self._cleanup_temp_file()

    @classmethod
    def is_available(cls):
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
            return True
        except ImportError:
            return False

    def _kill_process(self, process):
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
        process.terminate()
        try:
            process.wait(timeout=5)  # wait the process to finish,
        except subprocess.TimeoutExpired:
            process.kill()  # if it doesn't finish within 5 seconds, kill it


# TODO: support audio for clip
class DecordReader(VideoReader):
    """Video reader using Decord"""

    def __init__(
        self,
        video_source: Union[str, Path, bytes, IO[bytes]],
    ):
        """
        Initialize the video reader.

        :param video_source: Path, URL, bytes, or file-like object.
        """
        super().__init__(video_source)

        if isinstance(video_source, Path):
            self.reader = decord.VideoReader(str(video_source))
        elif isinstance(video_source, bytes):
            self.reader = decord.VideoReader(io.BytesIO(video_source))
        else:
            self.reader = decord.VideoReader(video_source)

    def get_metadata(self) -> VideoMetadata:
        fps = self.reader.get_avg_fps()
        num_frames = len(self.reader)

        return VideoMetadata(
            duration=num_frames / fps,
            fps=fps,
            width=self.reader[0].shape[1],
            height=self.reader[0].shape[0],
            num_frames=num_frames,
        )

    def _get_frame_index_by_time_span(
        self,
        start_time: Optional[float] = 0.0,
        end_time: Optional[float] = None,
    ) -> List[int]:
        # Get video properties
        fps = self.metadata.fps
        num_frames = self.metadata.num_frames
        total_duration = self.metadata.duration

        # Set default end_time if not provided
        if end_time is None:
            end_time = total_duration
        elif end_time > total_duration:
            end_time = total_duration

        # Convert time to frame indices (using ceiling for start and end)
        start_frame = math.ceil(start_time * fps)
        end_frame = math.ceil(end_time * fps)

        # Clamp frames to valid range [0, num_frames]
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(0, min(end_frame, num_frames))

        return start_frame, end_frame

    def extract_frames(
        self,
        start_time: Optional[float] = 0.0,
        end_time: Optional[float] = None,
    ) -> Iterator[np.ndarray]:
        """
        Get the video's frames within a specified time range using decord.

        :param start_time: Start time in seconds (default: 0.0).
        :param end_time: End time in seconds (exclusive). If None, decode until end.
        :return: Numpy array of frames in shape (num_frames, height, width, channels).
        """
        self.check_time_span(start_time, end_time)

        start_frame, end_frame = self._get_frame_index_by_time_span(start_time, end_time)

        # Handle empty frame range
        if start_frame >= end_frame:
            return np.array([])

        # Extract frames using decord
        frame_indices = range(start_frame, end_frame)
        frames = self.reader.get_batch(frame_indices).asnumpy()

        yield from frames

    def extract_keyframes(self, start_time: float = 0, end_time: Optional[float] = None):
        self.check_time_span(start_time, end_time)

        start_frame, end_frame = self._get_frame_index_by_time_span(start_time, end_time)

        key_indices = self.reader.get_key_indices()

        # filter key frames within the specified time range
        filtered_key_indices = []
        for idx in key_indices:
            if start_frame <= idx < end_frame:
                filtered_key_indices.append(idx)

        if not filtered_key_indices:
            print(f"Warning: No keyframes found between {start_time}s and {end_time}s")
            return Frames(frames=[], indices=[], pts_time=[])

        key_frames = self.reader.get_batch(filtered_key_indices)
        key_times = []
        for idx in filtered_key_indices:
            start_pts, _ = self.reader.get_frame_timestamp(idx)
            key_times.append(start_pts)
        key_frames = key_frames.asnumpy()

        return Frames(frames=key_frames, indices=filtered_key_indices, pts_time=key_times)

    def extract_clip(self, start_time, end_time, output_path: str = None, to_numpy=True):
        """
        Extract a clip from the video based on the start and end time.

        :param start_time: the start time in second.
        :param end_time: the end time in second. If it's None, this function
            will cut the video from the start_seconds to the end of the video.
        :param output_path: the path to output video.
        :param to_numpy: whether to return clip data as numpy array and save to Clip.frames.
        :return: Clip object.
        """
        if not to_numpy:
            raise ValueError("'to_numpy' must be True when using decord")

        if output_path:
            raise NotImplementedError("'output_path' is not supported when using decord")

        self.check_time_span(start_time, end_time)

        # Calculate frame indices
        start_frame, end_frame = self._get_frame_index_by_time_span(start_time, end_time)

        # Handle empty frame range
        if start_frame >= end_frame:
            return None

        # Extract frames using decord
        frame_indices = range(start_frame, end_frame)
        clip = self.reader.get_batch(frame_indices).asnumpy()

        if len(clip) == 0:
            return None

        return Clip(
            # id=uuid.uuid4(),
            source_video=self.video_source,
            path=output_path,
            span=(start_time, end_time),
            frames=clip,
        )

    def close(self):
        del self.reader

    @classmethod
    def is_available(cls):
        try:
            import decord  # noqa: F401

            return True
        except ImportError:
            return False


def create_video_reader(video_source: str, backend: str = "auto", **kwargs) -> VideoReader:
    backends = {"ffmpeg": FFmpegReader, "decord": DecordReader, "av": AVReader}

    if backend != "auto":
        cls = backends[backend]
        if cls.is_available():
            return cls(video_source, **kwargs)
        raise RuntimeError(f"Backend {backend} not available")

    # select available backend automatically
    for name, cls in backends.items():
        if cls.is_available():
            return cls(video_source, **kwargs)
    raise RuntimeError("No available video backend found")
