import os
import cv2
import json
from dataclasses import dataclass

import einops
import torch
from torch.utils.data import Dataset

from data_juicer.utils.mm_utils import load_video, close_video

@dataclass
class VideoDataset(Dataset):
    dataset_path: str
    seq_length: int
    height: int
    width: int
    mm_dir: str = None
    min_spacing: int = 1
    max_spacing: int = 1
    x_flip: bool = False
    video_key: str = 'videos'

    def __post_init__(self):
        assert self.seq_length >= 1

        assert os.path.exists(self.dataset_path), self.dataset_path

        self.video_paths = []
        with open(self.dataset_path) as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                for video_path in data[self.video_key]:
                    if self.mm_dir is not None:
                        video_path = os.path.join(self.mm_dir, video_path)
                    self.video_paths.append(video_path)

    def sample_frames(self, video_path):
        container = load_video(video_path)
        input_video_stream = container.streams.video[0]
        total_frame_num = input_video_stream.frames

        max_spacing = (
            1 if self.seq_length == 1 else min(self.max_spacing, (total_frame_num - 1) // (self.seq_length - 1))
        )
        if max_spacing < 1:
            raise ValueError(f'seq_length > frames num in {video_path}')

        spacing = torch.randint(self.min_spacing, max_spacing + 1, size=()).item()

        frame_span = (self.seq_length - 1) * spacing + 1
        max_start_index = total_frame_num - frame_span
        start_index = torch.randint(max_start_index + 1, size=()).item()
        sampled_idxs = set(range(start_index, start_index + frame_span, spacing))

        frame_id = 0
        sampled_frames = []
        container.seek(0, backward=False, any_frame=True)
        for packet in container.demux(input_video_stream):
            for frame in packet.decode():
                if frame_id in sampled_idxs:
                    img = frame.to_rgb().to_ndarray()
                    img_resized = cv2.resize(img, (self.width, self.height))
                    tensor_frame = torch.from_numpy(img_resized)
                    tensor_frame = einops.rearrange(tensor_frame, "h w c -> c h w")
                    tensor_frame = 2 * tensor_frame.to(torch.float32) / 255 - 1
                    sampled_frames.append(tensor_frame)
                frame_id += 1
        
        close_video(container)
        assert frame_id >= total_frame_num, 'frame num error'
        return sampled_frames, spacing
        
    def __getitem__(self, index: int) -> dict:
        video_path = self.video_paths[index]
        frames, spacing = self.sample_frames(video_path)
        video = torch.stack(frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return dict(video=video, spacing=spacing)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getstate__(self):
        return dict(self.__dict__)


@dataclass
class VideoDatasetPerImage(Dataset):
    dataset_path: str
    height: int
    width: int
    mm_dir: str = None
    seq_length: int = 1
    x_flip: bool = False
    video_key: str = 'videos'

    def __post_init__(self):

        assert os.path.exists(self.dataset_path), self.dataset_path

        self.start_frames = []
        with open(self.dataset_path) as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                for video_path in data[self.video_key]:
                    if self.mm_dir is not None:
                        video_path = os.path.join(self.mm_dir, video_path)
                    container = load_video(video_path)
                    input_video_stream = container.streams.video[0]
                    total_frame_num = input_video_stream.frames
                    num_samples_from_source = total_frame_num - self.seq_length + 1
                    for start_frame in range(0, num_samples_from_source):
                        self.start_frames.append((video_path, start_frame, num_samples_from_source))
                    close_video(container)

    def read_frames(self, video_path, start_index):
        container = load_video(video_path)
        input_video_stream = container.streams.video[0]
        sampled_idxs = set(range(start_index, start_index + self.seq_length))

        frame_id = 0
        sampled_frames = []
        container.seek(0, backward=False, any_frame=True)
        for packet in container.demux(input_video_stream):
            for frame in packet.decode():
                if frame_id in sampled_idxs:
                    img = frame.to_rgb().to_ndarray()
                    img_resized = cv2.resize(img, (self.width, self.height))
                    tensor_frame = torch.from_numpy(img_resized)
                    tensor_frame = einops.rearrange(tensor_frame, "h w c -> c h w")
                    tensor_frame = 2 * tensor_frame.to(torch.float32) / 255 - 1
                    sampled_frames.append(tensor_frame)
                frame_id += 1
        
        close_video(container)
        return sampled_frames

    def __getitem__(self, index: int) -> dict:
        video_path, start_frame, num_samples_from_source = self.start_frames[index]
        frames = self.read_frames(video_path, start_frame)
        video = torch.stack(frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return dict(video=video, num_samples_from_source=num_samples_from_source)

    def __len__(self) -> int:
        return len(self.start_frames)