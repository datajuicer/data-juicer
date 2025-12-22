# file: depth_anything_v2_mapper.py

import os
import glob
import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.ops.mapper.video_extract_frames_mapper import VideoExtractFramesMapper

OP_NAME = "depth_anything_v2_mapper"

torch = LazyLoader("torch")
cv2 = LazyLoader("cv2")
matplotlib = LazyLoader("matplotlib")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class DepthAnythingV2Mapper(Mapper):
    """Input a video and use DepthAnythingV2 to extract depth maps for all frames.
    
    - The operator processes a video and extracts frames based on the specified frame number
      and duration.
    - It uses the DepthAnythingV2 model to analyze each frame and generate depth maps.
    - The depth maps are saved as .npz files and paths are stored in metadata.
    - The operator can work with different encoder types (vits, vitb, vitl, vitg).
    """

    _accelerator = "cuda"

    def __init__(
        self,
        checkpoint_path: str = "./checkpoints",
        encoder: str = "vitl",
        input_size: int = 518,
        frame_num: int = 10,  # 改为 int 类型，默认值改为正整数
        duration: float = 0,
        tag_field_name: str = "depth_anything_v2_tags",
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        output_dir: str = "./depth_outputs",
        grayscale: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param checkpoint_path: Path to the DepthAnythingV2 checkpoints directory.
        :param encoder: Encoder type for DepthAnythingV2. 
            Options: ['vits', 'vitb', 'vitl', 'vitg'].
        :param input_size: Input size for the model. Default is 518.
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
        :param tag_field_name: The field name to store the tags. Default is
            "depth_anything_v2_tags".
        :param frame_dir: Output directory to save extracted frames.
        :param output_dir: Output directory to save depth maps (.npz files).
        :param grayscale: Whether to save grayscale depth maps instead of 
            applying colorful palette.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)

        self.checkpoint_path = checkpoint_path
        self.encoder = encoder
        self.input_size = input_size
        self.frame_num = frame_num
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.output_dir = output_dir
        self.grayscale = grayscale
        

        # 定义仓库保存路径
        depth_anything_v2_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "Depth-Anything-V2")

        # 如果仓库不存在，则克隆
        if not os.path.exists(depth_anything_v2_repo_path):
            import subprocess
            subprocess.run([
                "git",
                "clone",
                "https://github.com/DepthAnything/Depth-Anything-V2.git",
                depth_anything_v2_repo_path,
            ], check=True)
            
        import sys
        sys.path.append(depth_anything_v2_repo_path)

        # Create VideoExtractFramesMapper instance
        self.video_extractor = VideoExtractFramesMapper(
            frame_sampling_method="uniform" if frame_num > 0 else "all",
            frame_num=frame_num if frame_num > 0 else 1,  # 确保传递正整数
            duration=duration,
            frame_dir=frame_dir,
            frame_key=MetaKeys.video_frames,
            num_proc=1  # Force single process
        )

        # Prepare model configs
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # 移除 prepare_model 调用，直接在 process_single 中加载模型
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def process_single(self, sample, rank=None):
        """
        Process a single video sample with DepthAnythingV2.
        
        :param sample: Input sample containing video information
        :param rank: Rank for multi-GPU processing
        :return: Processed sample with depth information added to metadata
        """

        # Check if it's already processed
        if self.tag_field_name in sample[Fields.meta]:
            print(f"Depth maps already exist for {sample[self.video_key]}, skipping")
            return sample

        # Check if there is video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            print("No video found in sample, skipping")
            return sample

        # Extract frames from video
        processed_sample = self.video_extractor.process_single(sample)
        
        # Get video name
        video_path = sample[self.video_key][0] if isinstance(sample[self.video_key], list) else sample[self.video_key]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Define output npz path
        output_npz_path = os.path.join(self.output_dir, f"{video_name}.npz")
        
        # Check if output already exists
        if os.path.exists(output_npz_path):
            print(f"Depth maps already exist at {output_npz_path}, loading...")
            # Load existing depth data
            depth_data = np.load(output_npz_path)
            depth_array = depth_data['depth']
        else:
            # Import DepthAnythingV2 model
            from .depth_anything_v2.dpt import DepthAnythingV2
            
            # Load model
            depth_anything = DepthAnythingV2(**self.model_configs[self.encoder])
            model_path = os.path.join(self.checkpoint_path, f"depth_anything_v2_{self.encoder}.pth")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
                
            depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            device = "cuda" if self.use_cuda() else "cpu"
            if rank is not None and self.use_cuda():
                device = f"cuda:{rank}"
                
            depth_anything = depth_anything.to(device).eval()
            
            # Get frames root and paths
            frames_root = os.path.join(self.frame_dir, video_name)
            frame_names = os.listdir(frames_root)
            frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])
            
            # Process frames to get depth maps
            depth_frames = []
            
            with torch.no_grad():
                for frame_path in frames_path:
                    # Read frame
                    raw_frame = cv2.imread(frame_path)
                    
                    # Infer depth
                    depth = depth_anything.infer_image(raw_frame, self.input_size)
                    
                    # Store the raw depth
                    depth_frames.append(depth.copy())
            
            # Convert to array
            depth_array = np.array(depth_frames)
            
            # Save depth maps as NPZ file
            np.savez_compressed(output_npz_path, depth=depth_array)
            print(f"Saved depth data to {output_npz_path} with shape {depth_array.shape}")

        # Store results in metadata
        sample[Fields.meta][self.tag_field_name] = {
            "depth_npz_path": output_npz_path,
            "depth_shape": depth_array.shape,
            "encoder": self.encoder,
            "input_size": self.input_size
        }

        return sample