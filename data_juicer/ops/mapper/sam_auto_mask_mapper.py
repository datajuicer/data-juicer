# file: sam_auto_mask_mapper.py

import os
import shutil
import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.ops.mapper.video_extract_frames_mapper import VideoExtractFramesMapper

OP_NAME = "sam_auto_mask_mapper"

torch = LazyLoader("torch")
cv2 = LazyLoader("cv2")
logger = LazyLoader("loguru.logger")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class SAMAutoMaskMapper(Mapper):
    """Input a video and use SAM1 and SAM2 to extract automatic masks for all frames.
    
    - The operator processes a video and extracts frames based on the specified frame number
      and duration.
    - It uses SAM1 to generate initial masks and SAM2 to propagate them across frames.
    - The masks are saved as .npy files and paths are stored in metadata.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        sam1_checkpoint_path: str = "./checkpoints/sam1/sam_vit_h_4b8939.pth",
        sam2_checkpoint_path: str = "./checkpoints/sam2/sam2_hiera_large.pt",
        sam2_model_cfg: str = "/mnt/workspace/junyuanxiao/code/AutoSeg-SAM2/sam2_configs/sam2_hiera_l.yaml",
        output_dir: str = "./sam_outputs",
        frame_num: int = 10,
        duration: float = 0,
        detect_stride: int = 10,
        batch_size: int = 20,
        level: str = "default",
        tag_field_name: str = "sam_auto_mask_tags",
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        use_other_level: bool = False,
        postnms: bool = True,
        pred_iou_thresh: float = 0.7,
        box_nms_thresh: float = 0.7,
        stability_score_thresh: float = 0.85,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param sam1_checkpoint_path: Path to the SAM1 checkpoint file.
        :param sam2_checkpoint_path: Path to the SAM2 checkpoint file.
        :param sam2_model_cfg: SAM2 model configuration file.
        :param output_dir: Output directory to save mask .npy files.
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
        :param detect_stride: Frame stride for detection.
        :param batch_size: Batch size for processing.
        :param level: Mask level ('default', 'small', 'middle', 'large').
        :param tag_field_name: The field name to store the tags. Default is
            "sam_auto_mask_tags".
        :param frame_dir: Output directory to save extracted frames.
        :param use_other_level: Whether to use masks from other levels.
        :param postnms: Whether to apply post-NMS filtering.
        :param pred_iou_thresh: Predicted IoU threshold for SAM1.
        :param box_nms_thresh: Box NMS threshold for SAM1.
        :param stability_score_thresh: Stability score threshold for SAM1.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)

        self.sam1_checkpoint_path = sam1_checkpoint_path
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_model_cfg = sam2_model_cfg
        self.output_dir = output_dir
        self.frame_num = frame_num
        self.duration = duration
        self.detect_stride = detect_stride
        self.batch_size = batch_size
        self.level = level
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.use_other_level = use_other_level
        self.postnms = postnms
        self.pred_iou_thresh = pred_iou_thresh
        self.box_nms_thresh = box_nms_thresh
        self.stability_score_thresh = stability_score_thresh


        # Clone AutoSeg-SAM2 repository if not exists
        auto_seg_sam2_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "AutoSeg-SAM2")
        if not os.path.exists(auto_seg_sam2_repo_path):
            import subprocess
            import time
            
            # Try multiple approaches to clone the repository
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # First try HTTPS
                    result = subprocess.run([
                        "git",
                        "clone",
                        "--depth=1",  # Shallow clone to reduce download size
                        "https://github.com/zrporz/AutoSeg-SAM2.git",
                        auto_seg_sam2_repo_path,
                    ], check=True, capture_output=True, text=True, timeout=300)
                    print("Successfully cloned AutoSeg-SAM2 repository via HTTPS")
                    break
                except subprocess.CalledProcessError as e:
                    print(f"HTTPS clone attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(5)  # Wait 5 seconds before retry
                        continue
                        
                    # If HTTPS fails, try HTTP
                    try:
                        result = subprocess.run([
                            "git",
                            "clone",
                            "--depth=1",
                            "http://github.com/zrporz/AutoSeg-SAM2.git",
                            auto_seg_sam2_repo_path,
                        ], check=True, capture_output=True, text=True, timeout=300)
                        print("Successfully cloned AutoSeg-SAM2 repository via HTTP")
                        break
                    except subprocess.CalledProcessError as e2:
                        print(f"HTTP clone also failed: {e2}")
                        if attempt == max_attempts - 1:
                            raise Exception("Failed to clone AutoSeg-SAM2 repository after all attempts")
        
        # Add the repository path to sys.path
        import sys
        sys.path.append(auto_seg_sam2_repo_path)
        
        # Print debug information
        print(f"AutoSeg-SAM2 repo path: {auto_seg_sam2_repo_path}")
        print(f"Repository exists: {os.path.exists(auto_seg_sam2_repo_path)}")
        if os.path.exists(auto_seg_sam2_repo_path):
            print(f"Repository contents: {os.listdir(auto_seg_sam2_repo_path)}")


        # Create VideoExtractFramesMapper instance
        self.video_extractor = VideoExtractFramesMapper(
            frame_sampling_method="uniform" if frame_num > 0 else "all",
            frame_num=frame_num if frame_num > 0 else 1,
            duration=duration,
            frame_dir=frame_dir,
            frame_key=MetaKeys.video_frames,
            num_proc=1  # Force single process
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models (will be loaded in process_single)
        self.predictor = None
        self.mask_generator = None

    def _load_models(self):
        """Load SAM1 and SAM2 models."""
        if self.predictor is not None and self.mask_generator is not None:
            return
            
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            # from .sam2.build_sam2_video_predictor import build_sam2_video_predictor
            from .sam2.build_sam import build_sam2_video_predictor, build_sam2
            import os

            # 处理配置文件路径
            # 检查是否提供了绝对路径
            if os.path.isabs(self.sam2_model_cfg):
                config_path = self.sam2_model_cfg
            else:
                # 尝试在 AutoSeg-SAM2 项目中查找配置文件
                possible_paths = [
                    f"/mnt/workspace/junyuanxiao/code/AutoSeg-SAM2/sam2_configs/{self.sam2_model_cfg}",
                    f"./sam2_configs/{self.sam2_model_cfg}",
                    f"/workspace/sam2_configs/{self.sam2_model_cfg}",
                    self.sam2_model_cfg  # 最后尝试原始路径
                ]

                config_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        config_path = path
                        break
                
                if config_path is None:
                    # 如果都没找到，使用默认路径
                    config_path = f"/mnt/workspace/junyuanxiao/code/AutoSeg-SAM2/sam2_configs/{self.sam2_model_cfg}"
            
            print(f"Using SAM2 config path: {config_path}")
            print(f"Config file exists: {os.path.exists(config_path)}")
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"SAM2 config file not found: {config_path}")


            # Load SAM2 predictor
            self.predictor = build_sam2_video_predictor(
                config_path, 
                self.sam2_checkpoint_path
            )

            # # Load SAM2 predictor
            # self.predictor = build_sam2_video_predictor(
            #     self.sam2_model_cfg, 
            #     self.sam2_checkpoint_path
            # )
            
            # Load SAM1 model
            sam = sam_model_registry["vit_h"](checkpoint=self.sam1_checkpoint_path).to('cuda')
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=self.pred_iou_thresh,
                box_nms_thresh=self.box_nms_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=1,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100
            )
        except ImportError as e:
            raise ImportError(f"Required packages not found: {e}. Please install segment-anything and SAM2.")

    def _mask_nms(self, masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
        """Non-Maximum Suppression for masks."""
        scores, idx = scores.sort(0, descending=True)
        num_masks = idx.shape[0]
        masks_ord = masks[idx.view(-1), :]
        masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)
        iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
        inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
        
        for i in range(num_masks):
            for j in range(i, num_masks):
                intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
                union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
                iou = intersection / union
                iou_matrix[i, j] = iou
                if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                    inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                    inner_iou_matrix[i, j] = inner_iou
                if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                    inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                    inner_iou_matrix[j, i] = inner_iou
                    
        iou_matrix.triu_(diagonal=1)
        iou_max, _ = iou_matrix.max(dim=0)
        inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
        inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
        inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
        inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
        keep = iou_max <= iou_thr
        keep_conf = scores > score_thr
        keep_inner_u = inner_iou_max_u <= 1 - inner_thr
        keep_inner_l = inner_iou_max_l <= 1 - inner_thr
        
        if keep_conf.sum() == 0 and len(scores) > 0:
            index = scores.topk(min(3, len(scores))).indices
            keep_conf[index, 0] = True
        if keep_inner_u.sum() == 0 and len(scores) > 0:
            index = scores.topk(min(3, len(scores))).indices
            keep_inner_u[index, 0] = True
        if keep_inner_l.sum() == 0 and len(scores) > 0:
            index = scores.topk(min(3, len(scores))).indices
            keep_inner_l[index, 0] = True
            
        keep *= keep_conf
        keep *= keep_inner_u
        keep *= keep_inner_l
        selected_idx = idx[keep]
        return selected_idx

    def _filter_masks(self, keep, masks_result):
        """Filter masks based on NMS results."""
        keep = keep.int().cpu().numpy()
        result_keep = []
        for i, m in enumerate(masks_result):
            if i in keep: 
                result_keep.append(m)
        return result_keep

    def _update_masks(self, *args, **kwargs):
        """Update masks with NMS filtering."""
        masks_new = ()
        for masks_lvl in args:
            if not masks_lvl: 
                continue
            seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
            scores = stability * iou_pred
            keep_mask_nms = self._mask_nms(seg_pred, scores, **kwargs)
            masks_lvl = self._filter_masks(keep_mask_nms, masks_lvl)
            masks_new += (masks_lvl,)
        return masks_new

    def _search_new_objects(self, masks_from_prev, mask_list, other_masks_list=None, mask_ratio_thresh=0, ratio=0.5, area_threshold=5000):
        """Search for new objects in the current frame."""
        new_mask_list = []
        if not masks_from_prev:
            return [m for m in mask_list if m['segmentation'].sum() > area_threshold]
            
        mask_none = ~masks_from_prev[0].copy()[0]
        for prev_mask in masks_from_prev[1:]:
            mask_none &= ~prev_mask[0]
            
        for mask in mask_list:
            seg = mask['segmentation']
            if (mask_none & seg).sum() / seg.sum() > ratio and seg.sum() > area_threshold:
                new_mask_list.append(mask)
                
        for mask in new_mask_list:
            mask_none &= ~mask['segmentation']
            
        if other_masks_list is not None:
            for mask in other_masks_list:
                if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh:
                    seg = mask['segmentation']
                    if (mask_none & seg).sum() / seg.sum() > ratio and seg.sum() > area_threshold:
                        new_mask_list.append(mask)
                        mask_none &= ~seg
                else:
                    break
                    
        return new_mask_list

    def _calculate_no_mask_area_ratio(self, out_mask_list):
        """Calculate the ratio of area without masks."""
        if not out_mask_list or len(out_mask_list) == 0: 
            return 1.0
        if out_mask_list[0].size == 0: 
            return 1.0
            
        h = out_mask_list[0].shape[1]
        w = out_mask_list[0].shape[2]
        mask_none = ~out_mask_list[0].copy()
        for prev_mask in out_mask_list[1:]:
            mask_none &= ~prev_mask
        return(mask_none.sum() / (h * w))


    def _get_video_segments(self, prompts_loader, inference_state, step, start_frame_idx, final_output=False):
        """Get video segments from predictor."""
        video_segments = {}
        for batch_prompts in prompts_loader:
            self.predictor.reset_state(inference_state)
            for id, prompt_list in batch_prompts.items():
                for prompt in prompt_list:
                    self.predictor.add_new_mask(
                        inference_state=inference_state, 
                        frame_idx=prompt[0], 
                        obj_id=id, 
                        mask=prompt[1]
                    )
                    
            propagation_params = {"inference_state": inference_state}
            if final_output:
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(**propagation_params):
                    if out_frame_idx not in video_segments: 
                        video_segments[out_frame_idx] = {}
                    for i, out_obj_id in enumerate(out_obj_ids): 
                        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                        
                propagation_params["reverse"] = True
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(**propagation_params):
                    if out_frame_idx in video_segments:
                        for i, out_obj_id in enumerate(out_obj_ids): 
                            video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
            else:
                propagation_params.update({
                    "max_frame_num_to_track": step, 
                    "start_frame_idx": start_frame_idx
                })
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(**propagation_params):
                    if out_frame_idx not in video_segments: 
                        video_segments[out_frame_idx] = {}
                    for i, out_obj_id in enumerate(out_obj_ids): 
                        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                        
        return video_segments

    def _prepare_frames_for_sam2(self, frames_root):
        """
        准备符合 SAM2 要求的帧文件（创建临时目录并复制重命名文件）
        """
        import re
        import shutil
        
        # 创建临时目录
        temp_dir = os.path.join(frames_root, "sam2_temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # 如果存在就删除
        os.makedirs(temp_dir, exist_ok=True)
        
        # 获取所有帧文件（排除临时目录本身）
        frame_files = [f for f in os.listdir(frames_root) 
                    if (f.endswith('.jpg') or f.endswith('.png')) and f != "sam2_temp"]
        
        # 按照原始顺序排序
        frame_files.sort()
        
        print(f"Preparing {len(frame_files)} frames for SAM2 in temp directory")
        
        # 复制并重命名文件为纯数字格式
        copied_files = []
        for i, frame_file in enumerate(frame_files):
            src_path = os.path.join(frames_root, frame_file)
            # 使用索引作为文件名（确保唯一性和正确排序）
            new_name = f"{i}.jpg"
            dst_path = os.path.join(temp_dir, new_name)
            
            # 复制文件
            try:
                shutil.copy2(src_path, dst_path)
                copied_files.append((frame_file, new_name))
                print(f"Copied {frame_file} to {new_name}")
            except Exception as e:
                print(f"Failed to copy {frame_file} to {new_name}: {e}")
        
        print(f"Created {len(copied_files)} files in {temp_dir}")
        return temp_dir

    def process_single(self, sample, rank=None):
        """
        Process a single video sample with SAM auto-masking.
        """
        
        # Check if it's already processed
        if self.tag_field_name in sample[Fields.meta]:
            print(f"SAM masks already exist for {sample[self.video_key]}, skipping")
            return sample

        # Check if there is video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            print("No video found in sample, skipping")
            return sample

        # Load models if not already loaded
        self._load_models()

        # Extract frames from video
        processed_sample = self.video_extractor.process_single(sample)
        
        # Get video name
        video_path = sample[self.video_key][0] if isinstance(sample[self.video_key], list) else sample[self.video_key]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Define output npy path
        output_npy_path = os.path.join(self.output_dir, f"{video_name}.npy")
        
        # Check if output already exists
        if os.path.exists(output_npy_path):
            print(f"SAM masks already exist at {output_npy_path}, loading...")
            # Load existing mask data
            mask_array = np.load(output_npy_path)
        else:
            # Get frames root and paths
            frames_root = os.path.join(self.frame_dir, video_name)
            print(f"Processing frames in: {frames_root}")
            
            # List files before processing
            original_files = os.listdir(frames_root)
            print(f"Original files: {sorted(original_files)}")
            
            # 创建符合 SAM2 要求的临时帧目录
            sam2_frames_root = self._prepare_frames_for_sam2(frames_root)
            
            # 验证临时目录中的文件
            temp_files = os.listdir(sam2_frames_root)
            print(f"Temp directory files: {sorted(temp_files)}")
            
            num_frames = len([f for f in temp_files if f.endswith('.jpg')])
            if num_frames == 0:
                print(f"No frames prepared for SAM2 in {video_path}. Skipping.")
                # 清理临时目录
                if os.path.exists(sam2_frames_root):
                    import shutil
                    shutil.rmtree(sam2_frames_root)
                return sample
                
            # Initialize video predictor state with temp directory
            print(f"Initializing SAM2 with frames from: {sam2_frames_root}")
            inference_state = self.predictor.init_state(video_path=sam2_frames_root)
            
            # Get frame paths for SAM1 processing from temp directory
            temp_frame_names = [f for f in os.listdir(sam2_frames_root) if f.endswith('.jpg')]
            temp_frames_path = sorted([os.path.join(sam2_frames_root, frame_name) for frame_name in temp_frame_names])
            print(f"SAM1 will process frames: {temp_frames_path}")
            
            # Process frames
            masks_from_prev, is_key_frame, mask_ratio_thresh = [], True, 0.0
            prompts_loader = Prompts(bs=self.batch_size)
            
            for now_frame in range(0, num_frames, self.detect_stride):
                print(f"Processing frame stride starting at: {now_frame}")
                
                if is_key_frame:
                    print("Keyframe detected. Generating new masks with SAM1.")
                    sum_id = prompts_loader.get_obj_num()
                    # 注意：这里使用临时目录中的帧路径
                    image_path = temp_frames_path[now_frame]
                    print(f"Reading image: {image_path}")
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image {image_path}")
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(image)
                    
                    if self.postnms:
                        masks_default, masks_s, masks_m, masks_l = self._update_masks(
                            masks_default, masks_s, masks_m, masks_l, 
                            iou_thr=0.8, score_thr=0.7, inner_thr=0.5
                        )
                        
                    level_map = {
                        'default': (masks_default, masks_l + masks_m + masks_s), 
                        'small': (masks_s, None), 
                        'middle': (masks_m, masks_s), 
                        'large': (masks_l, masks_s + masks_m)
                    }
                    masks, other_masks = level_map[self.level]
                    if not self.use_other_level: 
                        other_masks = None
                        
                if now_frame == 0:
                    if masks:
                        for i, mask_data in enumerate(masks): 
                            prompts_loader.add(i, 0, mask_data['segmentation'])
                else:  
                    for id, mask in enumerate(masks_from_prev):
                        if mask.sum() > 0: 
                            prompts_loader.add(id, now_frame, mask[0])
                    if is_key_frame and masks:
                        new_mask_list = self._search_new_objects(
                            masks_from_prev, masks, other_masks, mask_ratio_thresh
                        )
                        for i, new_mask_data in enumerate(new_mask_list): 
                            prompts_loader.add(sum_id + i, now_frame, new_mask_data['segmentation'])
                                
                print(f"Total objects being tracked: {prompts_loader.get_obj_num()}")
                
                # video_segments = self._get_video_segments(
                #     [prompts_loader], inference_state, self.detect_stride, now_frame
                # )


                # 应该改为:
                video_segments = self._get_video_segments(
                    prompts_loader, inference_state, self.detect_stride, now_frame
                )



                out_frame_idx = now_frame + self.detect_stride
                if out_frame_idx >= num_frames: 
                    break
                    
                out_mask_list = list(video_segments.get(out_frame_idx, {}).values())
                no_mask_ratio = self._calculate_no_mask_area_ratio(out_mask_list)
                
                if now_frame == 0: 
                    mask_ratio_thresh = no_mask_ratio
                    
                if no_mask_ratio > mask_ratio_thresh + 0.01: 
                    is_key_frame, mask_ratio_thresh = True, no_mask_ratio
                else: 
                    is_key_frame = False
                    
                masks_from_prev = out_mask_list
                print(f"Current no-mask ratio: {no_mask_ratio:.4f}, Threshold: {mask_ratio_thresh:.4f}, Is next a keyframe? {is_key_frame}")
            
            print("Starting final propagation and mask collection.")
            # final_video_segments = self._get_video_segments(
            #     [prompts_loader], inference_state, self.detect_stride, 0, final_output=True
            # )


            # 应该改为:
            final_video_segments = self._get_video_segments(
                prompts_loader, inference_state, self.detect_stride, 0, final_output=True
            )


            # Collect all frames masks
            all_frames_masks = []
            
            # Get frame dimensions
            mask_h, mask_w = 0, 0
            for frame_data in final_video_segments.values():
                if frame_data:
                    first_mask = next(iter(frame_data.values()))
                    if first_mask.ndim == 3: 
                        mask_h, mask_w = first_mask.shape[1], first_mask.shape[2]
                    else: 
                        mask_h, mask_w = first_mask.shape[0], first_mask.shape[1]
                    break

            for out_frame_idx in range(num_frames):
                frame_masks_padded = []
                if out_frame_idx in final_video_segments:
                    sorted_masks = sorted(
                        final_video_segments[out_frame_idx].values(), 
                        key=lambda m: m.sum(), reverse=True
                    )
                    top_masks = sorted_masks[:3]
                    for mask in top_masks:
                        if mask.ndim == 3: 
                            frame_masks_padded.append(mask.squeeze(0))
                        else: 
                            frame_masks_padded.append(mask)
                
                # Pad to 3 masks if needed
                num_to_pad = 3 - len(frame_masks_padded)
                for _ in range(num_to_pad):
                    frame_masks_padded.append(np.zeros((mask_h, mask_w), dtype=bool))
                
                all_frames_masks.append(np.stack(frame_masks_padded, axis=0))

            mask_array = np.stack(all_frames_masks, axis=0)
            print(f"Final mask array created with shape: {mask_array.shape}")

            # Save mask array as NPZ file
            np.save(output_npy_path, mask_array)
            print(f"Successfully saved final masks to {output_npy_path}")
            
            # 清理临时目录
            if os.path.exists(sam2_frames_root):
                import shutil
                shutil.rmtree(sam2_frames_root)
                print(f"Cleaned up temporary directory: {sam2_frames_root}")

        # Store results in metadata
        sample[Fields.meta][self.tag_field_name] = {
            "mask_npy_path": output_npy_path,
            "mask_shape": mask_array.shape,
            "detect_stride": self.detect_stride,
            "batch_size": self.batch_size,
            "level": self.level
        }

        return sample


class Prompts:
    """Helper class to manage prompts for SAM2."""
    
    def __init__(self, bs: int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []
        
    def add(self, obj_id, frame_id, mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else: 
            new_obj = False
            
        self.prompts[obj_id].append((frame_id, mask))
        if frame_id not in self.key_frame_list and new_obj:
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            
    def get_obj_num(self): 
        return len(self.obj_list)
        
    def __iter__(self): 
        self.start_idx = 0
        self.iter_frameindex = 0
        return self
        
    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1 or not self.key_frame_list:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]: 
                    end_idx = self.start_idx+self.batch_size
                else: 
                    end_idx = self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex += 1
                    
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        else: 
            raise StopIteration

    # 添加 items 方法以兼容字典操作
    def items(self):
        """Return items like a dictionary"""
        return self.prompts.items()