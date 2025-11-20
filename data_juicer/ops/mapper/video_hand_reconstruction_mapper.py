import os
import subprocess
import sys

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

OP_NAME = "video_hand_reconstruction_mapper"


# LazyLoader.check_packages(["numpy==1.26"])
# To output visual overlay images, it is necessary to install pyrender.
# Note that pyrender requires numpy==1.26 to correctly generate rendering results.

numpy = LazyLoader("numpy")
cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoHandReconstructionMapper(Mapper):
    """Use the WiLoR model for hand localization and
    reconstruction."""

    _accelerator = "cuda"

    def __init__(
        self,
        wilor_model_path: str = "wilor_final.ckpt",
        wilor_model_config: str = "model_config.yaml",
        detector_model_path: str = "detector.pt",
        mano_right_path: str = "path_to_mano_right_pkl",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        batch_size: int = 16,
        tag_field_name: str = MetaKeys.hand_reconstruction_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_save_visualization: bool = True,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_save_mesh: bool = True,
        save_mesh_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param wilor_model_path: The path to 'wilor_final.ckpt'.
        :param wilor_model_config: The path to 'model_config.yaml' for the
            WiLOR model.
        :param detector_model_path: The path to 'detector.pt' for the WiLOR
            model.
        :param mano_right_path: The path to 'MANO_RIGHT.pkl'. Users need to
            download this file from https://mano.is.tue.mpg.de/ and comply
            with the MANO license.
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
        :param batch_size: Batch size for simultaneous hand inference.
        :param tag_field_name: The field name to store the tags. It's
            "hand_reconstruction_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_save_visualization: Whether to save overlay images.
        :param save_visualization_dir: The path for saving overlay images.
        :param if_save_mesh: Whether to save images of the hand mesh.
        :param save_mesh_dir: The path for saving images of the hand mesh.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        LazyLoader.check_packages(["chumpy @ git+https://github.com/mattloper/chumpy"])
        LazyLoader.check_packages(["smplx==0.1.28", "yacs", "timm", "pyrender", "pytorch_lightning"])
        LazyLoader.check_packages(["scikit-image"], pip_args=["--no-deps"])

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "frame_key": MetaKeys.video_frames,
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])
        self.model_key = prepare_model(
            model_type="wilor",
            wilor_model_path=wilor_model_path,
            wilor_model_config=wilor_model_config,
            detector_model_path=detector_model_path,
            mano_right_path=mano_right_path,
        )

        wilor_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "WiLoR")
        if not os.path.exists(wilor_repo_path):
            subprocess.run(["git", "clone", "https://github.com/rolpotamias/WiLoR.git", wilor_repo_path], check=True)

        sys.path.append(wilor_repo_path)
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        self.ViTDetDataset = ViTDetDataset
        self.cam_crop_to_full = cam_crop_to_full
        self.recursive_to = recursive_to

        self.frame_num = frame_num
        self.duration = duration
        self.batch_size = batch_size
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.if_save_mesh = if_save_mesh
        self.save_mesh_dir = save_mesh_dir

    def project_full_img(self, points, cam_trans, focal_length, img_res):
        camera_center = [img_res[0] / 2.0, img_res[1] / 2.0]
        K = torch.eye(3)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = camera_center[0]
        K[1, 2] = camera_center[1]
        points = points + cam_trans
        points = points / points[..., -1:]

        V_2d = (K @ points.T).T
        return V_2d[..., :-1]

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

        temp_frame_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]
        frames_root = os.path.join(self.frame_dir, temp_frame_name)
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])

        wilor_model, detector, model_cfg, renderer = get_model(self.model_key, rank, self.use_cuda())
        if rank is not None:
            device = f"cuda:{rank}" if self.use_cuda() else "cpu"
        else:
            device = "cuda" if self.use_cuda() else "cpu"

        if self.if_save_visualization:
            visualization_frame_dir = os.path.join(self.save_visualization_dir, temp_frame_name)
            os.makedirs(visualization_frame_dir, exist_ok=True)

        if self.if_save_mesh:
            mesh_frame_dir = os.path.join(self.save_mesh_dir, temp_frame_name)
            os.makedirs(mesh_frame_dir, exist_ok=True)

        final_all_verts = []
        final_all_cam_t = []
        final_all_right = []
        final_all_joints = []
        final_all_kpts = []
        for img_path in frames_path:
            img_cv2 = cv2.imread(img_path)
            detections = detector(img_cv2, conf=0.3, verbose=False)[0]
            bboxes = []
            is_right = []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            if len(bboxes) == 0:
                final_all_verts.append([])
                final_all_cam_t.append([])
                final_all_right.append([])
                final_all_joints.append([])
                final_all_kpts.append([])
                continue

            boxes = numpy.stack(bboxes)
            right = numpy.stack(is_right)
            dataset = self.ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            all_joints = []
            all_kpts = []

            for batch in dataloader:
                batch = self.recursive_to(batch, device)

                with torch.no_grad():
                    out = wilor_model(batch)

                multiplier = 2 * batch["right"] - 1
                pred_cam = out["pred_cam"]
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = (
                    self.cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))

                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()

                    is_right = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                    cam_t = pred_cam_t_full[n]
                    kpts_2d = self.project_full_img(verts, cam_t, scaled_focal_length, img_size[n])

                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)
                    all_joints.append(joints)
                    all_kpts.append(kpts_2d)

                    # Save all meshes to disk
                    if self.if_save_mesh:
                        camera_translation = cam_t.copy()
                        tmesh = renderer.vertices_to_trimesh(
                            verts, camera_translation, (0.25098039, 0.274117647, 0.65882353), is_right=is_right
                        )
                        tmesh.export(os.path.join(mesh_frame_dir, f"{img_fn}_{n}.obj"))

            final_all_verts.append(all_verts)
            final_all_cam_t.append(all_cam_t)
            final_all_right.append(all_right)
            final_all_joints.append(all_joints)
            final_all_kpts.append(all_kpts)

            # Render front view
            if len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=(0.25098039, 0.274117647, 0.65882353),
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args
                )

                # Overlay image
                input_img = img_cv2.astype(numpy.float32)[:, :, ::-1] / 255.0
                input_img = numpy.concatenate(
                    [input_img, numpy.ones_like(input_img[:, :, :1])], axis=2
                )  # Add alpha channel
                input_img_overlay = (
                    input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                )

                cv2.imwrite(os.path.join(visualization_frame_dir, f"{img_fn}.jpg"), 255 * input_img_overlay[:, :, ::-1])

        sample[Fields.meta][self.tag_field_name] = {
            "vertices": final_all_verts,
            "camera_translation": final_all_cam_t,
            "if_right_hand": final_all_right,
            "joints": final_all_joints,
            "keypoints": final_all_kpts,
        }

        return sample
