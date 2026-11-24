# Copied from https://github.com/ThunderVVV/HaWoR/tree/main

import logging
import math
import os
import pickle
import sys
from inspect import isfunction
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from data_juicer.ops.common import hawor_constants as constants
from data_juicer.ops.common.hawor_func_vit import vit
from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2", "opencv-contrib-python")
torch = LazyLoader("torch")
F = LazyLoader("torch.nn.functional")
smplx = LazyLoader("smplx")
yacs = LazyLoader("yacs")
einops = LazyLoader("einops")
pl = LazyLoader("pytorch_lightning")
torchvision = LazyLoader("torchvision")
nn = LazyLoader("torch.nn")
skimage = LazyLoader("skimage", "scikit-image")
timm = LazyLoader("timm")
scipy = LazyLoader("scipy")


_C = yacs.config.CfgNode(new_allowed=True)

_C.GENERAL = yacs.config.CfgNode(new_allowed=True)
_C.GENERAL.RESUME = True
_C.GENERAL.TIME_TO_RUN = 3300
_C.GENERAL.VAL_STEPS = 100
_C.GENERAL.LOG_STEPS = 100
_C.GENERAL.CHECKPOINT_STEPS = 20000
_C.GENERAL.CHECKPOINT_DIR = "checkpoints"
_C.GENERAL.SUMMARY_DIR = "tensorboard"
_C.GENERAL.NUM_GPUS = 1
_C.GENERAL.NUM_WORKERS = 4
_C.GENERAL.MIXED_PRECISION = True
_C.GENERAL.ALLOW_CUDA = True
_C.GENERAL.PIN_MEMORY = False
_C.GENERAL.DISTRIBUTED = False
_C.GENERAL.LOCAL_RANK = 0
_C.GENERAL.USE_SYNCBN = False
_C.GENERAL.WORLD_SIZE = 1

_C.TRAIN = yacs.config.CfgNode(new_allowed=True)
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.WARMUP = False
_C.TRAIN.NORMALIZE_PER_IMAGE = False
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.CLIP_GRAD_VALUE = 1.0
_C.LOSS_WEIGHTS = yacs.config.CfgNode(new_allowed=True)

_C.DATASETS = yacs.config.CfgNode(new_allowed=True)

_C.MODEL = yacs.config.CfgNode(new_allowed=True)
_C.MODEL.IMAGE_SIZE = 224

_C.EXTRA = yacs.config.CfgNode(new_allowed=True)
_C.EXTRA.FOCAL_LENGTH = 5000

_C.DATASETS.CONFIG = yacs.config.CfgNode(new_allowed=True)
_C.DATASETS.CONFIG.SCALE_FACTOR = 0.3
_C.DATASETS.CONFIG.ROT_FACTOR = 30
_C.DATASETS.CONFIG.TRANS_FACTOR = 0.02
_C.DATASETS.CONFIG.COLOR_SCALE = 0.2
_C.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
_C.DATASETS.CONFIG.TRANS_AUG_RATE = 0.5
_C.DATASETS.CONFIG.DO_FLIP = False
_C.DATASETS.CONFIG.FLIP_AUG_RATE = 0.5
_C.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.10


def default_config() -> yacs.config.CfgNode:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, merge: bool = True, update_cachedir: bool = False) -> yacs.config.CfgNode:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
        cfg = default_config()
    else:
        cfg = yacs.config.CfgNode(new_allowed=True)
    cfg.merge_from_file(config_file)

    if update_cachedir:

        def update_path(path: str) -> str:
            if os.path.basename("./_DATA") in path:
                return path
            if os.path.isabs(path):
                return path
            return os.path.join("./_DATA", path)

        cfg.MANO.MODEL_PATH = update_path(cfg.MANO.MODEL_PATH)
        cfg.MANO.MEAN_PARAMS = update_path(cfg.MANO.MEAN_PARAMS)

    cfg.freeze()
    return cfg


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == "vit":
        return vit(cfg)
    else:
        raise NotImplementedError("Backbone type is not implemented")


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = torch.nn.LayerNorm(data_dim)  # TODO: Check if elementwise_affine=True is correct
        self.linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, ..., data_dim)
        # t: (batch, norm_cond_dim)
        # return: (batch, data_dim)
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)

        # Add singleton dimensions to alpha and beta
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])

        return x * (1 + alpha) + beta


class FrequencyEmbedder(torch.nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)  # (N, D * 2 * num_frequencies + D)
        return embedded


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale + 1e-6
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0, asint=True):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)

    if asint:
        return new_pt[:2].astype(int) + 1
    else:
        return new_pt[:2] + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[old_y[0] : old_y[1], old_x[0] : old_x[1]]
    except Exception as e:
        print(f"Error: {e}, invalid bbox, fill with 0")

    if not rot == 0:
        # Remove padding
        new_img = skimage.transform.rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = skimage.transform.resize(new_img, res)
    return new_img


def boxes_2_cs(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    size = np.stack([w, h]).max(axis=0)

    centers = np.stack([cx, cy], axis=1)
    scales = size / 200
    return centers, scales


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def perspective_projection(points, rotation, translation, focal_length, camera_center, distortion=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]

    # Extrinsic
    if rotation is not None:
        points = torch.einsum("bij,bkj->bki", rotation, points)

    if translation is not None:
        points = points + translation.unsqueeze(1)

    if distortion is not None:
        kc = distortion
        points = points[:, :, :2] / points[:, :, 2:]

        r2 = points[:, :, 0] ** 2 + points[:, :, 1] ** 2
        dx = 2 * kc[:, [2]] * points[:, :, 0] * points[:, :, 1] + kc[:, [3]] * (r2 + 2 * points[:, :, 0] ** 2)

        dy = 2 * kc[:, [3]] * points[:, :, 0] * points[:, :, 1] + kc[:, [2]] * (r2 + 2 * points[:, :, 1] ** 2)

        x = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 0] + dx
        y = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 1] + dy

        points = torch.stack([x, y, torch.ones_like(x)], dim=-1)

    # Intrinsic
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Apply camera intrinsicsrf
    points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum("bij,bkj->bki", K, points)
    projected_points = projected_points[:, :, :-1]

    return projected_points


def avg_rot(rot):
    # input [B,...,3,3] --> output [...,3,3]
    rot = rot.mean(dim=0)
    U, _, V = torch.svd(rot)
    rot = U @ V.transpose(-1, -2)
    return rot


def rot9d_to_rotmat(x):
    """Convert 9D rotation representation to 3x3 rotation matrix.
    Based on Levinson et al., "An Analysis of SVD for Deep Rotation Estimation"
    Input:
        (B,9) or (B,J*9) Batch of 9D rotation (interpreted as 3x3 est rotmat)
    Output:
        (B,3,3) or (B*J,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 3)
    u, _, vh = torch.linalg.svd(x)

    sig = torch.eye(3).expand(len(x), 3, 3).clone()
    sig = sig.to(x.device)
    sig[:, -1, -1] = (u @ vh).det()

    R = u @ sig @ vh

    return R


"""
Deprecated in favor of: rotation_conversions.py

Useful geometric operations, e.g. differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


# rot6d_to_rotmat_hmr2
def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(rotmat):
    """Inverse function of the above.
    Input:
        (B,3,3) Batch of corresponding rotation matrices
    Output:
        (B,6) Batch of 6-D rotation representations
    """
    # rot6d = rotmat[:, :, :2]
    rot6d = rotmat[..., :2]
    rot6d = rot6d.reshape(rot6d.size(0), -1)
    return rot6d


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = (
            torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(rot_mat.shape[0], -1, -1)
        )
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    :param rotation_matrix (N, 3, 3)
    """
    *dims, m, n = rotation_matrix.shape
    rmat_t = torch.transpose(rotation_matrix.reshape(-1, m, n), -1, -2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa  # noqa
    q *= 0.5
    return q.reshape(*dims, 4)


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000.0, img_size=224.0):
    """
    This function is borrowed from https://github.com/nkolot/SPIN/utils/geometry.py

    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length, focal_length])
    # optical center
    center = np.array([img_size / 2.0, img_size / 2.0])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O_ = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O_ - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O_) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000.0, img_size=224.0):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, -24:, :3].cpu().numpy()
    joints_2d = joints_2d[:, -24:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)


def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:, 0].max()
        max_y = valid_keypoints[:, 1].max()
        min_x = valid_keypoints[:, 0].min()
        min_y = valid_keypoints[:, 1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0, 0, 0


def render_keypoints(
    img: np.array,
    keypoints: np.array,
    pairs: List,
    colors: List,
    thickness_circle_ratio: float,
    thickness_line_ratio_wrt_circle: float,
    pose_scales: List,
    threshold: float = 0.1,
    alpha: float = 1.0,
) -> np.array:
    """
    Render keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.array): Keypoint array of shape (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors: (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image.
    """
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(int)
                keypoint2 = keypoints[index2, :-1].astype(int)
                cv2.line(
                    img,
                    tuple(keypoint1.tolist()),
                    tuple(keypoint2.tolist()),
                    tuple(color.tolist()),
                    thicknessLineScaled,
                    lineType,
                    shift,
                )
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(int)
                cv2.circle(
                    img,
                    tuple(center.tolist()),
                    radiusScaled,
                    tuple(color.tolist()),
                    thicknessCircleScaled,
                    lineType,
                    shift,
                )
    return img


def render_hand_keypoints(
    img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0
):
    if use_confidence and map_fn is not None:
        # thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1.0 / 50 * map_fn(right_hand_keypoints[:, -1])
    else:
        # thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1.0 / 50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        0,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        0,
        9,
        9,
        10,
        10,
        11,
        11,
        12,
        0,
        13,
        13,
        14,
        14,
        15,
        15,
        16,
        0,
        17,
        17,
        18,
        18,
        19,
        19,
        20,
    ]
    pairs = np.array(pairs).reshape(-1, 2)

    colors = [
        100.0,
        100.0,
        100.0,
        100.0,
        0.0,
        0.0,
        150.0,
        0.0,
        0.0,
        200.0,
        0.0,
        0.0,
        255.0,
        0.0,
        0.0,
        100.0,
        100.0,
        0.0,
        150.0,
        150.0,
        0.0,
        200.0,
        200.0,
        0.0,
        255.0,
        255.0,
        0.0,
        0.0,
        100.0,
        50.0,
        0.0,
        150.0,
        75.0,
        0.0,
        200.0,
        100.0,
        0.0,
        255.0,
        125.0,
        0.0,
        50.0,
        100.0,
        0.0,
        75.0,
        150.0,
        0.0,
        100.0,
        200.0,
        0.0,
        125.0,
        255.0,
        100.0,
        0.0,
        100.0,
        150.0,
        0.0,
        150.0,
        200.0,
        0.0,
        200.0,
        255.0,
        0.0,
        255.0,
    ]
    colors = np.array(colors).reshape(-1, 3)
    # colors = np.zeros_like(colors)
    poseScales = [1]
    # img = render_keypoints(img, left_hand_keypoints, pairs, colors, thicknessCircleRatioLeft, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    img = render_keypoints(
        img,
        right_hand_keypoints,
        pairs,
        colors,
        thicknessCircleRatioRight,
        thicknessLineRatioWRTCircle,
        poseScales,
        threshold,
        alpha=alpha,
    )
    # img = render_keypoints(img, right_hand_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)
    return img


def render_hand_landmarks(
    img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0
):
    if use_confidence and map_fn is not None:
        # thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1.0 / 50 * map_fn(right_hand_keypoints[:, -1])
    else:
        # thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1.0 / 50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = []
    pairs = np.array(pairs).reshape(-1, 2)

    colors = [255, 0, 0]
    colors = np.array(colors).reshape(-1, 3)
    # colors = np.zeros_like(colors)
    poseScales = [1]
    # img = render_keypoints(img, left_hand_keypoints, pairs, colors, thicknessCircleRatioLeft, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    img = render_keypoints(
        img,
        right_hand_keypoints,
        pairs,
        colors,
        thicknessCircleRatioRight * 0.1,
        thicknessLineRatioWRTCircle * 0.1,
        poseScales,
        threshold,
        alpha=alpha,
    )
    # img = render_keypoints(img, right_hand_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)
    return img


def render_body_keypoints(img: np.array, body_keypoints: np.array) -> np.array:
    """
    Render OpenPose body keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image.
    """

    thickness_circle_ratio = 1.0 / 75.0 * np.ones(body_keypoints.shape[0])
    thickness_line_ratio_wrt_circle = 0.75
    pairs = []
    pairs = [
        1,
        8,
        1,
        2,
        1,
        5,
        2,
        3,
        3,
        4,
        5,
        6,
        6,
        7,
        8,
        9,
        9,
        10,
        10,
        11,
        8,
        12,
        12,
        13,
        13,
        14,
        1,
        0,
        0,
        15,
        15,
        17,
        0,
        16,
        16,
        18,
        14,
        19,
        19,
        20,
        14,
        21,
        11,
        22,
        22,
        23,
        11,
        24,
    ]
    pairs = np.array(pairs).reshape(-1, 2)
    colors = [
        255.0,
        0.0,
        85.0,
        255.0,
        0.0,
        0.0,
        255.0,
        85.0,
        0.0,
        255.0,
        170.0,
        0.0,
        255.0,
        255.0,
        0.0,
        170.0,
        255.0,
        0.0,
        85.0,
        255.0,
        0.0,
        0.0,
        255.0,
        0.0,
        255.0,
        0.0,
        0.0,
        0.0,
        255.0,
        85.0,
        0.0,
        255.0,
        170.0,
        0.0,
        255.0,
        255.0,
        0.0,
        170.0,
        255.0,
        0.0,
        85.0,
        255.0,
        0.0,
        0.0,
        255.0,
        255.0,
        0.0,
        170.0,
        170.0,
        0.0,
        255.0,
        255.0,
        0.0,
        255.0,
        85.0,
        0.0,
        255.0,
        0.0,
        0.0,
        255.0,
        0.0,
        0.0,
        255.0,
        0.0,
        0.0,
        255.0,
        0.0,
        255.0,
        255.0,
        0.0,
        255.0,
        255.0,
        0.0,
        255.0,
        255.0,
    ]
    colors = np.array(colors).reshape(-1, 3)
    pose_scales = [1]
    return render_keypoints(
        img, body_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1
    )


def render_openpose(img: np.array, hand_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image.
    """
    # img = render_body_keypoints(img, body_keypoints)
    img = render_hand_keypoints(img, hand_keypoints)
    return img


def render_openpose_landmarks(img: np.array, hand_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image.
    """
    # img = render_body_keypoints(img, body_keypoints)
    img = render_hand_landmarks(img, hand_keypoints)
    return img


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, pl.utilities.rank_zero_only(getattr(logger, level)))

    return logger


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, context=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args):
        for attn, ff in self.layers:
            x = attn(x, *args) + x
            x = ff(x, *args) + x
        return x


class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        emb_dropout_loc: str = "token",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        token_pe_numfreq: int = -1,
    ):
        super().__init__()
        if token_pe_numfreq > 0:
            token_dim_new = token_dim * (2 * token_pe_numfreq + 1)
            self.to_token_embedding = nn.Sequential(
                einops.layers.torch.Rearrange("b n d -> (b n) d", n=num_tokens, d=token_dim),
                FrequencyEmbedder(token_pe_numfreq, token_pe_numfreq - 1),
                einops.layers.torch.Rearrange("(b n) d -> b n d", n=num_tokens, d=token_dim_new),
                nn.Linear(token_dim_new, dim),
            )
        else:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")
        self.emb_dropout_loc = emb_dropout_loc

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, norm=norm, norm_cond_dim=norm_cond_dim
        )

    def forward(self, inp: torch.Tensor, *args, **kwargs):
        x = inp

        if self.emb_dropout_loc == "input":
            x = self.dropout(x)
        x = self.to_token_embedding(x)

        if self.emb_dropout_loc == "token":
            x = self.dropout(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        if self.emb_dropout_loc == "token_afterpos":
            x = self.dropout(x)
        x = self.transformer(x, *args)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True")

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x


class MANOTransformerDecoderHead(nn.Module):
    """HMR2 Cross-attention based SMPL Transformer decoder"""

    def __init__(self, cfg):
        super().__init__()
        transformer_args = dict(
            depth=6,  # originally 6
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=1280,
            num_tokens=1,
            token_dim=1,
            dim=1024,
        )
        self.transformer = TransformerDecoder(**transformer_args)

        dim = 1024
        npose = 16 * 6
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        init_hand_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_hand_pose", init_hand_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # Pass through transformer
        token = torch.zeros(batch_size, 1, 1).to(x.device)
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1)  # (B, C)

        # Readout from token_out
        pred_pose = self.decpose(token_out) + init_hand_pose
        pred_shape = self.decshape(token_out) + init_betas
        pred_cam = self.deccam(token_out) + init_cam

        return pred_pose, pred_shape, pred_cam


class temporal_attention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, residual=False):
        super(temporal_attention, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.residual = residual
        self.l1 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        TranLayer = nn.TransformerEncoderLayer(
            d_model=hdim, nhead=nhead, dim_feedforward=1024, dropout=0.1, activation="gelu"
        )
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)

        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (b,t,c) -> (t,b,c)

        h = self.l1(x)
        h = self.pos_embedding(h)
        h = self.trans(h)
        h = self.l2(h)

        if self.residual:
            x = x[..., : self.out_dim] + h
        else:
            x = h
        x = x.permute(1, 0, 2)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TrackDatasetEval(torch.utils.data.Dataset):
    """
    Track Dataset Class - Load images/crops of the tracked boxes.
    """

    def __init__(
        self,
        imgfiles,
        boxes,
        crop_size=256,
        dilate=1.0,
        img_focal=None,
        img_center=None,
        normalization=True,
        item_idx=0,
        do_flip=False,
    ):
        super(TrackDatasetEval, self).__init__()

        self.imgfiles = imgfiles
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD),
            ]
        )

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center
        self.item_idx = item_idx
        self.do_flip = do_flip

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, index):
        item = {}
        imgfile = self.imgfiles[index]
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]

        img_focal = self.img_focal
        img_center = self.img_center

        img = cv2.imread(imgfile)[:, :, ::-1]
        if self.do_flip:
            img = img[:, ::-1, :]
            img_width = img.shape[1]
            center[0] = img_width - center[0] - 1
        img_crop = crop(img, center, scale, [self.crop_size, self.crop_size], rot=0).astype("uint8")
        # cv2.imwrite('debug_crop.png', img_crop[:,:,::-1])

        if self.normalization:
            img_crop = self.normalize_img(img_crop)
        else:
            img_crop = torch.from_numpy(img_crop)
        item["img"] = img_crop

        if self.do_flip:
            # center[0] = img_width - center[0] - 1
            item["do_flip"] = torch.tensor(1).float()
        item["img_idx"] = torch.tensor(index).long()
        item["scale"] = torch.tensor(scale).float()
        item["center"] = torch.tensor(center).float()
        item["img_focal"] = torch.tensor(img_focal).float()
        item["img_center"] = torch.tensor(img_center).float()

        return item


def dataset_config() -> yacs.config.CfgNode:
    """
    Get dataset config file
    Returns:
      CfgNode: Dataset config as a yacs CfgNode object.
    """
    cfg = yacs.config.CfgNode(new_allowed=True)
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets_tar.yaml")
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def parse_chunks(frame, boxes, min_len=16):
    """If a track disappear in the middle,
    we separate it to different segments to estimate the HPS independently.
    If a segment is less than 16 frames, we get rid of it for now.
    """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk) >= min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk == breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk) >= min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks


def interpolate_bboxes(bboxes):

    zero_indices = np.where(np.all(bboxes == 0, axis=1))[0]

    non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]

    if len(zero_indices) == 0:
        return bboxes

    interpolated_bboxes = bboxes.copy()
    for i in range(5):
        interp_func = scipy.interpolate.interp1d(
            non_zero_indices, bboxes[non_zero_indices, i], kind="linear", fill_value="extrapolate"
        )
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)

    return interpolated_bboxes


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


class MANO(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANO, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        # 2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"), dtype=torch.float32),
            )
        self.register_buffer(
            "extra_joints_idxs",
            smplx.utils.to_tensor(list(smplx.vertex_ids.vertex_ids["mano"].values()), dtype=torch.long),
        )
        self.register_buffer("joint_map", torch.tensor(mano_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> smplx.utils.MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = smplx.lbs.vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output

    def query(self, hmr_output):
        batch_size = hmr_output["pred_rotmat"].shape[0]
        pred_rotmat = hmr_output["pred_rotmat"].reshape(batch_size, -1, 3, 3)
        pred_shape = hmr_output["pred_shape"].reshape(batch_size, 10)

        mano_output = self(
            global_orient=pred_rotmat[:, [0]], hand_pose=pred_rotmat[:, 1:], betas=pred_shape, pose2rot=False
        )

        return mano_output


def block_print():
    sys.stdout = open(os.devnull, "w")


def enable_print():
    sys.stdout = sys.__stdout__


def run_mano(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    block_print()
    MANO_cfg = {
        "DATA_DIR": "_DATA/data/",
        "MODEL_PATH": "_DATA/data/mano",
        "GENDER": "neutral",
        "NUM_HAND_JOINTS": 15,
        "CREATE_BODY_POSE": False,
    }
    mano_cfg = {k.lower(): v for k, v in MANO_cfg.items()}
    mano = MANO(**mano_cfg)
    if use_cuda:
        mano = mano.cuda()

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    mano_params = {
        "global_orient": root_orient.reshape(B * T, -1),
        "hand_pose": hand_pose.reshape(B * T * NUM_JOINTS, 3),
        "betas": betas.reshape(B * T, -1),
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params["global_orient"] = aa_to_rotmat(mano_params["global_orient"]).view(B * T, 1, 3, 3)
    rotmat_mano_params["hand_pose"] = aa_to_rotmat(mano_params["hand_pose"]).view(B * T, NUM_JOINTS, 3, 3)
    rotmat_mano_params["transl"] = trans.reshape(B * T, 3)

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k, v in rotmat_mano_params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k, v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right = mano.faces
    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ]
    )
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_n = len(faces_right)
    faces_left = faces_right[:, [0, 2, 1]]

    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if is_right is not None:
        # outputs["vertices"][..., 0] = (2*is_right-1)*outputs["vertices"][..., 0]
        # outputs["joints"][..., 0] = (2*is_right-1)*outputs["joints"][..., 0]
        is_right = is_right[:, :, 0].cpu().numpy() > 0
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0)
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0)
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    enable_print()
    return outputs


def run_mano_left(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True, fix_shapedirs=True):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    block_print()
    MANO_cfg = {
        "DATA_DIR": "_DATA/data_left/",
        "MODEL_PATH": "_DATA/data_left/mano_left",
        "GENDER": "neutral",
        "NUM_HAND_JOINTS": 15,
        "CREATE_BODY_POSE": False,
        "is_rhand": False,
    }
    mano_cfg = {k.lower(): v for k, v in MANO_cfg.items()}
    mano = MANO(**mano_cfg)
    if use_cuda:
        mano = mano.cuda()

    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if fix_shapedirs:
        mano.shapedirs[:, 0, :] *= -1

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    mano_params = {
        "global_orient": root_orient.reshape(B * T, -1),
        "hand_pose": hand_pose.reshape(B * T * NUM_JOINTS, 3),
        "betas": betas.reshape(B * T, -1),
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params["global_orient"] = aa_to_rotmat(mano_params["global_orient"]).view(B * T, 1, 3, 3)
    rotmat_mano_params["hand_pose"] = aa_to_rotmat(mano_params["hand_pose"]).view(B * T, NUM_JOINTS, 3, 3)
    rotmat_mano_params["transl"] = trans.reshape(B * T, 3)

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k, v in rotmat_mano_params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k, v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right = mano.faces
    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ]
    )
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_n = len(faces_right)
    faces_left = faces_right[:, [0, 2, 1]]

    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if is_right is not None:
        # outputs["vertices"][..., 0] = (2*is_right-1)*outputs["vertices"][..., 0]
        # outputs["joints"][..., 0] = (2*is_right-1)*outputs["joints"][..., 0]
        is_right = is_right[:, :, 0].cpu().numpy() > 0
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0)
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0)
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    enable_print()
    return outputs


def run_mano_twohands(init_trans, init_rot, init_hand_pose, is_right, init_betas, use_cuda=True, fix_shapedirs=True):
    outputs_left = run_mano_left(
        init_trans[0:1],
        init_rot[0:1],
        init_hand_pose[0:1],
        None,
        init_betas[0:1],
        use_cuda=use_cuda,
        fix_shapedirs=fix_shapedirs,
    )
    outputs_right = run_mano(
        init_trans[1:2], init_rot[1:2], init_hand_pose[1:2], None, init_betas[1:2], use_cuda=use_cuda
    )
    outputs_two = {
        "vertices": torch.cat((outputs_left["vertices"], outputs_right["vertices"]), dim=0),
        "joints": torch.cat((outputs_left["joints"], outputs_right["joints"]), dim=0),
    }
    return outputs_two


def quaternion_mul(q0, q1):
    """
    EXPECTS WXYZ
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    r0, r1 = q0[..., :1], q1[..., :1]
    v0, v1 = q0[..., 1:], q1[..., 1:]
    r = r0 * r1 - (v0 * v1).sum(dim=-1, keepdim=True)
    v = r0 * v1 + r1 * v0 + torch.linalg.cross(v0, v1)
    return torch.cat([r, v], dim=-1)


def quaternion_inverse(q, eps=1e-8):
    """
    EXPECTS WXYZ
    :param q (*, 4)
    """
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    mag = torch.square(q).sum(dim=-1, keepdim=True) + eps
    return conj / mag


def quaternion_slerp(t, q0, q1, eps=1e-8):
    """
    :param t (*, 1)  must be between 0 and 1
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    dims = q0.shape[:-1]
    t = t.view(*dims, 1)

    q0 = F.normalize(q0, p=2, dim=-1)
    q1 = F.normalize(q1, p=2, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # make sure we give the shortest rotation path (< 180d)
    neg = dot < 0
    q1 = torch.where(neg, -q1, q1)
    dot = torch.where(neg, -dot, dot)
    angle = torch.acos(dot)

    # if angle is too small, just do linear interpolation
    collin = torch.abs(dot) > 1 - eps
    fac = 1 / torch.sin(angle)
    w0 = torch.where(collin, 1 - t, torch.sin((1 - t) * angle) * fac)
    w1 = torch.where(collin, t, torch.sin(t * angle) * fac)
    slerp = q0 * w0 + q1 * w1
    return slerp


def angle_axis_to_rotation_matrix(angle_axis):
    """
    :param angle_axis (*, 3)
    return (*, 3, 3)
    """
    quat = angle_axis_to_quaternion(angle_axis)
    return quaternion_to_rotation_matrix(quat)


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    Taken from https://github.com/kornia/kornia, based on
    https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
    https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247
    :param quaternion (N, 4) expects WXYZ order
    returns rotation matrix (N, 3, 3)
    """
    # normalize the input quaternion
    quaternion_norm = F.normalize(quaternion, p=2, dim=-1, eps=1e-12)
    *dims, _ = quaternion_norm.shape

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(*dims, 3, 3)
    return matrix


def angle_axis_to_quaternion(angle_axis):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert angle axis to quaternion in WXYZ order
    :param angle_axis (*, 3)
    :returns quaternion (*, 4) WXYZ order
    """
    theta_sq = torch.sum(angle_axis**2, dim=-1, keepdim=True)  # (*, 1)
    # need to handle the zero rotation case
    valid = theta_sq > 0
    theta = torch.sqrt(theta_sq)
    half_theta = 0.5 * theta
    ones = torch.ones_like(half_theta)
    # fill zero with the limit of sin ax / x -> a
    k = torch.where(valid, torch.sin(half_theta) / theta, 0.5 * ones)
    w = torch.where(valid, torch.cos(half_theta), ones)
    quat = torch.cat([w, k * angle_axis], dim=-1)
    return quat


class HAWOR(pl.LightningModule):

    def __init__(self, cfg: yacs.config.CfgNode):
        """
        Setup HAWOR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=["init_renderer"])

        self.cfg = cfg
        self.crop_size = cfg.MODEL.IMAGE_SIZE
        self.seq_len = 16
        self.pose_num = 16
        self.pose_dim = 6  # rot6d representation
        self.box_info_dim = 3
        self.global_log = get_pylogger(__name__)

        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        try:
            if cfg.MODEL.BACKBONE.get("PRETRAINED_WEIGHTS", None):
                whole_state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location="cpu")["state_dict"]
                backbone_state_dict = {}
                for key in whole_state_dict:
                    if key[:9] == "backbone.":
                        backbone_state_dict[key[9:]] = whole_state_dict[key]
                self.backbone.load_state_dict(backbone_state_dict)
                print(f"Loaded backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}")
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                print("WARNING: init backbone from sratch !!!")
        except Exception:
            print("WARNING: init backbone from sratch !!!")

        # Space-time memory
        if cfg.MODEL.ST_MODULE:
            hdim = cfg.MODEL.ST_HDIM
            nlayer = cfg.MODEL.ST_NLAYER
            self.st_module = temporal_attention(in_dim=1280 + 3, out_dim=1280, hdim=hdim, nlayer=nlayer, residual=True)
            print(f"Using Temporal Attention space-time: {nlayer} layers {hdim} dim.")
        else:
            self.st_module = None

        # Motion memory
        if cfg.MODEL.MOTION_MODULE:
            hdim = cfg.MODEL.MOTION_HDIM
            nlayer = cfg.MODEL.MOTION_NLAYER

            self.motion_module = temporal_attention(
                in_dim=self.pose_num * self.pose_dim + self.box_info_dim,
                out_dim=self.pose_num * self.pose_dim,
                hdim=hdim,
                nlayer=nlayer,
                residual=False,
            )
            print(f"Using Temporal Attention motion layer: {nlayer} layers {hdim} dim.")
        else:
            self.motion_module = None

        # Create MANO head
        # self.mano_head = build_mano_head(cfg)
        self.mano_head = MANOTransformerDecoderHead(cfg)

        # default open torch compile
        if cfg.MODEL.BACKBONE.get("TORCH_COMPILE", 0):
            self.global_log.info("Model will use torch.compile")
            self.backbone = torch.compile(self.backbone)
            self.mano_head = torch.compile(self.mano_head)

        # Define loss functions
        # self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        # self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        # self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer("initialized", torch.tensor(False))

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        if cfg.MODEL.get("LOAD_WEIGHTS", None):
            whole_state_dict = torch.load(cfg.MODEL.LOAD_WEIGHTS, map_location="cpu")["state_dict"]
            self.load_state_dict(whole_state_dict, strict=True)
            print(f"load {cfg.MODEL.LOAD_WEIGHTS}")

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        if self.st_module is not None:
            all_params += list(self.st_module.parameters())
        if self.motion_module is not None:
            all_params += list(self.motion_module.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{"params": filter(lambda p: p.requires_grad, self.get_parameters()), "lr": self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(
            params=param_groups,
            # lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        image = batch["img"].flatten(0, 1)
        center = batch["center"].flatten(0, 1)
        scale = batch["scale"].flatten(0, 1)
        img_focal = batch["img_focal"].flatten(0, 1)
        img_center = batch["img_center"].flatten(0, 1)

        # estimate focal length, and bbox
        bbox_info = self.bbox_est(center, scale, img_focal, img_center)

        # backbone
        feature = self.backbone(image[:, :, :, 32:-32])
        feature = feature.float()

        # space-time module
        if self.st_module is not None:
            bb = einops.repeat(bbox_info, "b c -> b c h w", h=16, w=12)
            feature = torch.cat([feature, bb], dim=1)

            feature = einops.rearrange(feature, "(b t) c h w -> (b h w) t c", t=16)
            feature = self.st_module(feature)
            feature = einops.rearrange(feature, "(b h w) t c -> (b t) c h w", h=16, w=12)

        # smpl_head: transformer + smpl
        # pred_mano_params, pred_cam, pred_mano_params_list = self.mano_head(feature)
        # pred_shape = pred_mano_params_list['pred_shape']
        # pred_pose = pred_mano_params_list['pred_pose']
        pred_pose, pred_shape, pred_cam = self.mano_head(feature)
        pred_rotmat_0 = rot6d_to_rotmat(pred_pose).reshape(-1, self.pose_num, 3, 3)

        # smpl motion module
        if self.motion_module is not None:
            bb = einops.rearrange(bbox_info, "(b t) c -> b t c", t=16)
            pred_pose = einops.rearrange(pred_pose, "(b t) c -> b t c", t=16)
            pred_pose = torch.cat([pred_pose, bb], dim=2)

            pred_pose = self.motion_module(pred_pose)
            pred_pose = einops.rearrange(pred_pose, "b t c -> (b t) c")

        out = {}
        if "do_flip" in batch:
            pred_cam[..., 1] *= -1
            center[..., 0] = img_center[..., 0] * 2 - center[..., 0] - 1
        out["pred_cam"] = pred_cam
        out["pred_pose"] = pred_pose
        out["pred_shape"] = pred_shape
        out["pred_rotmat"] = rot6d_to_rotmat(out["pred_pose"]).reshape(-1, self.pose_num, 3, 3)
        out["pred_rotmat_0"] = pred_rotmat_0

        s_out = self.mano.query(out)
        j3d = s_out.joints
        j2d = self.project(j3d, out["pred_cam"], center, scale, img_focal, img_center)
        j2d = j2d / self.crop_size - 0.5  # norm to [-0.5, 0.5]

        trans_full = self.get_trans(out["pred_cam"], center, scale, img_focal, img_center)
        out["trans_full"] = trans_full

        output = {
            "pred_mano_params": {
                "global_orient": out["pred_rotmat"][:, :1].clone(),
                "hand_pose": out["pred_rotmat"][:, 1:].clone(),
                "betas": out["pred_shape"].clone(),
            },
            "pred_keypoints_3d": j3d.clone(),
            "pred_keypoints_2d": j2d.clone(),
            "out": out,
        }
        # print(output)
        # output['gt_project_j2d'] = self.project(batch['gt_j3d_wo_trans'].clone().flatten(0,1), out['pred_cam'], center, scale, img_focal, img_center)
        # output['gt_project_j2d'] = output['gt_project_j2d'] / self.crop_size - 0.5

        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_mano_params = output["pred_mano_params"]
        pred_keypoints_2d = output["pred_keypoints_2d"]
        pred_keypoints_3d = output["pred_keypoints_3d"]

        batch_size = pred_mano_params["hand_pose"].shape[0]

        # Get annotations
        gt_keypoints_2d = batch["gt_cam_j2d"].flatten(0, 1)
        gt_keypoints_2d = torch.cat(
            [gt_keypoints_2d, torch.ones(*gt_keypoints_2d.shape[:-1], 1, device=gt_keypoints_2d.device)], dim=-1
        )
        gt_keypoints_3d = batch["gt_j3d_wo_trans"].flatten(0, 1)
        gt_keypoints_3d = torch.cat(
            [gt_keypoints_3d, torch.ones(*gt_keypoints_3d.shape[:-1], 1, device=gt_keypoints_3d.device)], dim=-1
        )
        pose_gt = batch["gt_cam_full_pose"].flatten(0, 1).reshape(-1, 16, 3)
        rotmat_gt = angle_axis_to_rotation_matrix(pose_gt)
        gt_mano_params = {
            "global_orient": rotmat_gt[:, :1],
            "hand_pose": rotmat_gt[:, 1:],
            "betas": batch["gt_cam_betas"],
        }

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # to avoid nan
        loss_keypoints_2d = torch.nan_to_num(loss_keypoints_2d)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1))

        loss = (
            self.cfg.LOSS_WEIGHTS["KEYPOINTS_3D"] * loss_keypoints_3d
            + self.cfg.LOSS_WEIGHTS["KEYPOINTS_2D"] * loss_keypoints_2d
            + sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])
        )

        losses = dict(
            loss=loss.detach(),
            loss_keypoints_2d=loss_keypoints_2d.detach() * self.cfg.LOSS_WEIGHTS["KEYPOINTS_2D"],
            loss_keypoints_3d=loss_keypoints_3d.detach() * self.cfg.LOSS_WEIGHTS["KEYPOINTS_3D"],
        )

        for k, v in loss_mano_params.items():
            losses["loss_" + k] = v.detach() * self.cfg.LOSS_WEIGHTS[k.upper()]

        output["losses"] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(
        self,
        batch: Dict,
        output: Dict,
        step_count: int,
        train: bool = True,
        write_to_summary_writer: bool = True,
        render_log: bool = True,
    ) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = "train" if train else "val"
        batch_size = output["pred_keypoints_2d"].shape[0]
        images = batch["img"].flatten(0, 1)
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        losses = output["losses"]
        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode + "/" + loss_name, val.detach().item(), step_count)

        if render_log:
            gt_keypoints_2d = batch["gt_cam_j2d"].flatten(0, 1).clone()
            pred_keypoints_2d = output["pred_keypoints_2d"].clone().detach().reshape(batch_size, -1, 2)
            gt_project_j2d = pred_keypoints_2d
            # gt_project_j2d = output['gt_project_j2d'].clone().detach().reshape(batch_size, -1, 2)

            num_images = 4
            skip = 16

            predictions = self.visualize_tensorboard(
                images[: num_images * skip : skip].cpu().numpy(),
                pred_keypoints_2d[: num_images * skip : skip].cpu().numpy(),
                gt_project_j2d[: num_images * skip : skip].cpu().numpy(),
                gt_keypoints_2d[: num_images * skip : skip].cpu().numpy(),
            )
            summary_writer.add_image("%s/predictions" % mode, predictions, step_count)

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch["img"]
        optimizer = self.optimizers(use_pl_optimizer=True)

        batch_size = batch["img"].shape[0]
        output = self.forward_step(batch, train=True)
        # pred_mano_params = output['pred_mano_params']
        loss = self.compute_loss(batch, output, train=True)

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get("GRAD_CLIP_VAL", 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True
            )
            self.log(
                "train/grad_norm", gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size
            )
        optimizer.step()

        # if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
        if self.global_step > 0 and self.global_step % 100 == 0:
            self.tensorboard_logging(
                batch, output, self.global_step, train=True, render_log=self.cfg.TRAIN.get("RENDER_LOG", True)
            )

        self.log(
            "train/loss",
            output["losses"]["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            batch_size=batch_size,
        )

        return output

    def inference(self, imgfiles, boxes, img_focal, img_center, device="cuda", do_flip=False):
        db = TrackDatasetEval(
            imgfiles, boxes, img_focal=img_focal, img_center=img_center, normalization=True, dilate=1.2, do_flip=do_flip
        )

        # Results
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []

        # To-do: efficient implementation with batch
        items = []
        for i in tqdm(range(len(db))):
            item = db[i]
            items.append(item)

            # padding to 16
            if i == len(db) - 1 and len(db) % 16 != 0:
                pad = 16 - len(db) % 16
                for _ in range(pad):
                    items.append(item)

            if len(items) < 16:
                continue
            elif len(items) == 16:
                batch = torch.utils.data.default_collate(items)
                items = []
            else:
                raise NotImplementedError

            with torch.no_grad():
                batch = {k: v.to(device).unsqueeze(0) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # for image_i in range(16):
                #     hawor_input_cv2 = vis_tensor_cv2(batch['img'][:, image_i])
                #     cv2.imwrite(f'debug_vis_model.png', hawor_input_cv2)
                #     print("vis")
                output = self.forward(batch)
                out = output["out"]

            if i == len(db) - 1 and len(db) % 16 != 0:
                out = {k: v[: len(db) % 16] for k, v in out.items()}
            else:
                out = {k: v for k, v in out.items()}

            pred_cam.append(out["pred_cam"].cpu())
            pred_pose.append(out["pred_pose"].cpu())
            pred_shape.append(out["pred_shape"].cpu())
            pred_rotmat.append(out["pred_rotmat"].cpu())
            pred_trans.append(out["trans_full"].cpu())

        results = {
            "pred_cam": torch.cat(pred_cam),
            "pred_pose": torch.cat(pred_pose),
            "pred_shape": torch.cat(pred_shape),
            "pred_rotmat": torch.cat(pred_rotmat),
            "pred_trans": torch.cat(pred_trans),
            "img_focal": img_focal,
            "img_center": img_center,
        }

        return results

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output["loss"] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    def visualize_tensorboard(self, images, pred_keypoints, gt_project_j2d, gt_keypoints):
        pred_keypoints = 256 * (pred_keypoints + 0.5)
        gt_keypoints = 256 * (gt_keypoints + 0.5)
        gt_project_j2d = 256 * (gt_project_j2d + 0.5)
        pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
        gt_keypoints = np.concatenate((gt_keypoints, np.ones_like(gt_keypoints)[:, :, [0]]), axis=-1)
        gt_project_j2d = np.concatenate((gt_project_j2d, np.ones_like(gt_project_j2d)[:, :, [0]]), axis=-1)
        images_np = np.transpose(images, (0, 2, 3, 1))
        rend_imgs = []
        for i in range(images_np.shape[0]):
            pred_keypoints_img = render_openpose(255 * images_np[i].copy(), pred_keypoints[i]) / 255
            gt_project_j2d_img = render_openpose(255 * images_np[i].copy(), gt_project_j2d[i]) / 255
            gt_keypoints_img = render_openpose(255 * images_np[i].copy(), gt_keypoints[i]) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2, 0, 1))
            rend_imgs.append(torch.from_numpy(gt_project_j2d_img).permute(2, 0, 1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2, 0, 1))
        rend_imgs = torchvision.utils.make_grid(rend_imgs, nrow=4, padding=2)
        return rend_imgs

    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):

        trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

        # Projection in full frame image coordinate
        points = points + trans_full
        points2d_full = perspective_projection(
            points, rotation=None, translation=None, focal_length=img_focal, camera_center=img_center
        )

        # Adjust projected points to crop image coordinate
        # (s.t. 1. we can calculate loss in crop image easily
        #       2. we can query its pixel in the crop
        #  )
        b = scale * 200
        points2d = points2d_full - (center - b[:, None] / 2)[:, None, :]
        points2d = points2d * (self.crop_size / b)[:, None, None]

        if return_full:
            return points2d_full, points2d
        else:
            return points2d

    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        b = scale * 200
        cx, cy = center[:, 0], center[:, 1]  # center of crop
        s, tx, ty = pred_cam.unbind(-1)

        img_cx, img_cy = img_center[:, 0], img_center[:, 1]  # center of original image

        bs = b * s
        tx_full = tx + 2 * (cx - img_cx) / bs
        ty_full = ty + 2 * (cy - img_cy) / bs
        tz_full = 2 * img_focal / bs

        trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
        trans_full = trans_full.unsqueeze(1)

        return trans_full

    def bbox_est(self, center, scale, img_focal, img_center):
        # Original image center
        img_cx, img_cy = img_center[:, 0], img_center[:, 1]

        # Implement CLIFF (Li et al.) bbox feature
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)

        return bbox_info
