import os

from agentscope.service import ServiceExecStatus, ServiceResponse
from utils import execute_config, init_config

HF_MODEL_DIR = os.getenv("HF_MODEL_DIR", "")
diffusion_model_path = os.path.join(HF_MODEL_DIR, "CompVis/stable-diffusion-v1-4")
img2seq_model_path = os.path.join(HF_MODEL_DIR, "Salesforce/blip2-opt-2.7b")
video_blip_model_path = os.path.join(HF_MODEL_DIR, "kpyu/video-blip-opt-2.7b-ego4d")


def execute_image_caption_mapper(dataset_path: str) -> ServiceResponse:
    """
    Produce captions for each image in the dataset.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_config(
            dataset_path,
            "image_captioning_mapper",
            hf_img2seq=img2seq_model_path,
            keep_original_sample=False,
            mem_required="16GB",
        )
        result_path = execute_config(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_image_diffusion_mapper(dataset_path: str) -> ServiceResponse:
    """
    Produce images according to each text in the dataset.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_config(
            dataset_path,
            "image_diffusion_mapper",
            hf_diffusion=diffusion_model_path,
            keep_original_sample=False,
            caption_key="text",
            mem_required="8GB",
        )
        result_path = execute_config(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_image_face_blur_mapper(dataset_path: str) -> ServiceResponse:
    """
    Detect and blur face areas for each images in the dataset.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_config(dataset_path, "image_face_blur_mapper")
        result_path = execute_config(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_video_caption_mapper(dataset_path: str) -> ServiceResponse:
    """
    Produce captions for each video in the dataset.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_config(
            dataset_path,
            "video_captioning_from_video_mapper",
            keep_original_sample=False,
            hf_video_blip=video_blip_model_path,
            mem_required="20GB",
        )
        result_path = execute_config(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_video_face_blur_mapper(dataset_path: str) -> ServiceResponse:
    """
    Detect and blur face areas for each video in the dataset.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_config(dataset_path, "video_face_blur_mapper")
        result_path = execute_config(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)
