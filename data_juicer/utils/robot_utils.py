"""Robot episode I/O, geometry and shared perception helpers."""

from __future__ import annotations

import ast
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from data_juicer.utils.lazy_loader import LazyLoader


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def episode_root(sample, output_dir):
    if sample["source"] != "droid":
        raise ValueError("Expected DROID episode input")
    identifier = sample["episode_id"]
    if not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("episode_id must be a safe filename component")
    root = Path(output_dir).resolve() / sample["source"] / identifier
    root.mkdir(parents=True, exist_ok=True)
    sample["episode_dir"] = str(root)
    return root


def vlm_settings(model, endpoint):
    endpoint = (endpoint or os.environ.get("VLM_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
    model = model or os.environ.get("VLM_MODEL")
    if not model:
        raise ValueError("Set VLM_MODEL or the model parameter")
    return (model, endpoint)


def intersection(a, b):
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[2] > box[0] and box[3] > box[1] else None


def overlap_ratio(a, b):
    common = intersection(a, b)
    if common is None:
        return 0.0
    area = (common[2] - common[0]) * (common[3] - common[1])
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return area / max(1.0, min(area_a, area_b))


def normalized_center(box, width, height):
    return [round((box[0] + box[2]) * 500 / width), round((box[1] + box[3]) * 500 / height)]


cv2 = LazyLoader("cv2")
np = LazyLoader("numpy")
pc = LazyLoader("pyarrow.compute", "pyarrow")
pq = LazyLoader("pyarrow.parquet", "pyarrow")


def read_instruction(path):
    row = pq.read_table(path).to_pylist()[0]
    for key in (
        "other_information.language_instruction_2",
        "other_information.language_instruction_3",
        "other_information.language_instruction_1",
        "annotation.instruction_add",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return (value.strip(), key)
    raise ValueError(f"No task instruction in {path}")


def extract_gripper_anchor(path, episode_index, output):
    table = pq.read_table(path, columns=["frame_index", "episode_index", "observation.state.gripper_position"])
    table = table.filter(pc.equal(table["episode_index"], episode_index))
    rows = sorted(table.to_pylist(), key=lambda x: x["frame_index"])
    if len(rows) < 4:
        raise ValueError(f"Expected at least four state rows for episode {episode_index}")
    values = np.asarray([r["observation.state.gripper_position"] for r in rows], dtype=float)
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)
    (lo, hi) = (float(np.nanmin(values)), float(np.nanmax(values)))
    normalized = (values - lo) / (hi - lo + 1e-08)
    width = min(9, len(rows))
    smoothed = np.convolve(normalized, np.ones(width) / width, mode="same")
    anchor = int(rows[int(np.nanargmin(smoothed))]["frame_index"])
    result = {
        "operator": "gripper_signal_phase_anchor",
        "source": str(path),
        "state_frame_count": len(rows),
        "state_start": int(rows[0]["frame_index"]),
        "state_end": int(rows[-1]["frame_index"]),
        "minimum_smoothed_gripper_frame": anchor,
        "interpretation": "gripper-signal sampling anchor",
    }
    write_json(output, result)
    return result


def decode_camera_views(raw_dir, output, count=None):
    videos = sorted(raw_dir.glob("*.mp4"))
    if len(videos) < 2:
        raise ValueError("Cross-view demo needs at least two raw camera videos")
    sampled = {}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        if frame_count < 4 or (count is not None and frame_count < count):
            raise ValueError(f"Too few frames in {video}")
        indices = (
            list(range(frame_count))
            if count is None
            else np.linspace(0, frame_count - 1, count).round().astype(int).tolist()
        )
        frames = []
        view_dir = output / video.stem
        view_dir.mkdir(parents=True, exist_ok=True)
        for slot, index in enumerate(indices):
            if count is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            (ok, frame) = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read {video} frame {index}")
            image_path = view_dir / f"frame_{slot:02d}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write {image_path}")
            frames.append(frame)
        cap.release()
        (height, width) = frames[0].shape[:2]
        clip_path = view_dir / "sampled.mp4"
        writer = cv2.VideoWriter(
            str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps if count is None else 2.0, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open {clip_path} for writing")
        for frame in frames:
            writer.write(frame)
        writer.release()
        sampled[video.stem] = {
            "source_video": str(video),
            "source_frame_count": frame_count,
            "source_fps": fps,
            "frame_indices": indices,
            "sampled_video": str(clip_path),
            "images": [str(view_dir / f"frame_{i:02d}.jpg") for i in range(len(indices))],
            "image_size": {"width": width, "height": height},
        }
    return sampled


def load_common_modules():
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    return (np, torch, imageio, Image, ImageDraw)


def auto_device(torch, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype(torch, name: str):
    return {"auto": None, "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def snapshot_download_model(model_id: str, cache_dir: Path | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'modelscope'. Install it with: python -m pip install modelscope"
        ) from exc
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if model_id == "facebook/sam3":
        kwargs["ignore_file_pattern"] = ["sam3.pt"]
    return Path(snapshot_download(model_id, **kwargs))


def move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for (key, value) in inputs.items()}


def tensor_to_numpy(np, torch, value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        if getattr(value, "dtype", None) == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return value


def image_part(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def assistant_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join((str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content))
    return str(content)


def assistant_reasoning(message: dict[str, Any]) -> str:
    return str(message.get("reasoning_content") or message.get("reasoning") or "")


def parse_vlm_json(raw_content: str) -> dict[str, Any]:
    text = raw_content.strip()
    for marker in ("```json", "```", "JSON:", "Response:"):
        text = text.replace(marker, "")
    text = text.strip()
    try:
        result = json.loads(text, strict=False)
    except Exception:
        try:
            result = ast.literal_eval(text)
        except Exception:
            match = re.search("\\{.*\\}", text, flags=re.S)
            if not match:
                return {}
            try:
                result = json.loads(match.group(0), strict=False)
            except Exception:
                return {}
    if isinstance(result, list) and result:
        result = result[0]
    return result if isinstance(result, dict) else {}


def load_rows(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()


def parse_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"", "[]", "-1", "None", "nan"} else text


def read_video_frame(imageio, video_path: Path, frame_index: int):
    reader = imageio.get_reader(str(video_path))
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


def video_frame_count(imageio, video_path: Path) -> int | None:
    try:
        reader = imageio.get_reader(str(video_path))
        try:
            count = reader.count_frames()
            return int(count) if count and count > 0 else None
        finally:
            reader.close()
    except Exception:
        return None


def save_rgb_image(Image, frame_rgb, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).convert("RGB").save(output_path)


def save_context_frames(args, imageio, Image) -> tuple[Path, Path, int, int, int, int]:
    first_frame = read_video_frame(imageio, args.video, args.first_video_frame_index)
    (frame_height, frame_width) = first_frame.shape[:2]
    count = video_frame_count(imageio, args.video)
    last_index = (
        args.last_video_frame_index
        if args.last_video_frame_index >= 0
        else count - 1 if count else args.first_video_frame_index
    )
    last_frame = read_video_frame(imageio, args.video, last_index)
    first_path = args.output_dir / "first_frame.jpg"
    last_path = args.output_dir / "last_frame.jpg"
    save_rgb_image(Image, first_frame, first_path)
    save_rgb_image(Image, last_frame, last_path)
    return (first_path, last_path, frame_width, frame_height, args.first_video_frame_index, int(last_index))


def normalize_point(value: Any, width: int, height: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        nums = re.findall("-?\\d+(?:\\.\\d+)?", value)
        value = [float(num) for num in nums[:2]]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    (x, y) = (float(value[0]), float(value[1]))
    return [max(0.0, min(x, float(width))), max(0.0, min(y, float(height)))]


def mask_to_bbox(np, mask: Any) -> list[float] | None:
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    (ys, xs) = np.where(mask_bool)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def save_mask_png(Image, np, mask: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (np.asarray(mask).squeeze() > 0).astype("uint8") * 255
    Image.fromarray(mask_u8).save(output_path)


def perception_settings(model, endpoint, use_cuda, rank):
    torch = LazyLoader("torch")
    model, endpoint = vlm_settings(model, endpoint)
    device = f"cuda:{rank or 0}" if use_cuda and torch.cuda.is_available() else "cpu"
    return model, endpoint, device
