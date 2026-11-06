"""Object and gripper tracking, VLM requests and target grounding."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from data_juicer.utils import robot_utils as common
from data_juicer.utils import sam3_utils as sam3
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.robot_utils import vlm_settings

SYSTEM_PROMPT = "You are a precise robotic manipulation perception assistant. Return only valid JSON."


@dataclass
class ChatGenerationConfig:
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int | None = 20
    min_p: float | None = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


class VLLMChatClient:
    def __init__(self, model: str, base_url: str, api_key: str, generation: ChatGenerationConfig, timeout: int):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.generation = generation
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "presence_penalty": self.generation.presence_penalty,
        }
        if self.generation.top_k is not None:
            payload["top_k"] = self.generation.top_k
        if self.generation.min_p is not None:
            payload["min_p"] = self.generation.min_p
        if self.generation.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.generation.repetition_penalty
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        return response_payload["choices"][0]["message"]


class ChatJSONClient:
    def __init__(self, chat_client: VLLMChatClient):
        self.chat_client = chat_client

    def chat_json(self, prompt: str, image_paths: list[Path]) -> tuple[dict[str, Any], str, str]:
        content = [common.text_part(prompt)]
        if len(image_paths) > 1:
            from PIL import Image, ImageDraw

            images = [Image.open(path).convert("RGB") for path in image_paths]
            max_w = max((image.width for image in images))
            max_h = max((image.height for image in images))
            label_h = 32
            sheet = Image.new("RGB", (max_w * len(images), max_h + label_h), "white")
            draw = ImageDraw.Draw(sheet)
            for idx, image in enumerate(images):
                x = idx * max_w
                sheet.paste(image, (x, label_h))
                draw.text((x + 8, 8), f"image {idx + 1}", fill="black")
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="vlm_contact_", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            sheet.save(tmp_path, quality=92)
            try:
                content.append(common.image_part(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)
        elif image_paths:
            content.append(common.image_part(image_paths[0]))
        message = self.chat_client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        )
        raw_content = common.assistant_content(message)
        reasoning = common.assistant_reasoning(message)
        return (common.parse_vlm_json(raw_content), raw_content, reasoning)


def build_vllm_json_client(args) -> ChatJSONClient:
    generation = ChatGenerationConfig(
        max_tokens=args.vlm_max_new_tokens,
        temperature=args.vlm_temperature,
        top_p=args.vlm_top_p,
        top_k=args.vlm_top_k,
        min_p=args.vlm_min_p,
        presence_penalty=args.vlm_presence_penalty,
        repetition_penalty=args.vlm_repetition_penalty,
    )
    return ChatJSONClient(
        VLLMChatClient(
            model=args.vlm_model,
            base_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            generation=generation,
            timeout=args.vlm_timeout,
        )
    )


Image = LazyLoader("PIL.Image", "Pillow")
ImageDraw = LazyLoader("PIL.ImageDraw", "Pillow")


def contact_sheet(paths, output):
    tiles = []
    for i, path in enumerate(paths, 1):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((768, 432))
            tile = Image.new("RGB", (768, 462), "white")
            tile.paste(image, ((768 - image.width) // 2, 30))
            ImageDraw.Draw(tile).text((12, 8), f"Image {i}", fill="black")
            tiles.append(tile)
    columns = min(2, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 768, rows * 462), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i % columns * 768, i // columns * 462))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def ground_final_target(root, sampled, target, model=None, endpoint=None):
    """Locate the final target in a selected camera view."""
    root = Path(root)
    views = sorted(sampled)
    images = [sampled[v]["images"][-1] for v in views]
    paths = [Path(p) if Path(p).is_absolute() else root / p for p in images]
    prompt = f"Locate the {target} in the final camera views. Select a panel where the object itself is visible. Do not select the gripper, cables, or an intended placement location without the object. Return JSON with image_index (1-based), point_2d (x,y in [0,1000] relative to that panel's original image), and evidence describing the visible object. If the object is not visible in any view, use image_index 0 and point_2d null."
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    (model, endpoint) = vlm_settings(model, endpoint)
    request_hash = hashlib.sha256(json.dumps([prompt, hashes, model]).encode()).hexdigest()
    output = root / "ir/final_target_grounding.json"
    if output.exists():
        trace = json.loads(output.read_text())
        if trace.get("request_sha256") != request_hash:
            trace = None
    else:
        trace = None
    if trace is None:
        sheet = output.with_suffix(".jpg")
        contact_sheet(paths, sheet)
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(sheet.read_bytes()).decode()},
            },
        ]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get("VLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VLM_API_KEY"]
        request = urllib.request.Request(
            endpoint + "/chat/completions", data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read())["choices"][0]["message"]["content"]
        selection = json.loads(raw)
        selected = selection.get("image_index")
        if not isinstance(selected, int) or not 1 <= selected <= len(paths):
            raise ValueError("Final target is not visible in supplied views")
        point_prompt = f"Point to the visible {target} in this image. Return JSON with point_2d (x,y normalized to [0,1000] over this entire image) and evidence. Locate the object itself, not the gripper, cables, or an empty goal region."
        payload["messages"][0]["content"] = [
            {"type": "text", "text": point_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(paths[selected - 1].read_bytes()).decode()
                },
            },
        ]
        request = urllib.request.Request(
            endpoint + "/chat/completions", data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            point_raw = json.loads(response.read())["choices"][0]["message"]["content"]
        point_annotation = json.loads(point_raw)
        trace = {
            "request_sha256": request_hash,
            "prompt": point_prompt,
            "selection_prompt": prompt,
            "selection_response": raw,
            "image_sha256": hashes,
            "images": [str(p.relative_to(root)) for p in paths],
            "selection_input": str(sheet.relative_to(root)),
            "selection_input_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
            "selected_image_index": selected,
            "response": point_raw,
            "annotation": point_annotation,
        }
        output.write_text(json.dumps(trace, indent=2) + "\n")
    annotation = trace["annotation"]
    (index, point) = (trace["selected_image_index"], annotation.get("point_2d"))
    if not isinstance(index, int) or not 1 <= index <= len(views):
        raise ValueError("Final target is not visible in supplied views")
    if (
        not isinstance(point, list)
        or len(point) != 2
        or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not 0 <= v <= 1000) for v in point))
    ):
        raise ValueError("Invalid final target coordinates")
    view = views[index - 1]
    return {
        "view": view,
        "image": images[index - 1],
        "sample_index": len(sampled[view]["images"]) - 1,
        "point": [round(v) for v in point],
        "model_trace": trace,
    }


DEFAULT_TASK_INSTRUCTION_FALLBACK_COLUMNS = [
    "other_information.language_instruction_2",
    "other_information.language_instruction_3",
    "other_information.language_instruction_1",
    "annotation.instruction_add",
]
OBJECT_SAM3_MODEL = os.environ.get("SAM3_MODEL", "facebook/sam3")
TARGET_OBJECT_PROMPT = '\nGiven the instruction and the first frame of the video, identify the target object that the robot should directly manipulate.\n\nInstruction: {instruction}\n\nReturn exactly this JSON schema:\n{{\n  "target_object": "target object name extracted from the instruction",\n  "referring_expression": "specific phrase that can be used to point to the target object"\n}}\n\nThe referring_expression must describe the same target object but may include useful visual or spatial attributes from the first frame.\n'.strip()
POINTING_PROMPT = "\nPoint to the center of {target_object}. Report the object coordinate in JSON format.\n".strip()


class ObjectTrackingContext:
    def __init__(self, args: SimpleNamespace):
        (self.np, self.torch, self.imageio, self.Image, self.ImageDraw) = common.load_common_modules()
        self.device = common.auto_device(self.torch, args.device)
        if self.device.startswith("cuda"):
            self.torch.backends.cuda.matmul.allow_tf32 = True
            self.torch.backends.cudnn.allow_tf32 = True
        args.vlm_model = resolve_vllm_model(args.vllm_base_url, args.vllm_api_key, args.vlm_model, args.vlm_timeout)
        self.vlm_client = build_vllm_json_client(args)
        self.sam3_model = None
        self.sam3_processor = None
        self.sam3_model_path = None
        self.sam3_video_model = None
        self.sam3_video_processor = None
        self.sam3_video_metadata = None
        started = time.perf_counter()
        (self.sam3_model, self.sam3_processor, self.sam3_model_path) = sam3.load_sam3(
            args.sam3_model, args.sam3_cache_dir, self.torch, self.device, args.sam3_torch_dtype
        )
        print(f"[timer] sam3_load: {time.perf_counter() - started:.3f}s", flush=True)
        print(f"Using SAM3 model: {self.sam3_model_path}", flush=True)

    def load_video_tracker(self, args):
        if self.sam3_video_model is None:
            (self.sam3_video_model, self.sam3_video_processor, self.sam3_video_metadata) = sam3.load_video_tracker(
                args, self.torch, self.device
            )
        return (self.sam3_video_model, self.sam3_video_processor)


def resolve_vllm_model(base_url: str, api_key: str, model: str, timeout: int) -> str:
    if model != "auto":
        return model
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            resolved = str(item["id"])
            print(f"[vllm] using served model from /models: {resolved}", flush=True)
            return resolved
    raise RuntimeError(f"No served model found at {base_url.rstrip('/')}/models")


def read_first_frame(context: ObjectTrackingContext, video_path: Path, frame_index: int):
    frame = common.read_video_frame(context.imageio, video_path, frame_index)
    return context.Image.fromarray(frame).convert("RGB")


def save_first_frame(context: ObjectTrackingContext, image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def normalize_point_1000(value: Any) -> list[float] | None:
    point = common.normalize_point(value, 1000, 1000)
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def point_1000_to_pixel(point: list[float], width: int, height: int) -> list[float]:
    return [float(point[0]) / 1000.0 * float(width), float(point[1]) / 1000.0 * float(height)]


def lowercase_first_char(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def run_vlm_target_object(
    args: SimpleNamespace, context: ObjectTrackingContext, instruction: str, first_frame_path: Path
) -> dict[str, Any]:
    if context.vlm_client is None:
        raise RuntimeError("VLM client was not initialized")
    prompt = TARGET_OBJECT_PROMPT.format(instruction=lowercase_first_char(instruction))
    (parsed, raw_content, reasoning) = context.vlm_client.chat_json(prompt, [first_frame_path])
    target_object = str(parsed.get("target_object") or parsed.get("object") or parsed.get("label") or "").strip()
    if not target_object:
        raise RuntimeError(f"VLM did not return target_object. Raw response: {raw_content}")
    referring_expression = str(parsed.get("referring_expression") or parsed.get("referring") or target_object).strip()
    if not referring_expression:
        referring_expression = target_object
    return {
        "backend": "openai_compatible",
        "model": args.vlm_model,
        "target_object": target_object,
        "referring_expression": referring_expression,
        "prompt": prompt,
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
    }


def run_vlm_pointing(
    args: SimpleNamespace,
    context: ObjectTrackingContext,
    pointing_target: str,
    first_frame_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    if context.vlm_client is None:
        raise RuntimeError("VLM client was not initialized")
    prompt = POINTING_PROMPT.format(target_object=pointing_target, width=width, height=height)
    (parsed, raw_content, reasoning) = context.vlm_client.chat_json(prompt, [first_frame_path])
    point_1000 = normalize_point_1000(parsed.get("point_2d") or parsed.get("point_xy") or parsed.get("point"))
    if point_1000 is None:
        raise RuntimeError(f"VLM did not return a valid point_xy. Raw response: {raw_content}")
    point = point_1000_to_pixel(point_1000, width, height)
    return {
        "backend": "openai_compatible",
        "model": args.vlm_model,
        "pointing_target": pointing_target,
        "point_1000": point_1000,
        "point_xy": point,
        "prompt": prompt,
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
    }


def object_bbox_center(bbox: list[float] | None) -> list[float] | None:
    if bbox is None:
        return None
    return [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]


def distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def run_sam3_candidates(
    args: SimpleNamespace, context: ObjectTrackingContext, image, label: str, output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    if context.sam3_model is None or context.sam3_processor is None:
        raise RuntimeError("SAM3 model was not initialized")
    started = time.perf_counter()
    (processed, metadata) = sam3.run_sam3_text(
        context.np,
        context.torch,
        context.sam3_model,
        context.sam3_processor,
        image,
        label,
        context.device,
        args.score_threshold,
        args.mask_threshold,
    )
    sam3_seconds = time.perf_counter() - started
    candidates = sam3.processed_to_candidates(context.np, context.torch, processed)
    detections = sam3.save_candidates(context.Image, context.np, candidates, output_dir, args.max_candidates)
    return (detections, {"prompt_metadata": metadata, "num_candidates_total": len(candidates)}, sam3_seconds)


def score_value(detection: dict[str, Any]) -> float:
    score = detection.get("score")
    if score is None:
        return float("-inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("-inf")


def choose_best_candidate(
    detections: list[dict[str, Any]], point_xy: list[float], distance_topk: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = []
    for detection in detections:
        center = object_bbox_center(detection.get("bbox_from_mask") or detection.get("bbox_from_model"))
        dist = distance(center, point_xy)
        detection["center_xy"] = center
        detection["distance_to_vlm_point_px"] = dist
        if dist is None:
            continue
        candidates.append(detection)
    candidates = sorted(candidates, key=lambda item: float(item["distance_to_vlm_point_px"]))
    topk = candidates[: max(1, int(distance_topk))]
    best = max(topk, key=score_value) if topk else None
    return (best, topk)


def get_task_instruction_with_fallback(parquet_path: Path, columns: list[str]) -> tuple[str, str]:
    rows = common.load_rows(parquet_path)
    if not rows:
        raise RuntimeError(f"Empty parquet: {parquet_path}")
    row = rows[0]
    checked = []
    for column in columns:
        checked.append(column)
        text = common.parse_text(row.get(column))
        if text:
            return (text, column)
    raise KeyError(f"Task instruction columns are missing or empty: {checked}")


def run_target_tracking(
    args, context: ObjectTrackingContext, payload: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    best = (payload.get("selection") or {}).get("best_candidate") or {}
    mask_path = best.get("mask_path")
    if not mask_path:
        raise RuntimeError("Selected target candidate has no mask_path")
    mask = context.np.asarray(context.Image.open(mask_path).convert("L")) > 0
    (model, processor) = context.load_video_tracker(args)
    (frames, metadata) = sam3.track_video(
        context.np,
        context.torch,
        model,
        processor,
        Path(payload["video_path"]),
        best.get("bbox_from_mask") or best.get("bbox_from_model"),
        mask,
        payload["vlm_pointing"]["point_xy"],
        payload["image_size"][0],
        payload["image_size"][1],
        0,
        context.device,
        args.sam3_video_torch_dtype,
        True,
        "mask",
        payload["first_video_frame_index"],
        False,
    )
    for frame in frames:
        frame.pop("mask", None)
    tracking = {
        "model": context.sam3_video_metadata,
        "num_frames": len(frames),
        "frames": frames,
        "tracking_metadata": metadata,
    }
    path = output_dir / "video_object_tracking.json"
    path.write_text(json.dumps(common.json_ready(tracking), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"num_frames": len(frames), "output_json": str(path), "metadata": metadata}


def track_episode_object(args: SimpleNamespace, context: ObjectTrackingContext, parquet_path: Path) -> dict[str, Any]:
    episode_id = parquet_path.stem
    video_path = parquet_path.with_suffix(".mp4")
    output_dir = args.output_root / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started_total = time.perf_counter()
    (instruction, instruction_column) = get_task_instruction_with_fallback(parquet_path, args.task_instruction_columns)
    first_frame = read_first_frame(context, video_path, args.first_video_frame_index)
    (width, height) = first_frame.size
    first_frame_path = output_dir / "first_frame.jpg"
    save_first_frame(context, first_frame, first_frame_path)
    started = time.perf_counter()
    vlm_target_object = run_vlm_target_object(args, context, instruction, first_frame_path)
    vlm_target_seconds = time.perf_counter() - started
    log_timing(f"{episode_id}.vlm_target_object", vlm_target_seconds)
    target_object = vlm_target_object["target_object"]
    pointing_target = vlm_target_object.get("referring_expression") or target_object
    sam3_text_prompt = target_object if args.sam3_prompt_source == "target_object" else pointing_target
    started = time.perf_counter()
    vlm_pointing = run_vlm_pointing(args, context, pointing_target, first_frame_path, width, height)
    vlm_pointing_seconds = time.perf_counter() - started
    log_timing(f"{episode_id}.vlm_pointing", vlm_pointing_seconds)
    sam3_output_dir = output_dir / "sam3_candidates"
    (detections, sam3_metadata, sam3_seconds) = run_sam3_candidates(
        args, context, first_frame, sam3_text_prompt, sam3_output_dir
    )
    log_timing(f"{episode_id}.sam3_text", sam3_seconds)
    started = time.perf_counter()
    (best, distance_top_candidates) = choose_best_candidate(detections, vlm_pointing["point_xy"], args.distance_topk)
    select_seconds = time.perf_counter() - started
    payload = {
        "status": "ok",
        "episode_id": episode_id,
        "video_path": str(video_path),
        "parquet_path": str(parquet_path),
        "first_video_frame_index": args.first_video_frame_index,
        "first_frame_path": str(first_frame_path),
        "image_size": [width, height],
        "instruction": instruction,
        "instruction_column": instruction_column,
        "instruction_fallback_columns": args.task_instruction_columns,
        "vlm_target_object": vlm_target_object,
        "vlm_pointing": vlm_pointing,
        "sam3": {
            "model": args.sam3_model,
            "model_path": str(context.sam3_model_path) if context.sam3_model_path is not None else None,
            "device": context.device,
            "score_threshold": args.score_threshold,
            "mask_threshold": args.mask_threshold,
            "max_candidates": args.max_candidates,
            "text_prompt": sam3_text_prompt,
            "prompt_source": args.sam3_prompt_source,
            **sam3_metadata,
            "detections": detections,
        },
        "selection": {
            "method": "distance_topk_then_highest_sam3_score",
            "distance_topk": args.distance_topk,
            "distance_top_candidates": distance_top_candidates,
            "best_candidate": best,
        },
        "timing_seconds": {
            "vlm_target_object": vlm_target_seconds,
            "vlm_pointing": vlm_pointing_seconds,
            "sam3_text": sam3_seconds,
            "select_best": select_seconds,
            "total": time.perf_counter() - started_total,
        },
    }
    if args.run_tracking:
        started = time.perf_counter()
        payload["tracking"] = run_target_tracking(args, context, payload, output_dir)
        payload["timing_seconds"]["tracking"] = time.perf_counter() - started
    output_json = output_dir / "video_object_detection_new.json"
    output_json.write_text(
        json.dumps(common.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload["output_json"] = str(output_json)
    return payload


def object_tracking_config() -> SimpleNamespace:
    return SimpleNamespace(
        output_root=Path("outputs"),
        task_instruction_columns=list(DEFAULT_TASK_INSTRUCTION_FALLBACK_COLUMNS),
        first_video_frame_index=0,
        run_tracking=False,
        sam3_video_model=OBJECT_SAM3_MODEL,
        sam3_video_cache_dir=None,
        sam3_video_torch_dtype="bf16",
        vlm_model=os.environ.get("MODEL_NAME", "auto"),
        vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        vllm_api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        vlm_max_new_tokens=256,
        vlm_temperature=0.0,
        vlm_top_p=0.95,
        vlm_top_k=20,
        vlm_min_p=0.0,
        vlm_presence_penalty=1.5,
        vlm_repetition_penalty=1.0,
        vlm_timeout=300,
        sam3_model=OBJECT_SAM3_MODEL,
        sam3_cache_dir=None,
        sam3_torch_dtype="bf16",
        device="auto",
        sam3_prompt_source="referring_expression",
        score_threshold=0.0,
        mask_threshold=0.5,
        max_candidates=10,
        distance_topk=3,
    )


GRIPPER_SAM3_MODEL = "facebook/sam3"
GRIPPER_VLM_SELECTION_PROMPT = '\nYou are selecting the best segmentation candidate for a robot end-effector.\n\nYou will see the original frame first. It is only reference context and is not a candidate.\nThen you will see {num_candidates} candidate images in order. Image 2 is candidate 1, image 3 is candidate 2, and so on.\nEach candidate image has a red bounding box for one proposed robot end-effector region.\n\nChoose exactly one candidate such that the bounding box best matches and covers the full robot end-effector.\nReturn only valid JSON with this schema:\n{{\n  "selected_candidate": id\n}}\n\nThe selected_candidate must be an integer from 1 to {num_candidates}.\n'.strip()


def parse_selected_candidate(parsed: dict[str, Any], num_candidates: int) -> int:
    value = parsed.get("selected_candidate")
    if value is None:
        value = parsed.get("candidate")
    if value is None:
        value = parsed.get("candidate_id")
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"VLM did not return an integer selected_candidate: {parsed}") from exc
    if selected > num_candidates and 2 <= selected <= num_candidates + 1:
        selected -= 1
    if selected < 1 or selected > num_candidates:
        raise RuntimeError(f"selected_candidate out of range 1..{num_candidates}: {selected}")
    return selected


class GripperTrackingContext:
    def __init__(self, args: SimpleNamespace):
        (self.np, self.torch, self.imageio, self.Image, self.ImageDraw) = common.load_common_modules()
        self.device = common.auto_device(self.torch, args.device)
        self._vlm_client = None
        self._sam3_text = None
        self._sam3_video = None

    def vlm_client(self, args: SimpleNamespace):
        if self._vlm_client is None:
            self._vlm_client = build_vllm_json_client(args)
        return self._vlm_client

    def sam3_text(self, args: SimpleNamespace):
        key = (args.sam3_text_model, args.sam3_text_cache_dir, args.sam3_text_torch_dtype, self.device)
        if self._sam3_text is None or self._sam3_text[0] != key:
            (model, processor, model_dir) = sam3.load_sam3(
                args.sam3_text_model, args.sam3_text_cache_dir, self.torch, self.device, args.sam3_text_torch_dtype
            )
            self._sam3_text = (key, model, processor, model_dir)
        return (self._sam3_text[1], self._sam3_text[2], self._sam3_text[3])

    def sam3_video(self, args: SimpleNamespace):
        key = (args.sam3_video_model, args.sam3_video_cache_dir, args.sam3_video_torch_dtype, self.device)
        if self._sam3_video is None or self._sam3_video[0] != key:
            (model, processor, metadata) = sam3.load_video_tracker(args, self.torch, self.device)
            self._sam3_video = (key, model, processor, metadata)
        return (self._sam3_video[1], self._sam3_video[2], self._sam3_video[3])


def log_timing(stage: str, seconds: float, note: str | None = None) -> None:
    suffix = f" ({note})" if note else ""
    print(f"[timer] {stage}: {seconds:.3f}s{suffix}", flush=True)


def gripper_bbox_center(bbox: list[float] | None) -> list[float]:
    if bbox is None:
        return [0.0, 0.0]
    return [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]


def generate_text_candidates(args, context, anchor_image, output_dir):
    (model, processor, model_dir) = context.sam3_text(args)
    (processed, metadata) = sam3.run_sam3_text(
        context.np,
        context.torch,
        model,
        processor,
        anchor_image,
        args.text_prompt,
        context.device,
        args.sam3_text_score_threshold,
        args.sam3_text_mask_threshold,
    )
    candidates = sam3.processed_to_candidates(context.np, context.torch, processed)
    detections = sam3.save_candidates(context.Image, context.np, candidates, output_dir, args.max_candidates)
    masks = [candidates[detection["source_index"]]["mask"] for detection in detections]
    metadata["model_dir"] = str(model_dir)
    return (masks, detections, metadata, str(model_dir))


def save_bbox_only_candidate_images(
    args: SimpleNamespace,
    context: GripperTrackingContext,
    anchor_image,
    detections: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    output_dir = output_root / "vlm_selector_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = detections[: args.vlm_selector_max_candidates]
    result = []
    for rank_idx, detection in enumerate(selected):
        bbox = detection.get("bbox_from_mask")
        image = anchor_image.copy()
        if bbox is not None:
            draw = context.ImageDraw.Draw(image)
            draw.rectangle(tuple(bbox), outline="red", width=4)
            label = f"candidate {rank_idx + 1}"
            text_xy = (max(0, int(bbox[0])), max(0, int(bbox[1]) - 22))
            draw.rectangle((text_xy[0], text_xy[1], text_xy[0] + 112, text_xy[1] + 20), fill="black")
            draw.text((text_xy[0] + 4, text_xy[1] + 3), label, fill="red")
        path = output_dir / f"candidate_bbox_only_{rank_idx:02d}.jpg"
        image.save(path)
        item = dict(detection)
        item["index"] = rank_idx
        item["original_candidate_index"] = detection.get("index")
        item["vlm_selector_bbox_only_path"] = str(path)
        result.append(item)
    return result


def select_candidate_with_vlm(
    args: SimpleNamespace, context: GripperTrackingContext, first_frame_path: Path, detections: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_paths = []
    for detection in detections:
        candidate_path = detection.get("vlm_selector_bbox_only_path") or detection.get("overlay_path")
        if not candidate_path:
            raise RuntimeError(f"Candidate {detection.get('index')} has no candidate image path.")
        path = Path(candidate_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate image: {path}")
        candidate_paths.append(path)
    if not candidate_paths:
        raise RuntimeError("No SAM3 text candidates were generated.")
    prompt = GRIPPER_VLM_SELECTION_PROMPT.format(num_candidates=len(candidate_paths))
    (parsed, raw_content, reasoning) = context.vlm_client(args).chat_json(prompt, [first_frame_path, *candidate_paths])
    selected_number = parse_selected_candidate(parsed, len(candidate_paths))
    selected_idx = selected_number - 1
    rankings = []
    for rank, detection in enumerate(detections, start=1):
        candidate = dict(detection)
        candidate["vlm_rank"] = rank
        candidate["vlm_selected"] = int(candidate["index"]) == selected_idx
        rankings.append(candidate)
    selected_detection = dict(detections[selected_idx])
    selected_detection.update(
        {
            "vlm_selected_candidate_number": selected_number,
            "vlm_reason": str(parsed.get("reason") or "").strip(),
            "vlm_raw_response": parsed,
            "vlm_raw_content": raw_content,
            "vlm_reasoning": reasoning,
            "vlm_candidate_image_paths": [str(path) for path in candidate_paths],
        }
    )
    return (rankings, selected_detection)


def track_episode_gripper(args: SimpleNamespace, context: GripperTrackingContext | None = None) -> dict[str, Any]:
    context = context or GripperTrackingContext(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    started = time.perf_counter()
    (first_frame_path, last_frame_path, frame_width, frame_height, first_idx, last_idx) = common.save_context_frames(
        args, context.imageio, context.Image
    )
    timing["context_frames_seconds"] = time.perf_counter() - started
    log_timing("context_frames", timing["context_frames_seconds"])
    anchor_idx = int(last_idx)
    anchor_frame_path = args.output_dir / "anchor_frame.jpg"
    anchor_image = context.Image.open(last_frame_path).convert("RGB")
    anchor_image.save(anchor_frame_path)
    started = time.perf_counter()
    (masks, detections, text_metadata, sam3_text_model_dir) = generate_text_candidates(
        args, context, anchor_image, args.output_dir
    )
    timing["sam3_text_candidate_seconds"] = time.perf_counter() - started
    log_timing("sam3_text_candidate", timing["sam3_text_candidate_seconds"])
    started = time.perf_counter()
    vlm_selector_detections = save_bbox_only_candidate_images(args, context, anchor_image, detections, args.output_dir)
    (candidate_rankings, selected_detection) = select_candidate_with_vlm(
        args, context, anchor_frame_path, vlm_selector_detections
    )
    timing["vlm_candidate_selector_seconds"] = time.perf_counter() - started
    log_timing("vlm_candidate_selector", timing["vlm_candidate_selector_seconds"])
    anchor_image = context.Image.open(anchor_frame_path).convert("RGB")
    selected_idx = int(selected_detection.get("original_candidate_index", selected_detection["index"]))
    if selected_idx < 0 or selected_idx >= len(masks):
        raise RuntimeError(f"Selected candidate index out of range: {selected_idx}")
    selected_mask = masks[selected_idx]
    selected_bbox = selected_detection.get("bbox_from_mask")
    selected_mask_path = args.output_dir / "selected_mask.png"
    common.save_mask_png(context.Image, context.np, selected_mask, selected_mask_path)
    selected_detection["selected_mask_path"] = str(selected_mask_path)
    tracking_payload = None
    if args.run_tracking:
        started = time.perf_counter()
        (sam3_video_model, sam3_video_processor, sam3_video_metadata) = context.sam3_video(args)
        log_timing("sam3_video_load", time.perf_counter() - started)
        started = time.perf_counter()
        (frame_records, tracking_metadata) = sam3.track_video(
            context.np,
            context.torch,
            sam3_video_model,
            sam3_video_processor,
            args.video,
            selected_bbox,
            selected_mask,
            gripper_bbox_center(selected_bbox),
            frame_width,
            frame_height,
            args.sam3_video_obj_id,
            context.device,
            args.sam3_video_torch_dtype,
            args.sam3_video_binarize,
            args.sam3_video_prompt_type,
            prompt_frame_idx=anchor_idx,
            reverse=args.tracking_reverse,
        )
        tracking_frames = [
            {"frame_index": int(record["frame_index"]), "bbox_xyxy": record.get("bbox_xyxy")}
            for record in frame_records
        ]
        timing["sam3_tracking_seconds"] = time.perf_counter() - started
        log_timing("sam3_tracking", timing["sam3_tracking_seconds"])
        tracking_payload = {
            **sam3_video_metadata,
            **tracking_metadata,
            "num_frames": len(tracking_frames),
            "frames": tracking_frames,
        }
    else:
        timing["sam3_tracking_seconds"] = 0.0
        log_timing("sam3_tracking", 0.0, "disabled")
    timing["total_model_seconds"] = sum(timing.values())
    log_timing("total_model", timing["total_model_seconds"])
    payload = {
        "status": "ok",
        "video_path": str(args.video),
        "parquet_path": str(args.parquet) if args.parquet else None,
        "frame_context": {
            "first_frame_path": str(first_frame_path),
            "last_frame_path": str(last_frame_path),
            "anchor_frame_path": str(anchor_frame_path),
            "anchor_strategy": "last",
            "anchor_frame_index": anchor_idx,
            "first_video_frame_index": first_idx,
            "last_video_frame_index": last_idx,
            "width": frame_width,
            "height": frame_height,
        },
        "text_prompt": args.text_prompt,
        "sam3_text_candidates": {
            "model": args.sam3_text_model,
            "model_dir": sam3_text_model_dir,
            "score_threshold": args.sam3_text_score_threshold,
            "mask_threshold": args.sam3_text_mask_threshold,
            "max_candidates": args.max_candidates,
            "prompt_metadata": text_metadata,
            "detections": detections,
        },
        "candidate_selection": {
            "selector": "vlm",
            "model": args.vlm_model,
            "max_candidates": args.vlm_selector_max_candidates,
            "candidate_image_type": "bbox_only_overlay",
            "candidate_rankings": candidate_rankings,
            "selected_detection": selected_detection,
        },
        "selected_bbox_xyxy": selected_bbox,
        "tracking": tracking_payload,
        "timing_seconds": timing,
    }
    output_json = args.output_dir / "video_gripper_detection.json"
    output_json.write_text(
        json.dumps(common.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote gripper detection output: {output_json}")
    print(f"Selected gripper bbox: {selected_bbox}")
    return {"output_json": output_json, "payload": payload}


def gripper_tracking_config() -> SimpleNamespace:
    return SimpleNamespace(
        video=None,
        parquet=None,
        output_dir=None,
        first_video_frame_index=0,
        last_video_frame_index=-1,
        device="auto",
        text_prompt="robot gripper",
        sam3_text_model=GRIPPER_SAM3_MODEL,
        sam3_text_cache_dir=None,
        sam3_text_torch_dtype="fp32",
        sam3_text_score_threshold=0.0,
        sam3_text_mask_threshold=0.5,
        max_candidates=10,
        vlm_selector_max_candidates=5,
        vlm_model=os.environ.get("MODEL_NAME", "auto"),
        vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        vllm_api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        vlm_max_new_tokens=512,
        vlm_temperature=0.0,
        vlm_top_p=0.95,
        vlm_top_k=20,
        vlm_min_p=0.0,
        vlm_presence_penalty=1.5,
        vlm_repetition_penalty=1.0,
        vlm_timeout=300,
        run_tracking=True,
        sam3_video_model=GRIPPER_SAM3_MODEL,
        sam3_video_cache_dir=None,
        sam3_video_torch_dtype="bf16",
        sam3_video_binarize=True,
        sam3_video_obj_id=0,
        sam3_video_prompt_type="mask",
        tracking_reverse=True,
    )
