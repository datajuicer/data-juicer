"""SAM3 segmentation candidates and full-video mask propagation."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from data_juicer.utils import robot_utils as common


def resolve_modelscope_model(model_name: str, cache_dir: Path | None) -> Path:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path
    if cache_dir is not None:
        cached_model_path = cache_dir / model_name
        if cached_model_path.exists():
            return cached_model_path
    return common.snapshot_download_model(model_name, cache_dir)


def load_sam3(model_name: str, cache_dir: Path | None, torch, device: str, dtype_name: str):
    from transformers import Sam3Model, Sam3Processor

    model_path = resolve_modelscope_model(model_name, cache_dir)
    dtype = common.torch_dtype(torch, dtype_name)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = Sam3Model.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
    return (model, processor, model_path)


def run_sam3_text(
    np, torch, model, processor, image, prompt: str, device: str, score_threshold: float, mask_threshold: float
):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = common.move_inputs_to_device(dict(inputs), device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = [(image.height, image.width)]
    processed = processor.post_process_instance_segmentation(
        outputs, threshold=score_threshold, mask_threshold=mask_threshold, target_sizes=target_sizes
    )[0]
    metadata = {
        "prompt": prompt,
        "target_sizes": [list(target_sizes[0])],
        "input_keys": sorted(inputs.keys()),
        "output_fields": sorted((name for name in dir(outputs) if not name.startswith("_"))),
        "processed_keys": sorted(processed.keys()),
    }
    return (processed, metadata)


def score_from_segment(info: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "predicted_iou"):
        if key in info and info[key] is not None:
            return float(info[key])
    return None


def processed_to_candidates(np, torch, processed: dict[str, Any]) -> list[dict[str, Any]]:
    masks_value = processed.get("masks")
    if masks_value is not None:
        masks_np = common.tensor_to_numpy(np, torch, masks_value)
        masks_np = np.asarray(masks_np).squeeze()
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        elif masks_np.ndim > 3:
            masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
        boxes = common.tensor_to_numpy(np, torch, processed.get("boxes"))
        scores = common.tensor_to_numpy(np, torch, processed.get("scores"))
        candidates = []
        for idx in range(len(masks_np)):
            candidate: dict[str, Any] = {"mask": masks_np[idx]}
            if boxes is not None and idx < len(boxes):
                candidate["box"] = [float(value) for value in boxes[idx].tolist()]
            if scores is not None and idx < len(scores):
                candidate["score"] = float(scores[idx])
            candidates.append(candidate)
        return candidates
    segmentation = processed.get("segmentation")
    segments_info = processed.get("segments_info") or []
    if segmentation is None:
        return []
    segmentation_np = common.tensor_to_numpy(np, torch, segmentation)
    candidates = []
    for idx, info in enumerate(segments_info):
        segment_id = info.get("id", idx)
        mask = segmentation_np == segment_id
        if not mask.any():
            continue
        normalized_info = common.json_ready(info)
        candidates.append({"mask": mask, "score": score_from_segment(normalized_info), "segment_info": normalized_info})
    return candidates


def save_candidates(Image, np, candidates, output_dir: Path, max_candidates: int):
    candidate_dir = output_dir / "sam3_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: item[1].get("score") if item[1].get("score") is not None else float("-inf"),
        reverse=True,
    )
    detections = []
    for rank, (source_index, candidate) in enumerate(ranked[:max_candidates]):
        mask = candidate["mask"]
        path = candidate_dir / f"candidate_mask_{rank:02d}.png"
        common.save_mask_png(Image, np, mask, path)
        detections.append(
            {
                "index": rank,
                "source_index": source_index,
                "score": candidate.get("score"),
                "bbox_from_mask": common.mask_to_bbox(np, mask),
                "bbox_from_model": candidate.get("box"),
                "mask_path": str(path),
            }
        )
    return detections


def load_video_tracker(args, torch, device: str):
    try:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers' with Sam3TrackerVideoModel/Sam3TrackerVideoProcessor. Install a SAM3-video-capable Transformers build in the main .venv."
        ) from exc
    requested_model_path = Path(args.sam3_video_model)
    if requested_model_path.exists():
        model_dir = requested_model_path
    elif args.sam3_video_cache_dir is not None and (args.sam3_video_cache_dir / args.sam3_video_model).exists():
        model_dir = args.sam3_video_cache_dir / args.sam3_video_model
    else:
        model_dir = common.snapshot_download_model(args.sam3_video_model, args.sam3_video_cache_dir)
    dtype = common.torch_dtype(torch, args.sam3_video_torch_dtype)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = Sam3TrackerVideoModel.from_pretrained(str(model_dir), **model_kwargs)
    if dtype is not None:
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device)
    model.eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    return (
        model,
        processor,
        {
            "model": args.sam3_video_model,
            "model_dir": str(model_dir),
            "torch_dtype": args.sam3_video_torch_dtype,
            "device": device,
            "loader": "transformers.Sam3TrackerVideoModel",
        },
    )


def point_from_bbox_or_point(bbox: list[float] | None, point_xy: list[float]) -> list[float]:
    if bbox is not None:
        return [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]
    return [float(point_xy[0]), float(point_xy[1])]


def add_prompt_to_session(
    processor,
    inference_session,
    bbox: list[float] | None,
    mask: Any | None,
    point_xy: list[float],
    obj_id: int,
    prompt_type: str,
    frame_idx: int = 0,
) -> tuple[str, list[float] | None]:
    prompt_point = point_from_bbox_or_point(bbox, point_xy)
    if prompt_type == "mask" and mask is not None:
        processor.add_inputs_to_inference_session(
            inference_session=inference_session, frame_idx=int(frame_idx), obj_ids=int(obj_id), input_masks=mask
        )
        return ("mask_from_selected_candidate", prompt_point)
    if prompt_type == "bbox" and bbox is not None:
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=int(frame_idx),
            obj_ids=int(obj_id),
            input_boxes=[[[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]]],
        )
        return ("bbox_from_selected_candidate", prompt_point)
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=int(frame_idx),
        obj_ids=int(obj_id),
        input_points=[[[[float(prompt_point[0]), float(prompt_point[1])]]]],
        input_labels=[[[1]]],
    )
    return ("point_from_selected_bbox" if bbox is not None else "vlm_point", prompt_point)


def first_mask_from_postprocessed(np, torch, masks: Any):
    masks = common.tensor_to_numpy(np, torch, masks).squeeze()
    if masks.ndim == 2:
        return masks
    if masks.ndim >= 3:
        return masks.reshape((-1,) + masks.shape[-2:])[0]
    return None


def postprocess_output_mask(np, torch, processor, inference_session, output, binarize: bool):
    processed = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=binarize,
    )[0]
    return first_mask_from_postprocessed(np, torch, processed)


def track_video(
    np,
    torch,
    model,
    processor,
    video_path: Path,
    bbox: list[float] | None,
    mask: Any | None,
    point_xy: list[float],
    frame_width: int,
    frame_height: int,
    obj_id: int,
    device: str,
    dtype_name: str,
    binarize: bool,
    prompt_type: str,
    prompt_frame_idx: int = 0,
    reverse: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import transformers

    dtype = common.torch_dtype(torch, dtype_name)
    (video_frames, _metadata) = transformers.video_utils.load_video(str(video_path))
    inference_session = processor.init_video_session(video=video_frames, inference_device=device, dtype=dtype)
    (tracking_prompt_type, prompt_point) = add_prompt_to_session(
        processor, inference_session, bbox, mask, point_xy, obj_id, prompt_type, frame_idx=prompt_frame_idx
    )
    frame_records: list[dict[str, Any]] = []
    autocast_context = (
        torch.autocast("cuda", dtype=dtype) if device.startswith("cuda") and dtype is not None else nullcontext()
    )
    seen: set[int] = set()
    with torch.inference_mode(), autocast_context:
        initial_output = model(inference_session=inference_session, frame_idx=int(prompt_frame_idx))
        initial_mask = postprocess_output_mask(np, torch, processor, inference_session, initial_output, binarize)
        if initial_mask is not None:
            bbox = common.mask_to_bbox(np, initial_mask)
            frame_records.append({"frame_index": int(prompt_frame_idx), "mask": initial_mask, "bbox_xyxy": bbox})
            seen.add(int(prompt_frame_idx))
        iterator = model.propagate_in_video_iterator(
            inference_session, start_frame_idx=int(prompt_frame_idx), reverse=bool(reverse)
        )
        fallback_frame_idx = 0
        while True:
            try:
                output = next(iterator)
            except StopIteration:
                break
            frame_idx = int(getattr(output, "frame_idx", fallback_frame_idx))
            fallback_frame_idx += 1
            if frame_idx in seen:
                continue
            out_mask = postprocess_output_mask(np, torch, processor, inference_session, output, binarize)
            if out_mask is None:
                continue
            bbox = common.mask_to_bbox(np, out_mask)
            frame_records.append({"frame_index": frame_idx, "mask": out_mask, "bbox_xyxy": bbox})
            seen.add(frame_idx)
    return (
        sorted(frame_records, key=lambda item: item["frame_index"]),
        {
            "tracking_prompt_type": tracking_prompt_type,
            "tracking_prompt_point_xy": prompt_point,
            "tracking_prompt_obj_id": int(obj_id),
            "tracking_uses_selected_mask_as_prompt": tracking_prompt_type == "mask_from_selected_candidate",
            "tracking_requested_prompt_type": prompt_type,
            "tracking_prompt_frame_idx": int(prompt_frame_idx),
            "tracking_reverse": bool(reverse),
        },
    )
