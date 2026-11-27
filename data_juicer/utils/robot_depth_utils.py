"""File-backed DA3 predictions and tracked depth samples."""

from __future__ import annotations

from pathlib import Path

from data_juicer.utils.lazy_loader import LazyLoader

np = LazyLoader("numpy")


def resize_intrinsics(intrinsics, source_size, output_shape):
    matrix = np.asarray(intrinsics, dtype=np.float64).copy()
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Camera intrinsics must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or (not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError("Invalid pinhole camera intrinsics")
    (width, height) = source_size
    (out_h, out_w) = output_shape
    if min(width, height, out_h, out_w) <= 0:
        raise ValueError("Image dimensions must be positive")
    matrix[0] *= out_w / width
    matrix[1] *= out_h / height
    return matrix.astype(np.float32)


def depth_units(model_name, prediction_is_metric, has_calibration):
    if model_name.startswith("da3metric-"):
        return "metres" if has_calibration else "focal_normalized"
    return "metres" if prediction_is_metric else "relative"


def calibrated_depth(depth, model_name, intrinsics):
    result = np.asarray(depth, dtype=np.float32)
    if model_name.startswith("da3metric-") and intrinsics is not None:
        focal = (float(intrinsics[0, 0]) + float(intrinsics[1, 1])) / 2
        result = result * (focal / 300)
    return result


def sample_depth(depth, point):
    (x, y) = point
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    (height, width) = depth.shape
    col = min(width - 1, int(x / 1000 * width))
    row = min(height - 1, int(y / 1000 * height))
    value = float(depth[row, col])
    return value if np.isfinite(value) and value > 0 else None


def save_prediction_batch(prediction, entries, directory, source_size, model_name, intrinsics=None):
    depths = np.asarray(prediction.depth, dtype=np.float32)
    if depths.ndim != 3 or len(depths) != len(entries):
        raise ValueError("DA3 must return one depth map for every input frame")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_k = resize_intrinsics(intrinsics, source_size, depths.shape[1:]) if intrinsics is not None else None
    predicted_k = getattr(prediction, "intrinsics", None)
    extrinsics = getattr(prediction, "extrinsics", None)
    sky = getattr(prediction, "sky", None)
    for name, values in [("intrinsics", predicted_k), ("extrinsics", extrinsics), ("sky", sky)]:
        if values is not None and len(values) != len(entries):
            raise ValueError(f"DA3 {name} count does not match the input frames")
    output = []
    for index, entry in enumerate(entries):
        k = source_k if source_k is not None else np.asarray(predicted_k[index]) if predicted_k is not None else None
        values = {"depth": calibrated_depth(depths[index], model_name, k)}
        if k is not None:
            if k.shape != (3, 3) or not np.isfinite(k).all():
                raise ValueError("Invalid DA3 camera intrinsics")
            values["intrinsics"] = k.astype(np.float32)
        if extrinsics is not None:
            e = np.asarray(extrinsics[index], dtype=np.float32)
            if e.shape not in {(3, 4), (4, 4)} or not np.isfinite(e).all():
                raise ValueError("Invalid DA3 camera extrinsics")
            values["extrinsics"] = e
        if sky is not None:
            if np.asarray(sky[index]).shape != values["depth"].shape:
                raise ValueError("DA3 sky mask resolution differs from depth")
            values["sky"] = np.asarray(sky[index])
        filename = directory / f"{entry['source_frame_index']:06d}.npz"
        np.savez_compressed(filename, **values)
        record = dict(
            entry,
            path=str(filename),
            shape=list(values["depth"].shape),
            fields=sorted(values),
            units=depth_units(model_name, bool(prediction.is_metric), k is not None),
        )
        output.append(record)
    return output


def tracked_depth(root, manifest, camera, view, tracks):
    """Sample model depth at tracked centers, retaining both frame timelines."""
    from data_juicer.utils.robot_utils import normalized_center

    records = manifest["views"][camera]["frames"]
    if len(records) != len(view["frame_indices"]):
        raise ValueError("Depth and tracking frame counts differ")
    size = view["image_size"]
    output = []
    for index, record in enumerate(records):
        if record["sample_index"] != index or record["source_frame_index"] != view["frame_indices"][index]:
            raise ValueError("Depth and tracking frame indices differ")
        path = Path(root) / record["path"]
        with np.load(path, allow_pickle=False) as data:
            item = {"sample_index": index, "source_frame_index": record["source_frame_index"], "units": record["units"]}
            for name, boxes in tracks.items():
                box = boxes.get(index)
                item[name] = (
                    sample_depth(data["depth"], normalized_center(box, size["width"], size["height"]))
                    if box is not None
                    else None
                )
        output.append(item)
    return output
