import os
from pathlib import Path

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.robot_utils import episode_root, read_json, write_json

from ..base_op import OPERATORS, UNFORKABLE, Mapper

OP_NAME = "robot_depth_estimation_mapper"


@OPERATORS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
class RobotDepthEstimationMapper(Mapper):
    """Estimate every camera frame and persist aligned depth IR outside Arrow metadata."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = None,
        inference_batch_size: int = 8,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        output_dir: str = "output/episodes",
        *args,
        **kwargs,
    ):
        """Initialize the mapper.

        :param model_path: DA3 checkpoint directory or model ID. Defaults to DA3_MODEL or depth-anything/DA3METRIC-LARGE.
        :param inference_batch_size: Number of frames per inference batch. Use zero to process a complete view jointly.
        :param process_res: Resolution used for depth inference.
        :param process_res_method: Full-image resize method: upper_bound_resize or lower_bound_resize.
        :param output_dir: Directory for episode artifacts.
        :param args: Additional positional arguments for Mapper.
        :param kwargs: Additional keyword arguments for Mapper.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        if inference_batch_size < 0 or process_res < 14:
            raise ValueError("Use a nonnegative batch size and a resolution of at least 14")
        if process_res_method not in {"upper_bound_resize", "lower_bound_resize"}:
            raise ValueError("Depth IR requires full-image resize preprocessing")
        self.model_path = model_path or os.environ.get("DA3_MODEL", "depth-anything/DA3METRIC-LARGE")
        self.inference_batch_size = inference_batch_size
        self.process_res = process_res
        self.process_res_method = process_res_method
        self._model = None

    def process_single(self, sample, rank=None):
        torch = LazyLoader("torch")
        from depth_anything_3.api import DepthAnything3

        from data_juicer.utils.robot_depth_utils import save_prediction_batch

        root = episode_root(sample, self.output_dir)
        if sample["source"] != "droid":
            raise ValueError("Episode DA3 input must contain DROID camera frames")
        if self._model is None:
            device = f"cuda:{rank or 0}" if self.use_cuda() and torch.cuda.is_available() else "cpu"
            self._model = DepthAnything3.from_pretrained(self.model_path).to(device).eval()
        model_name = self._model.model_name
        independent = model_name.startswith(("da3metric-", "da3mono-"))
        if not independent and self.inference_batch_size:
            raise ValueError("Any-view models require inference_batch_size=0 to retain the whole view context")
        views = read_json(root / "ir/sampling.json")["views"]
        manifest = {
            "operator": self._name,
            "model_name": model_name,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "inference_scope": "independent_monocular_batches" if independent else "per_view_joint",
            "views": {},
        }
        for camera, view in views.items():
            if view["frame_indices"] != list(range(view["source_frame_count"])):
                raise ValueError("DA3 episode processing requires every source frame")
            fps = float(view["source_fps"])
            if fps <= 0:
                raise ValueError("Video FPS must be positive")
            count = len(view["images"])
            size = view["image_size"]
            batch = self.inference_batch_size or count
            calibration = sample.get("camera_intrinsics", {}).get(camera)
            records = []
            for begin in range(0, count, batch):
                images = view["images"][begin : begin + batch]
                with torch.inference_mode():
                    prediction = self._model.inference(
                        image=images, process_res=self.process_res, process_res_method=self.process_res_method
                    )
                entries = [
                    {
                        "sample_index": i,
                        "source_frame_index": view["frame_indices"][i],
                        "timestamp_seconds": view["frame_indices"][i] / fps,
                    }
                    for i in range(begin, begin + len(images))
                ]
                values = save_prediction_batch(
                    prediction,
                    entries,
                    root / "ir/da3_depth" / camera,
                    (size["width"], size["height"]),
                    model_name,
                    calibration,
                )
                for value in values:
                    value["path"] = str(Path(value["path"]).relative_to(root))
                records.extend(values)
                print(f"DA3 {camera}: {len(records)}/{count} source frames", flush=True)
            manifest["views"][camera] = {
                "source_frame_count": count,
                "fps": fps,
                "source_image_size": size,
                "frames": records,
            }
        path = root / "ir/depth_anything_3.json"
        write_json(path, manifest)
        sample["depth_ir"] = str(path)
        return sample
