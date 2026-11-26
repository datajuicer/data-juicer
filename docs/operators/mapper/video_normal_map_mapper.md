# video_normal_map_mapper

Metric3D surface normals 视频表面法线

## Input and frame order 输入与帧顺序

Provide a list of videos in `videos` (or the configured `video_key`), or provide
`video_frames` containing one ordered list of image paths or a list of frame lists,
one per video. Existing frame paths take precedence over video extraction.
Paths are processed in the supplied order. Directory contents are not used to
expand or reorder a frame list. An unreadable frame raises an error with its path.
Empty input returns empty result lists without loading a model.

输入可为视频路径列表，或 `video_frames` 中已按顺序排列的帧路径。
多个视频使用嵌套帧列表；不同视频分别处理，不将目录中的其他图片混入输入。

`frame_num` defaults to 3. With `duration: 0`, frames are sampled from the whole
video; a positive duration samples each segment separately. `frame_dir` is the
extraction directory. A null directory uses the Data-Juicer assets cache.

The operator stores results in `__dj__meta__.video_normal_map_tags`:

- `pred_norm`: one H × W × 3 list per frame, resized to that frame's original resolution.
- `frame_paths`: the corresponding input frame paths.
- `video_indices`: the zero-based source video index for each prediction.

`model_path` accepts a local Metric3D ONNX checkpoint. If it is not a file, the
operator downloads `onnx/model.onnx` from `onnx-community/metric3d-vit-large`.
Use `accelerator: cpu` with `onnxruntime`, or `accelerator: cuda` with a compatible
`onnxruntime-gpu` installation. The default accelerator is CUDA.
CPU worker environments use `onnxruntime`; CUDA worker environments use
`onnxruntime-gpu`. The two runtime packages are not requested together.
CUDA mode requires an active `CUDAExecutionProvider`; missing or failed provider
initialization raises an error.

## Visualization 可视化

`if_save_visualization` defaults to true. Images are written below
`save_visualization_dir/<operator>/<video-index>_<frame-path-hash>/`.
Segmentation adds a subdirectory for each enabled task. This keeps outputs from
different input sequences separate. Set `if_save_visualization: false` to disable
writing images. Prediction arrays are returned as JSON-serializable lists.

## Related links 相关链接

- [Source](../../../data_juicer/ops/mapper/video_normal_map_mapper.py)
- [Regression tests](../../../tests/ops/mapper/test_video_perception_regression.py)
- [Operators](../../Operators.md)
