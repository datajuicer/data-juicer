# video_universal_segmentation_mapper

OneFormer video segmentation 视频分割

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

The operator uses `shi-labs/oneformer_ade20k_swin_large` by default. A compatible
local Hugging Face model directory is also accepted through `model_path`.
Results are stored in `__dj__meta__.video_universal_segmentation_tags`:

- `semantic_segmentation_map`: H × W class maps.
- `instance_segmentation_map` and `panoptic_segmentation_map`: H × W segment-ID maps.
- `instance_segmentation_info` and `panoptic_segmentation_info`: the matching
  segment metadata returned by the processor.
- `frame_paths` and `video_indices`: the source frame and video of each map.

Maps use each original frame's resolution. The three tasks are enabled by default
and can be selected independently with `if_output_semantic_segmentation`,
`if_output_instance_segmentation`, and `if_output_panoptic_segmentation`.
Disabled tasks return empty lists. Disabling all tasks avoids model inference.
CPU and CUDA execution use the loaded model's device.

## Visualization 可视化

`if_save_visualization` defaults to true. Images are written below
`save_visualization_dir/<operator>/<video-index>_<frame-path-hash>/`.
Segmentation adds a subdirectory for each enabled task. This keeps outputs from
different input sequences separate. Set `if_save_visualization: false` to disable
writing images. Prediction arrays are returned as JSON-serializable lists.

## Related links 相关链接

- [Source](../../../data_juicer/ops/mapper/video_universal_segmentation_mapper.py)
- [Regression tests](../../../tests/ops/mapper/test_video_perception_regression.py)
- [Operators](../../Operators.md)
