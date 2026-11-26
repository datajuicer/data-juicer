# video_optical_flow_mapper

RAFT optical flow 视频光流

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

The operator uses torchvision RAFT-Large with its default pretrained weights
and stores results in `__dj__meta__.video_optical_flow_tags`:

- `pred_flow`: one 2 × 520 × 960 flow field per consecutive frame pair.
- `frame_pairs`: the two input paths corresponding to each flow field.
- `video_indices`: the source video index for each pair.

Flow components are horizontal and vertical displacement in pixels at the model's
resized 520 × 960 resolution. They are not rescaled to the original video size.
No pair crosses a video boundary. A video with fewer than two frames contributes
no flow field. CPU and CUDA execution use the loaded model's device.

## Visualization 可视化

`if_save_visualization` defaults to true. Images are written below
`save_visualization_dir/<operator>/<video-index>_<frame-path-hash>/`.
Segmentation adds a subdirectory for each enabled task. This keeps outputs from
different input sequences separate. Set `if_save_visualization: false` to disable
writing images. Prediction arrays are returned as JSON-serializable lists.

## Related links 相关链接

- [Source](../../../data_juicer/ops/mapper/video_optical_flow_mapper.py)
- [Regression tests](../../../tests/ops/mapper/test_video_perception_regression.py)
- [Operators](../../Operators.md)
