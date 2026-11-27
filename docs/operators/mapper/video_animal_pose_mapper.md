# video_animal_pose_mapper

Detect quadruped animal keypoints with YOLOE and ViTPose (APT-36K).
使用 YOLOE 与 ViTPose 提取四足动物关键点。

## Inputs and frame order 输入与帧顺序

Accepts videos under the configured `video_key`, or `video_frames` containing
an ordered list of image paths or a list of frame lists (one per video).
Explicit frame lists take precedence, including an explicitly empty list.
The supplied order and subset are preserved; directory files are not rescanned.
Empty sequences return empty predictions without loading inference models.

See [video frame sequence contracts](../../VideoFrameSequences.md) for grouping,
empty-input behavior and integration with calibration and reconstruction.

## Model setup 模型准备

This operator uses the legacy [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
MMPose API and [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/).
Prepare the upstream dependencies and APT-36K checkpoint in a compatible
environment. Set `vitpose_model_path`, `vitpose_config` and `yoloe_model_path`
to the prepared model files. The inherited setup code may install the legacy
MMPose/MMCV stack; use a dedicated environment for real-model runs.

`animal_class` accepts a list of category names. An empty list selects the
default quadruped categories. `frame_num` defaults to 3; positive `duration`
samples frames from each full video segment.

```yaml
- video_animal_pose_mapper:
    vitpose_model_path: /path/to/apt36k.pth
    vitpose_config: configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py
    yoloe_model_path: /path/to/yoloe-26x-seg.pt
    animal_class: [cat, dog]
    frame_num: 3
    if_save_visualization: false
```

## Output 输出

`__dj__meta__.video_animal_pose_tags` contains JSON-serializable lists:

- `pose_list`: one entry per frame, then per detected animal, with 17 XY keypoints.
- `pose_score_list`: matching per-keypoint confidence scores.
- `animal_bboxes`: matching XYXY bounding boxes.
- `frame_paths`: exact source frame paths in prediction order.
- `video_indices`: zero-based source group indices, retaining empty-group offsets.

A frame without a detected animal still occupies one output entry with empty
detections. Visualization files are separated by operator and input sequence.

## Related links 相关链接

- [Source](../../../data_juicer/ops/mapper/video_animal_pose_mapper.py)
- [Tests](../../../tests/ops/mapper/test_video_animal_pose_mapper.py)
- [Frame contract regressions](../../../tests/ops/mapper/test_video_frame_sequence_contract.py)
