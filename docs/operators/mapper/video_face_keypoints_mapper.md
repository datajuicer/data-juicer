# video_face_keypoints_mapper

Detect 98 face keypoints with InsightFace detection and LDEQ regression.
使用 InsightFace 与 LDEQ 检测 98 个人脸关键点。

## Inputs and frame order 输入与帧顺序

Accepts videos under the configured `video_key`, or an ordered flat/grouped
frame-path list under `video_frames`. Explicit frame lists take precedence.
Paths are processed exactly in the provided order, without directory scanning.
Empty inputs return empty predictions without loading inference models.

See [video frame sequence contracts](../../VideoFrameSequences.md).

## Model setup 模型准备

Prepare the checkpoint and upstream dependencies from
[LDEQ_RwR](https://github.com/polo5/LDEQ_RwR) and InsightFace.
`ldeq_model_path` accepts a local `final.pth.tar` checkpoint. When absent, the
inherited loader downloads the upstream checkpoint and InsightFace detector.
The real-model path requires the upstream CUDA/ONNX environment.

```yaml
- video_face_keypoints_mapper:
    ldeq_model_path: /path/to/final.pth.tar
    frame_num: 3
    if_save_visualization: false
```

## Output 输出

`__dj__meta__.video_face_keypoints_tags` contains JSON-serializable lists:

- `face_keypoints`: one entry per frame, then per detected face, with 98 XY points
  in original image coordinates.
- `face_bboxes`: matching XYXY bounding boxes.
- `frame_paths`: exact source frame paths in prediction order.
- `video_indices`: zero-based source group indices.

Frames without faces retain empty detection lists at their original positions.
Visualization files are separated by operator and input sequence.

## Related links 相关链接

- [Source](../../../data_juicer/ops/mapper/video_face_keypoints_mapper.py)
- [Tests](../../../tests/ops/mapper/test_video_face_keypoints_mapper.py)
- [Frame contract regressions](../../../tests/ops/mapper/test_video_frame_sequence_contract.py)
