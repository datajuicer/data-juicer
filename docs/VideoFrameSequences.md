# Ordered video frame sequences 有序视频帧序列

The animal-pose, face-keypoint, whole-body-pose, depth-estimation, MoGe
calibration and HaWoR reconstruction operators accept frame-only samples.
A `videos` field is not required when frames are already provided.

## Input shape and order 输入格式与顺序

```python
# A single ordered sequence. The order can be an explicit subset.
{"video_frames": ["clip/frame_10.jpg", "clip/frame_2.jpg"]}

# Several videos. Empty groups keep their positions.
{"video_frames": [[], ["a/frame_0.jpg"], ["b/frame_2.jpg", "b/frame_10.jpg"]]}
```

`video_extract_frames_mapper` already produces the grouped format. Pose
operators use its returned paths directly when extracting raw videos. They
do not sort filenames, scan the output directory or include stale files.
The caller must supply explicit frame lists in the desired temporal order.
An explicit empty frame field takes precedence over raw video paths.

MoGe and HaWoR retain their existing support for encoded image bytes. The
other operators accept image paths. MoGe and HaWoR use the configured
`frame_field`; pose and depth operators use `video_frames`.

## Empty frames and result correspondence 空帧与结果对应

Animal, face and whole-body pose results keep their existing prediction keys
and add `frame_paths` and `video_indices`. Prediction lists are flattened in
group order; the mapping fields identify the source of every entry.
A frame without a detected subject retains an empty result entry. Empty
groups add no frame results but do not renumber later `video_indices`.

MoGe and HaWoR retain one result dictionary per video group. Empty MoGe
groups have an empty dictionary. Empty HaWoR groups have zero focal/FoV
values and empty left/right hand outputs. Empty inputs do not load models.

Unreadable path frames raise an error rather than silently shifting
predictions, calibration or reconstructed frame IDs.

## Depth and frame rates 深度与帧率

Frame-only depth inputs require positive finite `fps`, supplied either as a
shared scalar or one value per video group. Empty groups do not require a
frame rate. Temporal depth inference is run separately for each video.

Depth output retains flattened `depth_data` and the scalar `fps` of the first
nonempty sequence. `video_fps` gives one value per group (`-1` for empty groups).
`video_indices` and `frame_indices` map every depth map to its group and local
frame index. Frame-only inputs additionally export the ordered `frame_paths`.
Visualization and point-cloud files are placed in separate per-sequence
directories to prevent different inputs from overwriting each other.

## Calibration to reconstruction 标定与手部重建

Use the current operator names, rather than the retired
`video_camera_calibration_static_moge_mapper`:

```yaml
- video_extract_frames_mapper:
    frame_sampling_method: uniform
    frame_num: 12
    frame_dir: /path/to/frames
    legacy_split_by_text_token: false
- video_camera_calibration_moge_mapper:
    tag_field_name: camera_calibration
    output_points: false
    output_depth: false
    output_mask: false
- video_hand_reconstruction_hawor_mapper:
    camera_calibration_field: camera_calibration
```

For pre-extracted frames, start at the calibration operator. Both calibration
and reconstruction use the supplied frame order. HaWoR requires one
calibration entry per video group and accepts either one shared static FoV
or one FoV per input frame. MANO assets and model dependencies must be prepared
as documented by the corresponding operator.

## Regression verification 回归验证

The offline suite mocks predictions while exercising real image decoding,
group boundaries, empty inputs, frame-to-result mapping, and a real
MP4-to-frames-to-dataset-to-JSONL path:

```bash
PYTHONPATH=. python tests/run.py --mode regression --tag standalone \
  --test_dir tests/ops/mapper --pattern test_video_frame_sequence_contract.py
```

These tests verify data-flow contracts; they do not establish real checkpoint
accuracy or full CUDA model compatibility.

## Real Ray and CUDA checks 真实 Ray 与 CUDA 验证

The opt-in Ray suite starts real task and actor workers for all six operators.
It mocks model predictions, while checking transport, empty-first samples,
group positions, and frame/result correspondence:

```bash
DJ_RUN_RAY_TESTS=1 PYTHONPATH=.:tests/ops/mapper python -m pytest -q \
  tests/ops/mapper/test_video_frame_sequence_ray.py
```

The opt-in GPU suite uses real checkpoints and requires CUDA. Set
`DJ_GPU_MODEL_DIR` to a directory containing `apt36k.pth`,
`yoloe-26x-seg.pt`, `final.pth.tar`, `video_depth_anything_vits.pth`,
`yolox_l.onnx`, `dw-ll_ucoco_384.onnx`, and `moge-model.pt` (MoGe-2).
Prepare the model dependencies described by each operator, and point
`DATA_JUICER_ASSETS_CACHE` to the directory containing the official
ViTPose and Video-Depth-Anything repositories.

Run models in separate processes to release GPU resources between cases:

```bash
export DJ_RUN_GPU_TESTS=1
export DJ_GPU_MODEL_DIR=/path/to/checkpoints
export PYTHONPATH=.:tests/ops/mapper
for model in animal face body depth moge; do
  python -m pytest -q \
    tests/ops/mapper/test_video_frame_sequence_gpu.py::VideoFrameSequenceGpuTest::test_$model
done
DJ_RUN_RAY_TESTS=1 python -m pytest -q \
  tests/ops/mapper/test_video_frame_sequence_gpu.py::VideoFrameSequenceGpuTest::test_ray_gpu_actor
```

These checks assert nonempty real predictions, finite numeric outputs,
CUDA model/provider selection, frame correspondence, and execution in a
separate GPU Ray actor. They do not benchmark model accuracy.
Legacy ViTPose requires a compatible NumPy 1.26/MMCV stack; use a matching
Arrow/datasets version (for example Arrow 22 and datasets 4.7).
MoGe-2 was verified with official commit
`b942f00bdc2a2a23ebb474fbe034d487e6dcceec` and its declared utils3d dependency.

Full HaWoR checkpoint inference additionally requires licensed MANO assets.
The offline/Ray contract tests cover HaWoR data flow, but do not replace
that full reconstruction check.
