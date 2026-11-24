# DROID episode perception workflow / DROID episode 感知流程

The six `robot_*` mappers operate on one DROID episode per sample. They write
large predictions to files and keep paths and small scalar fields in the dataset.
They do not generate VQA questions or answers by themselves.

六个算子以一个 DROID episode 为一条样本。大体积预测结果保存到文件中，
数据集保留路径和少量字段；这些算子本身不生成 VQA 问答。

## Input sample / 输入样本

```json
{
  "source": "droid",
  "episode_id": "episode-7",
  "episode_index": 7,
  "metadata_path": "/data/episode-7/metadata.parquet",
  "state_path": "/data/episode-7/states.parquet",
  "raw_dir": "/data/episode-7/cameras",
  "primary_view": "left"
}
```

- `source` must be `droid`. `episode_id` must be a single safe filename component.
- `metadata_path` is a Parquet file whose first row contains the instruction.
  The shared fallback order is `other_information.language_instruction_2`,
  `other_information.language_instruction_3`, `other_information.language_instruction_1`,
  then `annotation.instruction_add`. Empty strings and placeholder values such as
  `-1`, `None`, `nan`, and `[]` are skipped.
- `state_path` must have `frame_index`, `episode_index`, and
  `observation.state.gripper_position`. At least four rows must match the requested
  `episode_index`. State rows are sorted by the original frame index.
- `raw_dir` contains at least two MP4 camera videos. `primary_view` is the stem of
  one filename, for example `left` for `left.mp4`. The cameras must already be
  synchronized; the operators do not synchronize recordings.
- Optional `camera_intrinsics` maps each camera name to its 3x3 pinhole matrix at
  the source image resolution. Depth maps and intrinsics are resized together.

## Execution order and artifacts / 执行顺序与产物

All stages must use the **same `output_dir` and episode identity**. The root is
`<output_dir>/droid/<episode_id>/`. Every worker needs access to that directory,
input Parquet files and camera videos using the same filesystem paths. Copy the
artifact directory with the exported dataset; paths alone do not contain the
predictions. Do not process the same episode concurrently into the same root.

所有阶段必须使用相同的输出目录和 episode 标识；跨进程或节点执行时，输入文件
及产物目录需要以相同路径访问。导出数据时应同时保留这些产物。

| Stage | Required input | Dataset fields / files produced |
| --- | --- | --- |
| `robot_gripper_anchor_mapper` | Metadata, state Parquet and episode index | `instruction`, `instruction_source_column`, `episode_dir`; `ir/phase_anchor.json` |
| `robot_object_tracking_mapper` | Camera videos, metadata, primary view | `object_outputs` (detection JSON followed by tracking JSON); `ir/sampling.json`, decoded images, sampled camera videos and `operator_input/` links |
| `robot_gripper_tracking_mapper` | Primary-view video link created by object tracking | `gripper_output`; `gripper_operator/video_gripper_detection.json` |
| `robot_destination_grounding_mapper` | Object detection JSON and sampled views | `ir/destination.json` and the cached final-view VLM trace |
| `robot_depth_estimation_mapper` (optional) | `ir/sampling.json`; optional camera intrinsics | `depth_ir`; `ir/depth_anything_3.json` and per-frame `.npz` arrays |
| `robot_interaction_geometry_mapper` | State anchor, synchronized views, object/gripper tracks and destination; optional `depth_ir` | `ir/episode_ir.json`, including contact geometry and optional tracked depth |

The reference order is anchor → object tracking → gripper tracking → destination
→ optional depth → geometry. The object and gripper track `frame_index` values
refer to positions in the decoded camera video. Depth records retain both
`sample_index` and `source_frame_index`; state anchors retain original state
frame indices. The default workflow decodes every camera frame. Geometry rejects
unequal cross-view frame-index sequences, fewer than three common track frames,
and the absence of a frame meeting its geometric contact threshold. The contact
score is an overlap heuristic, not a physical contact sensor.

Object localization starts at the first video frame. Gripper tracking starts at
the last frame and propagates backwards, so the primary view needs to show the
target object initially and the gripper in the final frame.

Depth is optional: omit the depth stage and do not supply `depth_ir` to produce
geometry without it. DA3 metric checkpoints report `focal_normalized` depth without
intrinsics and `metres` after focal calibration. Other checkpoints retain the
SDK's metric/relative distinction. Any-view checkpoints require
`inference_batch_size: 0`; monocular checkpoints support smaller batches.

## Models and API configuration / 模型与 API 配置

- Set `VLM_MODEL` and `VLM_URL` (OpenAI-compatible base URL, including `/v1`),
  or pass `model` and `endpoint` to the three VLM-backed mappers. Set
  `VLM_API_KEY` if authentication is required. Requests use Data-Juicer's shared
  API model client; transport failures stop processing instead of writing a
  fabricated annotation.
  Object pointing requests use `{"point_2d": [x, y]}` with both coordinates
  normalized to `[0, 1000]` over the entire input image.
- Set `SAM3_MODEL` or `sam3_model` for object and gripper tracking. A local model
  directory is accepted; remote IDs retain the ModelScope download path. Install
  a Transformers build providing `Sam3Model`, `Sam3Processor`,
  `Sam3TrackerVideoModel` and `Sam3TrackerVideoProcessor`; verify these imports in
  the operator environment instead of assuming the generic extra includes them.
- Install the Depth Anything 3 SDK separately when enabling depth. `DA3_MODEL` or
  `model_path` selects its checkpoint. SAM3 and DA3 use the shared model registry
  and worker-local model cache; heavy models are loaded on first use.
- CPU tests use tiny files and mocked model predictions. They verify data flow
  and serialization, not the accuracy of SAM3, VLM or DA3 predictions.

## Minimal recipe / 最小配方

```yaml
dataset_path: /data/episodes.jsonl
export_path: ./output/episodes.jsonl
np: 1
process:
  - robot_gripper_anchor_mapper:
      output_dir: ./output/episodes
  - robot_object_tracking_mapper:
      output_dir: ./output/episodes
  - robot_gripper_tracking_mapper:
      output_dir: ./output/episodes
  - robot_destination_grounding_mapper:
      output_dir: ./output/episodes
  - robot_depth_estimation_mapper:
      output_dir: ./output/episodes
  - robot_interaction_geometry_mapper:
      output_dir: ./output/episodes
```

The recipe assumes prepared DROID episode samples and configured model/API access.
It does not download the DROID dataset or install optional model SDKs.

Tests: [episode mappers](../../../tests/ops/mapper/test_robot_episode_mappers.py),
[episode utilities](../../../tests/utils/test_robot_utils.py),
[depth utilities](../../../tests/utils/test_robot_depth_utils.py),
and [perception/API contracts](../../../tests/utils/test_robot_perception_utils.py).
