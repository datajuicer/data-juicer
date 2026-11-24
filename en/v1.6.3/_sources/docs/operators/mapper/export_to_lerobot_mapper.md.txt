# export_to_lerobot_mapper

Export processed video data to LeRobot v2.0 dataset format (LIBERO-style).

Designed for Ray distributed execution: each actor writes files
independently using UUID-based names (no cross-process coordination).
After all actors finish, call `finalize_dataset()` once to assign
sequential episode indices, rename files, and generate metadata.

Processing phase (parallel, per actor):
  staging/
  ├── data/{uuid}.parquet
  ├── videos/{uuid}.mp4
  └── meta/episodes_{uuid}.jsonl

After finalize_dataset() (single-threaded):
  dataset_dir/
  ├── data/chunk-{NNN}/episode_XXXXXX.parquet
  ├── videos/chunk-{NNN}/observation.images.image/episode_XXXXXX.mp4
  └── meta/
      ├── info.json
      ├── tasks.jsonl
      ├── episodes.jsonl
      └── modality.json

将处理后的视频数据导出为 LeRobot v2.0 数据集格式（LIBERO 风格）。

专为 Ray 分布式执行设计：每个 actor 使用基于 UUID 的名称独立写入文件（无需跨进程协调）。所有 actor 完成后，调用一次 `finalize_dataset()` 以分配连续的 episode 索引、重命名文件并生成元数据。

处理阶段（并行，每个 actor）：
  staging/
  ├── data/{uuid}.parquet
  ├── videos/{uuid}.mp4
  └── meta/episodes_{uuid}.jsonl

调用 finalize_dataset() 后（单线程）：
  dataset_dir/
  ├── data/chunk-{NNN}/episode_XXXXXX.parquet
  ├── videos/chunk-{NNN}/observation.images.image/episode_XXXXXX.mp4
  └── meta/
      ├── info.json
      ├── tasks.jsonl
      ├── episodes.jsonl
      └── modality.json

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `output_dir` | <class 'str'> | `'./lerobot_output'` | Root directory for the LeRobot dataset output. |
| `hand_action_field` | <class 'str'> | `'hand_action_tags'` | Meta field with action/state data. Used in whole-video mode (segment_field=None). |
| `fps` | <class 'int'> | `10` | Frames per second for the dataset. |
| `robot_type` | <class 'str'> | `'egodex_hand'` | Robot type identifier for info.json. |
| `chunks_size` | <class 'int'> | `1000` | Max episodes per chunk directory (default 1000). |
| `segment_field` | <class 'str'> | `None` | Meta field storing atomic action segments. When set, each segment becomes a separate episode with its own caption as task description. When None (default), falls back to whole-video export via hand_action_field. |
| `frame_field` | <class 'str'> | `'video_frames'` | Sample field with extracted frame image paths. Used in segment mode to create per-segment videos. |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/export_to_lerobot_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_export_to_lerobot_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)