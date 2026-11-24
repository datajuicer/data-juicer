# video_clip_reassembly_mapper

Reassemble hand-action results from overlapping video clips.

When long videos are chopped into overlapping clips (e.g. 5 s with 2 s
overlap via ``VideoSplitByDurationMapper``), each clip is processed
independently through the 3-D motion labelling pipeline.  This operator
merges the per-clip results back into **one unified result** per original
video, including:

* ``hand_action_tags`` — states, actions, valid_frame_ids, joints
* ``video_camera_pose_tags`` — ``cam_c2w`` array
* ``hand_reconstruction_hawor_tags`` — frame_ids converted to global
* ``video_frames`` — per-clip frame path lists merged into one global list
* ``camera_calibration_moge_tags`` — per-clip depth/intrinsics merged
* ``clips`` — replaced with the original video path

Clip global offsets are determined automatically by **pixel-matching**
overlapping frames between consecutive clips, rather than assuming an
ideal step size.  This handles ffmpeg keyframe-alignment drift that
causes actual clip boundaries to differ from the nominal
``(split_duration - overlap_duration) * fps`` calculation.

Reference (paper §3.1):
    "To enhance efficiency, we chop long videos into overlapping
    20-second clips in this stage and recompose their results."

从重叠的视频片段中重组手部动作结果。

当长视频被切分为重叠的片段时（例如通过 ``VideoSplitByDurationMapper`` 切分为 5 秒片段，重叠 2 秒），每个片段会通过 3D 动作标注管线进行独立处理。此算子将各片段的结果合并回原始视频的**单一统一结果**中，包括：

* ``hand_action_tags`` — states、actions、valid_frame_ids、joints
* ``video_camera_pose_tags`` — ``cam_c2w`` 数组
* ``hand_reconstruction_hawor_tags`` — 转换为全局的 frame_ids
* ``video_frames`` — 各片段的帧路径列表合并为一个全局列表
* ``camera_calibration_moge_tags`` — 各片段的深度/内参合并
* ``clips`` — 替换为原始视频路径

片段的全局偏移量通过**像素匹配**连续片段之间的重叠帧来自动确定，而不是假设理想的步长。这可以处理 ffmpeg 关键帧对齐漂移问题，该问题会导致实际片段边界与标称的 ``(split_duration - overlap_duration) * fps`` 计算结果存在差异。

参考文献（论文 §3.1）：
    “为提高效率，我们在此阶段将长视频切分为重叠的 20 秒片段，并重组其结果。”

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hand_action_field` | <class 'str'> | `'hand_action_tags'` |  |
| `camera_pose_field` | <class 'str'> | `'video_camera_pose_tags'` |  |
| `hand_reconstruction_field` | <class 'str'> | `'hand_reconstruction_hawor_tags'` |  |
| `frame_field` | <class 'str'> | `'video_frames'` |  |
| `moge_field` | <class 'str'> | `'camera_calibration_moge_tags'` |  |
| `clip_field` | <class 'str'> | `'clips'` |  |
| `video_key` | <class 'str'> | `'videos'` |  |
| `split_duration` | <class 'float'> | `None` |  |
| `overlap_duration` | <class 'float'> | `None` |  |
| `fps` | <class 'float'> | `None` |  |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_clip_reassembly_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_clip_reassembly_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)