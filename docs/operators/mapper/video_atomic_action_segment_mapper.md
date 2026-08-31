# video_atomic_action_segment_mapper

Segment a unified hand trajectory into atomic action clips.

Implements the algorithm from paper https://arxiv.org/pdf/2510.21571:

    "we detect speed minima of the 3D hand wrists in the world space
    and use them as cutting points.  We smooth the hand trajectory and
    select points that are local speed minima within a fixed window
    centered on each point."

The operator reads the merged hand_action_tags (output of
``VideoClipReassemblyMapper``) and produces a list of segments.
Each segment contains the start and end frame indices, plus sliced
states / actions / joints for that segment.

Segmentation is applied **independently** for left and right hands.
A frame is a cutting point if it is a speed local minimum within a
window of ``min_window`` frames on each side.

Output field (``segment_field``) structure::

    [
        {
            "hand_type": "right",
            "segment_id": 0,
            "start_frame": 10,
            "end_frame": 45,
            "states": [...],
            "actions": [...],
            "valid_frame_ids": [...],
            "joints_world": [...],
        },
        ...
    ]

将统一的手部轨迹分割为原子动作片段。

实现了论文 https://arxiv.org/pdf/2510.21571 中的算法：

    "we detect speed minima of the 3D hand wrists in the world space
    and use them as cutting points.  We smooth the hand trajectory and
    select points that are local speed minima within a fixed window
    centered on each point."

该算子读取合并后的 hand_action_tags（``VideoClipReassemblyMapper`` 的输出）并生成片段列表。每个片段包含起始和结束帧索引，以及该片段对应的切片 states / actions / joints。

分割操作对左手和右手**独立**应用。如果一帧在两侧各 ``min_window`` 帧的窗口内是速度局部极小值，则该帧为切割点。

输出字段（``segment_field``）结构::

    [
        {
            "hand_type": "right",
            "segment_id": 0,
            "start_frame": 10,
            "end_frame": 45,
            "states": [...],
            "actions": [...],
            "valid_frame_ids": [...],
            "joints_world": [...],
        },
        ...
    ]

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hand_action_field` | <class 'str'> | `'hand_action_tags'` | Meta field storing merged hand action results (output of VideoClipReassemblyMapper). |
| `segment_field` | <class 'str'> | `'atomic_action_segments'` | Output meta field for atomic segments. |
| `speed_smooth_window` | <class 'int'> | `5` | Window size for Savitzky-Golay smoothing of the speed signal before minima detection. Must be odd. |
| `min_window` | <class 'int'> | `15` | Half-window size for local minima detection. A frame is a local minimum only if it is the minimum within ``[t - min_window, t + min_window]``. Larger values → fewer, longer segments. |
| `min_segment_frames` | <class 'int'> | `8` | Minimum frames per segment. Segments shorter than this are merged with neighbors. |
| `max_segment_frames` | <class 'int'> | `300` | Maximum frames per segment. Segments longer than this are forcibly split at the deepest speed minimum. |
| `hand_type` | <class 'str'> | `'both'` | Which hand(s) to segment: 'left', 'right', or 'both'. |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_atomic_action_segment_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_atomic_action_segment_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)