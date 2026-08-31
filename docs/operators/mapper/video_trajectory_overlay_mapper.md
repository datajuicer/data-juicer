# video_trajectory_overlay_mapper

Prepare VLM-ready frames by sampling and overlaying hand trajectories.

Implements the visualization step from paper https://arxiv.org/pdf/2510.21571:

    "From each segment, we evenly sample 8 frames and highlight hand
    trajectories on each frame by projecting the world-space trajectory
    of the hand palm from the current frame to the end of the clip."

For each atomic action segment (output of
``VideoAtomicActionSegmentMapper``), this operator:

1. Evenly samples ``n_sample_frames`` frames from the segment.
2. For each sampled frame, projects the **future** world-space wrist
   trajectory (from the current frame to the end of the segment) onto
   the image using camera intrinsics and cam_c2w.
3. Draws the trajectory as a colored line with a dot at the current
   wrist position.
4. Saves the overlay images and stores their paths in the segment.

The output is written back into each segment dict under
``"overlay_frames"``, ready to be consumed by the VLM captioning
operator.

通过采样和叠加手部轨迹来准备适用于 VLM 的帧。

实现论文 https://arxiv.org/pdf/2510.21571 中的可视化步骤：

    “从每个片段中，我们均匀采样 8 帧，并通过将手掌的世界坐标系轨迹从当前帧投影到片段末尾，在每帧上突出显示手部轨迹。”

对于每个原子动作片段（``VideoAtomicActionSegmentMapper`` 的输出），此算子：

1. 从片段中均匀采样 ``n_sample_frames`` 帧。
2. 对于每个采样帧，使用相机内参和 cam_c2w 将**未来**的世界坐标系手腕轨迹（从当前帧到片段末尾）投影到图像上。
3. 将轨迹绘制为彩色线条，并在当前手腕位置绘制一个圆点。
4. 保存叠加图像并将其路径存储在片段中。

输出被写回到每个片段字典的 ``"overlay_frames"`` 键下，以供 VLM 描述生成算子使用。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `segment_field` | <class 'str'> | `'atomic_action_segments'` | Meta field storing atomic action segments. |
| `camera_pose_field` | <class 'str'> | `'video_camera_pose_tags'` | Meta field storing camera pose (cam_c2w). |
| `moge_field` | <class 'str'> | `'camera_calibration_moge_tags'` | Meta field storing MoGe calibration (for fov_x). |
| `frame_field` | <class 'str'> | `'video_frames'` | Field storing frame image paths. |
| `save_dir` | <class 'str'> | `None` | Directory to save overlay images.  If None, uses a temp directory derived from the first frame path. |
| `n_sample_frames` | <class 'int'> | `8` | Number of frames to evenly sample from each segment. |
| `palm_joint_index` | <class 'int'> | `9` | MANO joint index for the palm position. Default 9 = middle finger MCP (palm center proxy). Joint 0 = wrist root. |
| `dot_radius` | <class 'int'> | `10` | Radius of the dot at the current wrist position. |
| `line_thickness` | <class 'int'> | `4` | Thickness of the trajectory line. |
| `trajectory_alpha` | <class 'float'> | `0.7` | Alpha blending for the trajectory overlay. |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_trajectory_overlay_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_trajectory_overlay_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)