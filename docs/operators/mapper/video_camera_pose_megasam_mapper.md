# video_camera_pose_megasam_mapper

Extract camera poses by leveraging MegaSaM and MoGe-2.

利用 MegaSaM 和 MoGe-2 提取相机位姿。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tag_field_name` | <class 'str'> | `'video_camera_pose_tags'` | The field name to store the tags. It's "video_camera_pose_tags" in default. |
| `frame_field` | <class 'str'> | `'video_frames'` | The field name where the video frames are stored. |
| `camera_calibration_field` | <class 'str'> | `'camera_calibration'` | The field name where the camera calibration info is stored. |
| `max_frames` | <class 'int'> | `1000` | Maximum number of frames to save. |
| `droid_buffer` | <class 'int'> | `1024` | DROID SLAM pre-allocated frame buffer size. Controls GPU memory usage — each buffer slot pre-allocates correlation volumes on GPU. Default 1024, sufficient for clips up to ~100 frames. Reduce for shorter clips to save VRAM, increase for longer videos. |
| `save_dir` | <class 'str'> | `None` | Directory to save large numpy arrays (depth, cam_c2w) as .npy files instead of storing them inline. When set, tag_dict stores file paths (strings) instead of numpy arrays, which avoids memory limit. |
| `use_prepare_env` | <class 'bool'> | `False` | Whether to prepare the environment. |
| `args` |  | `` | extra args |
| `kwargs` |  | `` | extra args |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_camera_pose_megasam_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_camera_pose_megasam_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)