# video_camera_pose_mapper

Extract camera poses by leveraging MegaSaM and MoGe-2.

利用MegaSaM和MoGe-2提取相机姿态。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `moge_model_path` | <class 'str'> | `'Ruicheng/moge-2-vitl'` | The path to the Moge-2 model. |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of frames to be extracted uniformly from the video. If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. If "duration" > 0, frame_num is the number of frames per segment. |
| `duration` | <class 'float'> | `0` | The duration of each segment in seconds. If 0, frames are extracted from the entire video. If duration > 0, the video is segmented into multiple segments based on duration, and frames are extracted from each segment. |
| `tag_field_name` | <class 'str'> | `'video_camera_pose_tags'` | The field name to store the tags. It's "video_camera_pose_tags" in default. |
| `frame_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | Output directory to save extracted frames. |
| `if_output_moge_info` | <class 'bool'> | `False` | Whether to save the results from MoGe-2 to an JSON file. |
| `moge_output_info_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | Output directory for saving camera parameters. |
| `if_save_info` | <class 'bool'> | `True` | Whether to save the results to an npz file. |
| `output_info_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | Path for saving the results. |
| `max_frames` | <class 'int'> | `1000` | Maximum number of frames to save. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_camera_pose_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_camera_pose_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)