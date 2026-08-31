# video_camera_calibration_droidcalib_mapper

Extract camera intrinsics from videos using DroidCalib.

**Notice**: This operator will download the DroidCalib component from
GitHub at runtime. This component follows the AGPL-3.0 license, please
be aware for commercial use.

使用 DroidCalib 从视频中提取相机内参。

**注意**：此算子将在运行时从 GitHub 下载 DroidCalib 组件。该组件遵循 AGPL-3.0 许可证，商业使用请注意相关合规要求。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `weights_path` | typing.Optional[str] | `None` | Path to the model weights. |
| `image_size` | <class 'list'> | `[384, 512]` | Target image size [height, width]. |
| `stride` | <class 'int'> | `2` | Frame stride. |
| `max_frames` | <class 'int'> | `300` | Maximum number of frames to process. |
| `buffer` | <class 'int'> | `1024` | Buffer size for Droid. |
| `beta` | <class 'float'> | `0.3` | Weight for translation / rotation components of flow. |
| `filter_thresh` | <class 'float'> | `2.4` | Motion threshold before considering new keyframe. |
| `warmup` | <class 'int'> | `8` | Number of warmup frames. |
| `keyframe_thresh` | <class 'float'> | `4.0` | Threshold to create a new keyframe. |
| `frontend_thresh` | <class 'float'> | `16.0` | Add edges between frames within this distance. |
| `frontend_window` | <class 'int'> | `25` | Frontend optimization window. |
| `frontend_radius` | <class 'int'> | `2` | Force edges between frames within radius. |
| `frontend_nms` | <class 'int'> | `1` | Non-maximal suppression of edges. |
| `backend_thresh` | <class 'float'> | `22.0` | Backend threshold. |
| `backend_radius` | <class 'int'> | `2` | Backend radius. |
| `backend_nms` | <class 'int'> | `3` | Backend NMS. |
| `upsample` | <class 'bool'> | `False` | Whether to upsample. |
| `disable_vis` | <class 'bool'> | `True` | Whether to disable visualization. |
| `verbose` | <class 'bool'> | `False` |  |
| `tag_field_name` | <class 'str'> | `'camera_calibration_droidcalib_tags'` |  |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_camera_calibration_droidcalib_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_camera_calibration_droidcalib_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)