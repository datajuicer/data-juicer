# video_hand_reconstruction_hawor_mapper

Use HaWoR and MoGe-2 for hand reconstruction.

使用 HaWoR 和 MoGe-2 进行手部重建。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hawor_model_path` | <class 'str'> | `'hawor.ckpt'` | The path to 'hawor.ckpt'. for the HaWoR model. |
| `hawor_config_path` | <class 'str'> | `'model_config.yaml'` | The path to 'model_config.yaml' for the HaWoR model. |
| `hawor_detector_path` | <class 'str'> | `'detector.pt'` | The path to 'detector.pt' for the HaWoR model. |
| `mano_right_path` | <class 'str'> | `'path_to_mano_right_pkl'` | The path to 'MANO_RIGHT.pkl'. Users need to download this file from https://mano.is.tue.mpg.de/ and comply with the MANO license. |
| `mano_left_path` | <class 'str'> | `'path_to_mano_left_pkl'` | The path to 'MANO_LEFT.pkl'. Users need to download this file from https://mano.is.tue.mpg.de/ and comply with the MANO license. Used for accurate left-hand wrist offset computation (with shapedirs bug-fix). |
| `frame_field` | <class 'str'> | `'video_frames'` | The field name where the video frames are stored. |
| `camera_calibration_field` | <class 'str'> | `'camera_calibration'` | The field name where the camera calibration info is stored. |
| `tag_field_name` | <class 'str'> | `'hand_reconstruction_hawor_tags'` | The field name to store the tags. It's "hand_reconstruction_hawor_tags" in default. |
| `thresh` | <class 'float'> | `0.2` | The confidence threshold for hand detection. Default is 0.2. |
| `args` |  | `` | extra args |
| `kwargs` |  | `` | extra args |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_hand_reconstruction_hawor_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_hand_reconstruction_hawor_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)