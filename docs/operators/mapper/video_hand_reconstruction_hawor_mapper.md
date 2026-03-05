# video_hand_reconstruction_hawor_mapper

Use HaWoR and MoGe-2 for hand reconstruction.

使用HaWoR和MoGe-2进行手部重建。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hawor_model_path` | <class 'str'> | `'hawor.ckpt'` | The path to 'hawor.ckpt'. for the HaWoR model. |
| `hawor_config_path` | <class 'str'> | `'model_config.yaml'` | The path to 'model_config.yaml' for the HaWoR model. |
| `hawor_detector_path` | <class 'str'> | `'detector.pt'` | The path to 'detector.pt' for the HaWoR model. |
| `moge_model_path` | <class 'str'> | `'Ruicheng/moge-2-vitl'` | The path to the Moge-2 model. |
| `mano_right_path` | <class 'str'> | `'path_to_mano_right_pkl'` | The path to 'MANO_RIGHT.pkl'. Users need to download this file from https://mano.is.tue.mpg.de/ and comply with the MANO license. |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of frames to be extracted uniformly from the video. If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. If "duration" > 0, frame_num is the number of frames per segment. |
| `duration` | <class 'float'> | `0` | The duration of each segment in seconds. If 0, frames are extracted from the entire video. If duration > 0, the video is segmented into multiple segments based on duration, and frames are extracted from each segment. |
| `thresh` | <class 'float'> | `0.2` | Confidence threshold for hand detection. |
| `tag_field_name` | <class 'str'> | `'hand_reconstruction_hawor_tags'` | The field name to store the tags. It's "hand_reconstruction_hawor_tags" in default. |
| `frame_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | Output directory to save extracted frames. |
| `if_output_moge_info` | <class 'bool'> | `False` | Whether to save the results from MoGe-2 to an JSON file. |
| `moge_output_info_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | Output directory for saving camera parameters. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_hand_reconstruction_hawor_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_hand_reconstruction_hawor_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)