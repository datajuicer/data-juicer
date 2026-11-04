# video_active_speaker_detect_mapper

Detect active speakers in a video by analyzing visual face tracks and audio signals, including consistency checks for gender and age. This operator uses the Light-ASD model to determine which tracked persons are actively speaking. It supports optional consistency detection using prior metadata from other operators. It must be operated after video_human_tracks_extraction_mapper and video_tagging_from_audio_mapper.

通过分析视觉人脸轨迹和音频信号来检测视频中的活跃说话者，包括性别和年龄的一致性检查。此算子使用 Light-ASD 模型来确定哪些被跟踪的人物正在主动说话。它支持使用其他算子的先验元数据进行可选的一致性检测。它必须在 video_human_tracks_extraction_mapper 和 video_tagging_from_audio_mapper 之后运行。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `temp_save_path` | <class 'str'> | `'./temp_path'` | path for temporary file storage. |
| `Light_ASD_model_path` | <class 'str'> | `'./thirdparty/humanvbench_models/Light-ASD/weight/finetuning_TalkSet.model'` | path to the Light-ASD model weights. |
| `active_threshold` | <class 'int'> | `15` | threshold for active speaker detection. Higher values are stricter. |
| `active_speaker_flag` | <class 'str'> | `'active_speaker_flag'` | field name to store the active speaker flags. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_active_speaker_detect_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_active_speaker_detect_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
