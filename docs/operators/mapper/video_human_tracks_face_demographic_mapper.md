# video_human_tracks_face_demographic_mapper

Detect face demographics (age, gender, race) for each tracked person in a video using DeepFace. This operator processes the human track data produced by video_human_tracks_extraction_mapper and analyzes the facial attributes of each detected person. It must be operated after video_human_tracks_extraction_mapper.

使用 DeepFace 检测视频中每个跟踪人物的人脸属性（年龄、性别、种族）。此算子处理由 video_human_tracks_extraction_mapper 生成的人体轨迹数据，并分析每个检测到的人物的面部属性。它必须在 video_human_tracks_extraction_mapper 之后运行。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `original_data_save_path` | <class 'str'> | `''` | path to save intermediate detection results. |
| `detect_interval` | <class 'int'> | `5` | interval (in frames) at which face demographic detection is performed. |
| `tag_field_name` | <class 'str'> | `'video_facetrack_attribute_demographic'` | field name to store the demographic results. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_human_tracks_face_demographic_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_human_tracks_face_demographic_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
