# video_human_tracks_extraction_mapper

Extract face and human bounding box tracks from videos. This operator performs multi-stage processing including scene detection, face detection (S3FD), face tracking, and human detection (YOLOv8). It eventually generates synchronized face and human tracks and saves the bbox sequences into pickle files.

从视频中提取人脸和人体边界框轨迹。此算子执行多阶段处理，包括场景检测、人脸检测（S3FD）、人脸跟踪和人体检测（YOLOv8）。最终生成同步的人脸和人体轨迹，并将边界框序列保存为 pickle 文件。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `face_track_bbox_path` | <class 'str'> | `'./HumanVBenchRecipe/dj_human_track'` | path to save the bbox pickle files. |
| `YOLOv8_human_model_path` | <class 'str'> | `'./thirdparty/humanvbench_models/YOLOv8_human/weights/best.pt'` | path to the YOLOv8 human detection model. |
| `face_detect_S3FD_model_path` | <class 'str'> | `'./thirdparty/humanvbench_models/Light-ASD/model/faceDetector/s3fd/sfd_face.pth'` | path to the S3FD face detection model. |
| `tag_field_name_human_track_path` | <class 'str'> | `'human_track_data_path'` | field name to store the human track data path. |
| `tag_field_name_people_num` | <class 'str'> | `'number_people_in_video'` | field name to store the number of people. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_human_tracks_extraction_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_human_tracks_extraction_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
