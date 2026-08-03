# video_captioning_face_attribute_emotion_mapper

Generate facial attribute and emotion descriptions for each person tracked in a video using a video-to-text model (VideoLLaMA3). This operator crops face-centric video segments and generates detailed descriptions of facial expressions and emotions. It must be operated after video_human_tracks_extraction_mapper.

使用视频转文本模型（VideoLLaMA3）为视频中每个跟踪的人物生成面部属性和情感描述。此算子裁剪以面部为中心的视频片段，并生成面部表情和情感的详细描述。它必须在 video_human_tracks_extraction_mapper 之后运行。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `face_track_query` | <class 'str'> | `'Please describe the person's facial expression, tell me the person's emotion through the video, ...'` | prompt for describing facial attributes and emotions. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `cropping_face_video_temp_path` | <class 'str'> | `'./temp_video_path'` | path for temporary face-cropped video storage. |
| `video_describe_model_path` | <class 'str'> | `'DAMO-NLP-SG/VideoLLaMA3-7B'` | path to the VideoLLaMA3 model. |
| `video_facetrack_attribute_emotion` | <class 'str'> | `'video_facetrack_attribute_emotion'` | field name to store the face attribute emotion results. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_face_attribute_emotion_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_face_attribute_emotion_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
