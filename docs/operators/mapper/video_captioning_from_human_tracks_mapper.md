# video_captioning_from_human_tracks_mapper

Generate per-person captions for each tracked human in a video using a video-to-text model (VideoLLaMA3). This operator crops video segments around each tracked person and generates a description of the person's appearance, as well as determining whether the person is a child. It must be operated after video_human_tracks_extraction_mapper.

使用视频转文本模型（VideoLLaMA3）为视频中每个跟踪的人物生成个体描述。此算子根据每个跟踪人物裁剪视频片段，生成人物外观描述，并判断该人物是否为儿童。它必须在 video_human_tracks_extraction_mapper 之后运行。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `human_track_query` | <class 'str'> | `'Descibe the person's apperance. Less than 80 words. '` | prompt for describing the person's appearance. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `video_describe_model_path` | <class 'str'> | `'DAMO-NLP-SG/VideoLLaMA3-7B'` | path to the VideoLLaMA3 model. |
| `temp_video_path` | <class 'str'> | `None` | path for temporary video storage. |
| `tag_field_name_track_video_caption` | <class 'str'> | `'track_video_caption'` | field name to store the track captions. |
| `tag_field_name_video_track_is_child` | <class 'str'> | `'video_track_is_child'` | field name to store the child flag. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_human_tracks_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_human_tracks_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
