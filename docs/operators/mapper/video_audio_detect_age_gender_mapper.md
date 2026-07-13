# video_audio_detect_age_gender_mapper

Detect age and gender (male, female, child) from video audio signals using a pretrained wav2vec2 model. This operator processes videos tagged as containing speech and classifies the speaker's age and gender from the audio stream. It must be operated after video_tagging_from_audio_mapper.

使用预训练的 wav2vec2 模型从视频音频信号中检测年龄和性别（男性、女性、儿童）。此算子处理被标记为包含语音的视频，并从音频流中分类说话者的年龄和性别。它必须在 video_tagging_from_audio_mapper 之后运行。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_audio_mapper` | <class 'str'> | `'audeering/wav2vec2-large-robust-24-ft-age-gender'` | HuggingFace model for age/gender classification. |
| `tag_field_name` | <class 'str'> | `'audio_speech_attribute'` | field name to store the age/gender results. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_audio_detect_age_gender_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_audio_detect_age_gender_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
