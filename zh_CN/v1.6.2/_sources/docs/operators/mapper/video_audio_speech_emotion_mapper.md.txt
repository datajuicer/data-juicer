# video_audio_speech_emotion_mapper

Perform speech emotion recognition on video audio streams using the SenseVoiceSmall model. This operator extracts audio from videos tagged as containing speech and recognizes the emotional state of the speaker. It must be operated after video_tagging_from_audio_mapper.

使用 SenseVoiceSmall 模型对视频音频流进行语音情感识别。此算子从被标记为包含语音的视频中提取音频，并识别说话者的情感状态。它必须在 video_tagging_from_audio_mapper 之后运行。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `model_dir_emo` | <class 'str'> | `'FunAudioLLM/SenseVoiceSmall'` | path to the SenseVoiceSmall emotion recognition model. |
| `speech_Emo` | <class 'str'> | `'speech_emotion'` | field name to store the emotion results. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_audio_speech_emotion_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_audio_speech_emotion_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
