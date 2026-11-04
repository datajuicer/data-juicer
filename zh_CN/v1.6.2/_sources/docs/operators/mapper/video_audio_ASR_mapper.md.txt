# video_audio_ASR_mapper

Perform automatic speech recognition (ASR) on video audio streams using the SenseVoiceSmall model. This operator extracts audio from videos and transcribes speech content. It must be operated after video_tagging_from_audio_mapper, as it only processes videos tagged as containing speech.

使用 SenseVoiceSmall 模型对视频音频流进行自动语音识别（ASR）。此算子从视频中提取音频并转录语音内容。它必须在 video_tagging_from_audio_mapper 之后运行，因为仅处理被标记为包含语音的视频。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `model_dir_ASR` | <class 'str'> | `'FunAudioLLM/SenseVoiceSmall'` | path to the SenseVoiceSmall ASR model. |
| `speech_ASR` | <class 'str'> | `'speech_ASR'` | field name to store the ASR results. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_audio_ASR_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_audio_ASR_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)
