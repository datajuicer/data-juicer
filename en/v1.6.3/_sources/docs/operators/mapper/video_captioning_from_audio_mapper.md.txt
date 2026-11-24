# video_captioning_from_audio_mapper

Mapper to caption a video according to its audio streams based on Qwen-Audio model.

基于Qwen-Audio模型根据音频流为视频添加字幕的映射器。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only captioned sample in the final datasets and the original sample will be removed. It's True in default. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_audio_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_audio_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)