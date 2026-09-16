# video_captioning_from_video_mapper

Generates video captions using a Hugging Face video-to-text model and sampled video frames.

This operator processes video samples to generate captions based on the provided video frames. It uses a Hugging Face video-to-text model, such as 'kpyu/video-blip-opt-2.7b-ego4d', to generate multiple caption candidates for each video. The number of generated captions and the strategy to keep or filter these candidates can be configured. The operator supports different frame sampling methods, including extracting all keyframes or uniformly sampling a specified number of frames. Additionally, it allows for horizontal and vertical flipping of the frames. The final output can include both the original sample and the generated captions, depending on the configuration.

使用 Hugging Face 视频转文本模型和采样的视频帧生成视频字幕。

该算子处理视频样本，基于提供的视频帧生成字幕。它使用 Hugging Face 视频转文本模型（如 'kpyu/video-blip-opt-2.7b-ego4d'）为每个视频生成多个字幕候选。可以配置生成的字幕数量和保留或筛选这些候选字幕的策略。该算子支持不同的帧采样方法，包括提取所有关键帧或均匀采样指定数量的帧。此外，还允许对帧进行水平和垂直翻转。最终输出可以包括原始样本和生成的字幕，具体取决于配置。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_video_blip` | <class 'str'> | `'kpyu/video-blip-opt-2.7b-ego4d'` | video-blip model name on huggingface |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `caption_num` | typing.Annotated[int, Gt(gt=0)] | `1` | how many candidate captions to generate |
| `keep_candidate_mode` | <class 'str'> | `'random_any'` | retain strategy for the generated |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `prompt` | typing.Optional[str] | `None` | a string prompt to guide the generation of video-blip |
| `prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from |
| `horizontal_flip` | <class 'bool'> | `False` | flip frame video horizontally (left to right). |
| `vertical_flip` | <class 'bool'> | `False` | flip frame video vertically (top to bottom). |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_video_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_video_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)