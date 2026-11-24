# video_captioning_from_frames_mapper

Generates video captions from sampled frames using an image-to-text model. Captions from different frames are concatenated into a single string.

- Uses a Hugging Face image-to-text model to generate captions for sampled video frames.
- Supports different frame sampling methods: 'all_keyframes' or 'uniform'.
- Can apply horizontal and vertical flips to the frames before captioning.
- Offers multiple strategies for retaining generated captions: 'random_any', 'similar_one_simhash', or 'all'.
- Optionally keeps the original sample in the final dataset.
- Allows setting a global prompt or per-sample prompts to guide caption generation.
- Generates a specified number of candidate captions per video, which can be reduced based on the selected retention strategy.
- The number of output samples depends on the retention strategy and whether original samples are kept.

使用图像到文本模型从采样的帧中生成视频字幕。来自不同帧的字幕被连接成一个字符串。

- 使用Hugging Face图像到文本模型为采样的视频帧生成字幕。
- 支持不同的帧采样方法：'all_keyframes'或'uniform'。
- 可以在字幕前对帧进行水平和垂直翻转。
- 提供多种保留生成字幕的策略：'random_any'、'similar_one_simhash'或'all'。
- 可选地在最终数据集中保留原始样本。
- 允许设置全局提示或每个样本的提示来指导字幕生成。
- 为每个视频生成指定数量的候选字幕，可以根据选定的保留策略进行减少。
- 输出样本的数量取决于保留策略以及是否保留原始样本。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_img2seq` | <class 'str'> | `'Salesforce/blip2-opt-2.7b'` | model name on huggingface to generate caption |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `caption_num` | typing.Annotated[int, Gt(gt=0)] | `1` | how many candidate captions to generate for each video |
| `keep_candidate_mode` | <class 'str'> | `'random_any'` | retain strategy for the generated $caption_num$ candidates.      'random_any': Retain the random one from generated captions      'similar_one_simhash': Retain the generated one that is most         similar to the original caption      'all': Retain all generated captions by concatenation  Note:     This is a batched_OP, whose input and output type are     both list. Suppose there are $N$ list of input samples, whose batch     size is $b$, and denote caption_num as $M$.     The number of total samples after generation is $2Nb$ when     keep_original_sample is True and $Nb$ when keep_original_sample is     False. For 'random_any' and 'similar_one_simhash' mode,     it's $(1+M)Nb$ for 'all' mode when keep_original_sample is True     and $MNb$ when keep_original_sample is False. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated captions in the final datasets and the original captions will be removed. It's True in default. |
| `prompt` | typing.Optional[str] | `None` | a string prompt to guide the generation of image-to-text model for all samples globally. It's None in default, which means no prompt provided. |
| `prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts for each sample. It's used for set different prompts for different samples. If it's none, use prompt in parameter "prompt". It's None in default. |
| `frame_field` | typing.Optional[str] | `None` | the field name of video frames to generate caption. If frame_field is None, extract frames from the video field. |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame videos from the videos. Should be one of ["all_keyframes", "uniform"]. Only works when "frame_field" is none. The former one extracts all key frames (the number of which depends on the duration of the video) and the latter one extract specified number of frames uniformly from the video. Default: "all_keyframes". |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from the video frames. Only works when "frame_sampling_method" is "uniform" or "frame_field" is given. If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. |
| `horizontal_flip` | <class 'bool'> | `False` | flip frame video horizontally (left to right). |
| `vertical_flip` | <class 'bool'> | `False` | flip frame video vertically (top to bottom). |
| `text_update_strategy` | <class 'str'> | `'rewrite'` | strategy to update the text field after caption generation. Can be one of ['keep_origin', 'rewrite']. 'keep_origin': keep the original text unchanged. 'rewrite': rewrite the text field with the generated captions concated by special tokens. |
| `caption_field` | typing.Optional[str] | `None` | the field name to save the generated captions. |
| `legacy_split_by_text_token` | <class 'bool'> | `True` | Whether to split by special tokens (e.g. <__dj__video>) in the text field and read videos in order, or use the 'videos' or 'frames' field directly. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_frames_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_frames_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)