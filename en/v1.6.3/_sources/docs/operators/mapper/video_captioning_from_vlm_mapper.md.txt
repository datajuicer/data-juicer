# video_captioning_from_vlm_mapper

Generates video captions using a VLM that accepts videos as inputs.

This operator processes video samples to generate captions based on the provided video. It uses a VLM model that accept videos as inputs, such as 'Qwen/Qwen3-VL-8B-Instruct', to generate multiple caption candidates for each video. The number of generated captions and the strategy to keep or filter these candidates can be configured. The final output can include both the original sample and the generated captions, depending on the configuration.

使用接受视频作为输入的视觉语言模型（VLM）生成视频字幕。

此算子处理视频样本，根据提供的视频生成字幕。它使用支持视频输入的VLM模型（例如 'Qwen/Qwen3-VL-8B-Instruct'），为每个视频生成多个字幕候选。生成字幕的数量以及保留或过滤这些候选的策略均可配置。最终输出可根据配置包含原始样本和生成的字幕。

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen3-VL-8B-Instruct'` | VLM model name on huggingface to generate caption |
| `enable_vllm` | <class 'bool'> | `False` | If true, use VLLM for loading hugging face or local llm. |
| `caption_num` | typing.Annotated[int, Gt(gt=0)] | `1` | how many candidate captions to generate for each video |
| `keep_candidate_mode` | <class 'str'> | `'random_any'` | retain strategy for the generated $caption_num$ candidates.      'random_any': Retain the random one from generated captions      'similar_one_simhash': Retain the generated one that is most         similar to the original caption      'all': Retain all generated captions by concatenation  Note:     This is a batched_OP, whose input and output type are     both list. Suppose there are $N$ list of input samples, whose batch     size is $b$, and denote caption_num as $M$.     The number of total samples after generation is $2Nb$ when     keep_original_sample is True and $Nb$ when keep_original_sample is     False. For 'random_any' and 'similar_one_simhash' mode,     it's $(1+M)Nb$ for 'all' mode when keep_original_sample is True     and $MNb$ when keep_original_sample is False. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated captions in the final datasets and the original captions will be removed. It's True in default. |
| `prompt` | typing.Optional[str] | `None` | a string prompt to guide the generation of video-blip model for all samples globally. It's None in default, which means using the DEFAULT_PROMPT. |
| `prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts for each sample. It's used for set different prompts for different samples. If it's none, use prompt in parameter "prompt". It's None in default. |
| `model_params` | typing.Dict | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Dict | `None` | Extra parameters passed to the model calling. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra kwargs |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_vlm_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_vlm_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)