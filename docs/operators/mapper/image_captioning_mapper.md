# image_captioning_mapper

Generates image captions using a Hugging Face model and appends them to samples.

This operator generates captions for images in the input samples using a specified Hugging Face model. It can generate multiple captions per image and apply different strategies to retain the generated captions. The operator supports three retention modes: 'random_any', 'similar_one_simhash', and 'all'. In 'random_any' mode, a random caption is retained. In 'similar_one_simhash' mode, the most similar caption to the original text (based on SimHash) is retained. In 'all' mode, all generated captions are concatenated and retained. The operator can also keep or discard the original sample based on the `keep_original_sample` parameter. If both `prompt` and `prompt_key` are set, the `prompt_key` takes precedence.

使用 Hugging Face 模型生成图像描述并将其附加到样本中。

该算子使用指定的 Hugging Face 模型为输入样本中的图像生成描述。它可以为每张图像生成多个描述，并应用不同的策略来保留生成的描述。该算子支持三种保留模式：'random_any'、'similar_one_simhash' 和 'all'。在 'random_any' 模式下，随机保留一个描述。在 'similar_one_simhash' 模式下，保留与原始文本最相似的描述（基于 SimHash）。在 'all' 模式下，所有生成的描述被连接并保留。该算子还可以根据 `keep_original_sample` 参数保留或丢弃原始样本。如果同时设置了 `prompt` 和 `prompt_key`，则 `prompt_key` 优先。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_img2seq` | <class 'str'> | `'Salesforce/blip2-opt-2.7b'` | model name on huggingface to generate caption |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `caption_num` | typing.Annotated[int, Gt(gt=0)] | `1` | how many candidate captions to generate for each image |
| `keep_candidate_mode` | <class 'str'> | `'random_any'` | retain strategy for the generated $caption_num$ candidates.      'random_any': Retain the random one from generated captions      'similar_one_simhash': Retain the generated one that is most         similar to the original caption      'all': Retain all generated captions by concatenation  Note:     This is a batched_OP, whose input and output type are     both list. Suppose there are $N$ list of input samples, whose batch     size is $b$, and denote caption_num as $M$.     The number of total samples after generation is $2Nb$ when     keep_original_sample is True and $Nb$ when keep_original_sample is     False. For 'random_any' and 'similar_one_simhash' mode,     it's $(1+M)Nb$ for 'all' mode when keep_original_sample is True     and $MNb$ when keep_original_sample is False. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated captions in the final datasets and the original captions will be removed. It's True in default. |
| `prompt` | typing.Optional[str] | `None` | a string prompt to guide the generation of blip2 model for all samples globally. It's None in default, which means no prompt provided. |
| `prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts for each sample. It's used for set different prompts for different samples. If it's none, use prompt in parameter "prompt". It's None in default. |
| `gpu_batch_size` | typing.Annotated[int, Gt(gt=0)] | `8` | the batch size for GPU inference. This controls how many images are processed together in a single GPU forward pass. Useful when the dataset batch size is larger than what the GPU can handle. Default is 8. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_captioning_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_captioning_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)