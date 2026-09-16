# image_diffusion_mapper

Generate images using a diffusion model based on provided captions.

This operator uses a Hugging Face diffusion model to generate images from given captions. It supports different modes for retaining generated samples, including random selection, similarity-based selection, and retaining all. The operator can also generate captions if none are provided, using a Hugging Face image-to-sequence model. The strength parameter controls the extent of transformation from the reference image, and the guidance scale influences how closely the generated images match the text prompt. Generated images can be saved in a specified directory or the same directory as the input files. This is a batched operation, processing multiple samples at once and producing a specified number of augmented images per sample.

根据提供的描述使用扩散模型生成图像。

该算子使用 Hugging Face 扩散模型从给定的描述生成图像。它支持不同的保留生成样本模式，包括随机选择、基于相似性的选择和保留所有。如果未提供描述，该算子可以使用 Hugging Face 的图像到序列模型生成描述。强度参数控制从参考图像的变换程度，指导尺度影响生成图像与文本提示的匹配程度。生成的图像可以保存在指定目录或与输入文件相同的目录中。这是一个批量操作，一次处理多个样本并为每个样本生成指定数量的增强图像。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_diffusion` | <class 'str'> | `'CompVis/stable-diffusion-v1-4'` | diffusion model name on huggingface to generate the image. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `torch_dtype` | <class 'str'> | `'fp32'` | the floating point type used to load the diffusion model. Can be one of ['fp32', 'fp16', 'bf16'] |
| `revision` | <class 'str'> | `'main'` | The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier allowed by Git. |
| `strength` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `0.8` | Indicates extent to transform the reference image. Must be between 0 and 1. image is used as a starting point and more noise is added the higher the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in num_inference_steps. A value of 1 essentially ignores image. |
| `guidance_scale` | <class 'float'> | `7.5` | A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1. |
| `aug_num` | typing.Annotated[int, Gt(gt=0)] | `1` | The image number to be produced by stable-diffusion model. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated captions in the final datasets and the original captions will be removed. It's True by default. |
| `caption_key` | typing.Optional[str] | `None` | the key name of fields in samples to store captions for each images. It can be a string if there is only one image in each sample. Otherwise, it should be a list. If it's none, ImageDiffusionMapper will produce captions for each images. |
| `hf_img2seq` | <class 'str'> | `'Salesforce/blip2-opt-2.7b'` | model name on huggingface to generate caption if caption_key is None. |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_diffusion_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_diffusion_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)