# sdxl_prompt2prompt_mapper

Generates pairs of similar images using the SDXL model.

This operator uses a Hugging Face diffusion model to generate image pairs based on two text prompts. The quality and similarity of the generated images are controlled by parameters such as `num_inference_steps` and `guidance_scale`. The first and second text prompts are specified using `text_key` and `text_key_second`, respectively. The generated images are saved in the specified `output_dir` with unique filenames. The operator requires both text keys to be set for processing.

使用 SDXL 模型生成相似的图像对。

该算子使用 Hugging Face 的扩散模型根据两个文本提示生成图像对。生成图像的质量和相似度由 `num_inference_steps` 和 `guidance_scale` 等参数控制。第一个和第二个文本提示分别通过 `text_key` 和 `text_key_second` 指定。生成的图像保存在指定的 `output_dir` 中，并带有唯一的文件名。算子要求设置两个文本键才能进行处理。

Type 算子类型: **mapper**

Tags 标签: gpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_diffusion` | <class 'str'> | `'stabilityai/stable-diffusion-xl-base-1.0'` | diffusion model name on huggingface to generate the image. |
| `trust_remote_code` |  | `False` | whether to trust the remote code of HF models. |
| `torch_dtype` | <class 'str'> | `'fp32'` | the floating point type used to load the diffusion model. |
| `num_inference_steps` | <class 'float'> | `50` | The larger the value, the better the image generation quality; however, this also increases the time required for generation. |
| `guidance_scale` | <class 'float'> | `7.5` | A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when |
| `text_key` |  | `None` | the key name used to store the first caption in the caption pair. |
| `text_key_second` |  | `None` | the key name used to store the second caption in the caption pair. |
| `output_dir` |  | `DATA_JUICER_ASSETS_CACHE` | the storage location of the generated images. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/sdxl_prompt2prompt_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_sdxl_prompt2prompt_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)