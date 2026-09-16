# imgdiff_difference_caption_generator_mapper

Generates difference captions for bounding box regions in two images.

This operator processes pairs of images and generates captions for the differences in their bounding box regions. It uses a multi-step process:
- Describes the content of each bounding box region using a Hugging Face model.
- Crops the bounding box regions from both images.
- Checks if the cropped regions match the generated captions.
- Determines if there are differences between the two captions.
- Marks the difference area with a red box.
- Generates difference captions for the marked areas.
- The key metric is the similarity score between the captions, computed using a CLIP model.
- If no valid bounding boxes or differences are found, it returns empty captions and zeroed bounding boxes.
- Uses 'cuda' as the accelerator if any of the fused operations support it.
- Caches temporary images during processing and clears them afterward.

为两幅图像的边界框区域生成差异描述。

此算子处理成对的图像并为其边界框区域的差异生成描述。它使用多步骤过程：
- 使用 Hugging Face 模型描述每个边界框区域的内容。
- 从两幅图像中裁剪出边界框区域。
- 检查裁剪区域是否与生成的描述匹配。
- 确定两个描述之间是否存在差异。
- 用红色框标记差异区域。
- 为标记区域生成差异描述。
- 关键指标是使用 CLIP 模型计算的描述之间的相似度得分。
- 如果没有找到有效的边界框或差异，则返回空描述和零化的边界框。
- 如果任何融合操作支持，则使用 'cuda' 作为加速器。
- 在处理过程中缓存临时图像并在之后清除它们。

Type 算子类型: **mapper**

Tags 标签: gpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mllm_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for multimodal language model mapper. Controls the generation of captions for bounding box regions. Default empty dict will use fixed values: max_new_tokens=256, temperature=0.2, top_p=None, num_beams=1, hf_model="llava-hf/llava-v1.6-vicuna-7b-hf". |
| `image_text_matching_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for image-text matching filter. Controls the matching between cropped regions and generated captions. Default empty dict will use fixed values: min_score=0.1, max_score=1.0, hf_blip="Salesforce/blip-itm-base-coco", num_proc=1. |
| `text_pair_similarity_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for text pair similarity filter. Controls the similarity comparison between caption pairs. Default empty dict will use fixed values: min_score=0.1, max_score=1.0, hf_clip="openai/clip-vit-base-patch32", text_key_second="target_text", num_proc=1. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/imgdiff_difference_caption_generator_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_imgdiff_difference_caption_generator_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)