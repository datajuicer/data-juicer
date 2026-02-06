# imgdiff_difference_area_generator_mapper

Generates and filters bounding boxes for image pairs based on similarity, segmentation, and text matching.

This operator processes image pairs to identify and filter regions with significant differences. It uses a sequence of operations:
- Filters out image pairs with large differences.
- Segments the images to identify potential objects.
- Crops sub-images based on bounding boxes.
- Determines if the sub-images contain valid objects using image-text matching.
- Filters out sub-images that are too similar.
- Removes overlapping bounding boxes.
- Uses Hugging Face models for similarity and text matching, and FastSAM for segmentation.
- Caches intermediate results in `DATA_JUICER_ASSETS_CACHE`.
- Returns the filtered bounding boxes in the `MetaKeys.bbox_tag` field.

基于相似性、分割和文本匹配生成并过滤图像对的边界框。

此算子处理图像对以识别和过滤具有显著差异的区域。它使用一系列操作：
- 过滤掉差异较大的图像对。
- 分割图像以识别潜在对象。
- 基于边界框裁剪子图像。
- 使用图像-文本匹配确定子图像是否包含有效对象。
- 过滤掉过于相似的子图像。
- 移除重叠的边界框。
- 使用 Hugging Face 模型进行相似性和文本匹配，使用 FastSAM 进行分割。
- 在 `DATA_JUICER_ASSETS_CACHE` 中缓存中间结果。
- 返回 `MetaKeys.bbox_tag` 字段中的过滤后的边界框。

Type 算子类型: **mapper**

Tags 标签: gpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `image_pair_similarity_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for image pair similarity filter. Controls the similarity filtering between image pairs. Default empty dict will use fixed values: min_score_1=0.1, max_score_1=1.0, min_score_2=0.1, max_score_2=1.0, hf_clip="openai/clip-vit-base-patch32", num_proc=1. |
| `image_segment_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for image segmentation mapper. Controls the image segmentation process. Default empty dict will use fixed values: imgsz=1024, conf=0.05, iou=0.5, model_path="FastSAM-x.pt". |
| `image_text_matching_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for image-text matching filter. Controls the matching between cropped image regions and text descriptions. Default empty dict will use fixed values: min_score=0.1, max_score=1.0, hf_blip="Salesforce/blip-itm-base-coco", num_proc=1. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/imgdiff_difference_area_generator_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_imgdiff_difference_area_generator_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)