# detect_character_locations_mapper

Given an image and a list of main character names, extract the bounding boxes for each present character.

Detects and extracts bounding boxes for main characters in an image, this operator uses a YOLOE model to detect the presence of these characters. It then generates and refines bounding boxes for each detected character using a multimodal language model and an image-text matching filter. The final bounding boxes are stored in the metadata under 'main_character_locations_list'. The operator considers two bounding boxes as overlapping if their Intersection over Union (IoU) score exceeds a specified threshold. Additionally, it uses a matching score threshold to determine if a cropped image region matches the character's name. The operator utilizes a Hugging Face tokenizer and a BLIP model for image-text matching.

给定一张图像和主要角色的名称列表，提取每个在场角色的边界框。

检测并提取图像中主要角色的边界框，该算子使用YOLOE模型检测这些角色的存在。然后，它使用多模态语言模型和图文匹配过滤器生成并优化每个检测到的角色的边界框。最终的边界框存储在元数据的'main_character_locations_list'下。如果两个边界框的交并比（IoU）得分超过指定阈值，则算子认为它们是重叠的。此外，它使用匹配得分阈值来确定裁剪的图像区域是否与角色名称匹配。算子使用Hugging Face的tokenizer和BLIP模型进行图文匹配。

Type 算子类型: **mapper**

Tags 标签: gpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mllm_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for multimodal language model mapper. Controls the generation of captions for bounding box regions. Default empty dict will use fixed values: max_new_tokens=256, temperature=0.2, top_p=None, num_beams=1, hf_model="llava-hf/llava-v1.6-vicuna-7b-hf". |
| `image_text_matching_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for image-text matching filter. Controls the matching between cropped image regions and text descriptions. Default empty dict will use fixed values: min_score=0.1, max_score=1.0, hf_blip="Salesforce/blip-itm-base-coco", num_proc=1. |
| `yoloe_path` |  | `'yoloe-11l-seg.pt'` | The path to the YOLOE model. |
| `iou_threshold` |  | `0.7` | We consider two bounding boxes from different models to be overlapping when their IOU score is higher than the iou_threshold. |
| `matching_score_threshold` |  | `0.4` | If the matching score between the cropped image and the character's name exceeds the matching_score_threshold, they are considered a match. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/detect_character_locations_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_detect_character_locations_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)