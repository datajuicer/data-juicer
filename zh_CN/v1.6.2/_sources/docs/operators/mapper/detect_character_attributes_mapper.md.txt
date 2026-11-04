# detect_character_attributes_mapper

Takes an image, a caption, and main character names as input to extract the characters' attributes.

Extracts and classifies attributes of main characters in an image using a combination of object detection, image-text matching, and language model inference. It first locates the main characters in the image using YOLOE and then uses a Hugging Face tokenizer and a LLaMA-based model to classify each character into categories like 'object', 'animal', 'person', 'text', or 'other'. The operator also extracts detailed features such as color, material, and action for each character. The final output includes bounding boxes and a list of characteristics for each main character. The results are stored in the 'main_character_attributes_list' field under the 'meta' key.

以图像、标题和主要角色名称作为输入，提取角色的属性。

该算子结合目标检测、图文匹配和语言模型推理，提取并分类图像中主要角色的属性。首先使用 YOLOE 定位图像中的主要角色，然后利用 Hugging Face tokenizer 和基于 LLaMA 的模型将每个角色分类为“object”（物体）、“animal”（动物）、“person”（人物）、“text”（文本）或“other”（其他）等类别。该算子还会为每个角色提取颜色、材质和动作等详细特征。最终输出包括每个主要角色的边界框及其特征列表，并将结果存储在 'meta' 键下的 'main_character_attributes_list' 字段中。

Type 算子类型: **mapper**

Tags 标签: gpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `detect_character_locations_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for detect_character_locations_mapper_args. Controls the threshold for locating the main character. Default empty dict will use fixed values: default mllm_mapper_args, default image_text_matching_filter_args, yoloe_path="yoloe-11l-seg.pt", iou_threshold=0.7, matching_score_threshold=0.4, |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/detect_character_attributes_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_detect_character_attributes_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)