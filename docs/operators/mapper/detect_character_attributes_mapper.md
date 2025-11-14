# detect_character_attributes_mapper

Takes an image, a caption, and main character names as input to extract the characters' attributes. 

Extracts and classifies attributes of main characters in an image using a combination of object detection, image-text matching, and language model inference. It first locates the main characters in the image using YOLOE and then uses a Hugging Face tokenizer and a LLaMA-based model to classify each character into categories like 'object', 'animal', 'person', 'text', or 'other'. The operator also extracts detailed features such as color, material, and action for each character. The final output includes bounding boxes and a list of characteristics for each main character. The results are stored in the 'main_character_attributes_list' field under the 'meta' key.

根据给定的图像、图像描述信息和（多个）角色名称，提取图像中主要角色的属性。

使用对象检测、图文匹配和语言模型推理的组合来提取和分类图像中主要角色的属性。它首先使用YOLOE定位图片中的主要角色，然后使用Hugging Face的tokenizer和基于LLaMA的模型将每个角色分类为'object'、'animal'、'person'、'text'或'other'等类别。算子还提取每个角色的详细特征，如颜色、材质和动作。最终输出包括每个主要角色的边界框和特征列表。结果存储在'meta'键下的'main_character_attributes_list'字段中。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `detect_character_locations_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for detect_character_locations_mapper_args. Controls the threshold for locating the main character. Default empty dict will use fixed values: default mllm_mapper_args, default image_text_matching_filter_args, yoloe_path="yoloe-11l-seg.pt", iou_threshold=0.7, matching_score_threshold=0.4, |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/detect_character_attributes_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_detect_character_attributes_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)