# image_tagging_mapper

Generates image tags for each image in the sample.

This operator processes images to generate descriptive tags. It uses a Hugging Face model to analyze the images and produce relevant tags. The tags are stored in the specified field, defaulting to 'image_tags'. If the tags are already present in the sample, the operator will not recompute them. For samples without images, an empty tag array is assigned. The generated tags are sorted by frequency and stored as a list of strings.

为样本中的每张图像生成描述性标签。

此算子处理图像以生成描述性标签。它使用 Hugging Face 模型分析图像并生成相关标签。标签存储在指定字段中，默认为 'image_tags'。如果样本中已经存在标签，则不会重新计算。对于没有图像的样本，分配一个空标签数组。生成的标签按频率排序并作为字符串列表存储。

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tag_field_name` | <class 'str'> | `'image_tags'` | the field name to store the tags. It's |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_tagging_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_tagging_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)