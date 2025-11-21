# detect_main_character_mapper

Extract all main character names based on the given image and its caption. 

This operator uses a multimodal language model to generate a description of the main characters in the given image. It then parses the generated JSON to extract the list of main characters. The operator filters out samples where the number of main characters is less than the specified threshold. The default arguments for the multimodal language model include using a Hugging Face model with specific generation parameters. The key metric, `main_character_list`, is stored in the sample's metadata.

根据给定的图像及其图像描述，提取所有主要角色的名字。

该算子使用多模态语言模型生成给定图像中主要角色的描述。然后解析生成的JSON以提取主要角色列表。算子会过滤掉主要角色数量少于指定阈值的样本。多模态语言模型的默认参数包括使用特定生成参数的Hugging Face模型。关键指标`main_character_list`存储在样本的元数据中。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mllm_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for multimodal language model mapper. Controls the generation of captions for bounding box regions. Default empty dict will use fixed values: max_new_tokens=256, temperature=0.2, top_p=None, num_beams=1, hf_model="llava-hf/llava-v1.6-vicuna-7b-hf". |
| `filter_min_character_num` | <class 'int'> | `0` | Filters out samples where the number of main characters in the image is less than this threshold. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/detect_main_character_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_detect_main_character_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)