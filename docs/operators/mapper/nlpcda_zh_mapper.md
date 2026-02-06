# nlpcda_zh_mapper

Augments Chinese text samples using the nlpcda library.

This operator applies various augmentation methods to Chinese text, such as replacing similar words, homophones, deleting random characters, swapping characters, and replacing equivalent numbers. The number of augmented samples generated can be controlled by the `aug_num` parameter. If `sequential` is set to True, the augmentation methods are applied in sequence; otherwise, they are applied independently. The original sample can be kept or removed based on the `keep_original_sample` flag. It is recommended to use 1-3 augmentation methods at a time to avoid significant changes in the semantics of the samples. Some augmentation methods may not work for special texts, resulting in no augmented samples being generated.

使用nlpcda库增强中文文本样本。

该算子应用各种增强方法来增强中文文本，如替换相似词、同音字、随机删除字符、交换字符和替换等价数字。可以通过`aug_num`参数控制生成的增强样本数量。如果`sequential`设置为True，则按顺序应用增强方法；否则，独立应用。可以根据`keep_original_sample`标志选择保留或移除原始样本。建议一次使用1-3种增强方法，以避免样本语义发生显著变化。某些增强方法可能对特殊文本不起作用，导致无法生成增强样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `sequential` | <class 'bool'> | `False` | whether combine all augmentation methods to a sequence. If it's True, a sample will be augmented by all opened augmentation methods sequentially. If it's False, each opened augmentation method would generate its augmented samples independently. |
| `aug_num` | typing.Annotated[int, Gt(gt=0)] | `1` | number of augmented samples to be generated. If `sequential` is True, there will be total aug_num augmented samples generated. If it's False, there will be (aug_num * #opened_aug_method) augmented samples generated. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated texts in the final datasets and the original texts will be removed. It's True in default. |
| `replace_similar_word` | <class 'bool'> | `False` | whether to open the augmentation method of replacing random words with their similar words in the original texts. e.g. "这里一共有5种不同的数据增强方法" --> "这边一共有5种不同的数据增强方法" |
| `replace_homophone_char` | <class 'bool'> | `False` | whether to open the augmentation method of replacing random characters with their homophones in the original texts. e.g. "这里一共有5种不同的数据增强方法" --> "这里一共有5种不同的濖据增强方法" |
| `delete_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of deleting random characters from the original texts. e.g. "这里一共有5种不同的数据增强方法" --> "这里一共有5种不同的数据增强" |
| `swap_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of swapping random contiguous characters in the original texts. e.g. "这里一共有5种不同的数据增强方法" --> "这里一共有5种不同的数据强增方法" |
| `replace_equivalent_num` | <class 'bool'> | `False` | whether to open the augmentation method of replacing random numbers with their equivalent representations in the original texts. **Notice**: Only for numbers for now. e.g. "这里一共有5种不同的数据增强方法" --> "这里一共有伍种不同的数据增强方法" |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/nlpcda_zh_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_nlpcda_zh_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)