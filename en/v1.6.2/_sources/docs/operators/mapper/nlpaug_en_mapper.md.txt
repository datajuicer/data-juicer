# nlpaug_en_mapper

Augments English text samples using various methods from the nlpaug library.

This operator applies a series of text augmentation techniques to generate new samples. It supports both word-level and character-level augmentations, such as deleting, swapping, and inserting words or characters. The number of augmented samples can be controlled, and the original samples can be kept or removed. When multiple augmentation methods are enabled, they can be applied sequentially or independently. Sequential application means each sample is augmented by all enabled methods in sequence, while independent application generates multiple augmented samples for each method. We recommend using 1-3 augmentation methods at a time to avoid significant changes in sample semantics.

使用nlpaug库中的各种方法增强英文文本样本。

该算子应用一系列文本增强技术来生成新的样本。它支持词级和字符级的增强，如删除、交换和插入单词或字符。可以控制增强样本的数量，并且可以选择保留或移除原始样本。当启用多个增强方法时，它们可以按顺序应用或独立应用。顺序应用意味着每个样本按顺序由所有启用的方法进行增强，而独立应用则为每种方法生成多个增强样本。建议一次使用1-3种增强方法，以避免样本语义发生显著变化。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `sequential` | <class 'bool'> | `False` | whether combine all augmentation methods to a sequence. If it's True, a sample will be augmented by all opened augmentation methods sequentially. If it's False, each opened augmentation method would generate its augmented samples independently. |
| `aug_num` | typing.Annotated[int, Gt(gt=0)] | `1` | number of augmented samples to be generated. If `sequential` is True, there will be total aug_num augmented samples generated. If it's False, there will be (aug_num * #opened_aug_method) augmented samples generated. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated texts in the final datasets and the original texts will be removed. It's True in default. |
| `delete_random_word` | <class 'bool'> | `False` | whether to open the augmentation method of deleting random words from the original texts. e.g. "I love LLM" --> "I LLM" |
| `swap_random_word` | <class 'bool'> | `False` | whether to open the augmentation method of swapping random contiguous words in the original texts. e.g. "I love LLM" --> "Love I LLM" |
| `spelling_error_word` | <class 'bool'> | `False` | whether to open the augmentation method of simulating the spelling error for words in the original texts. e.g. "I love LLM" --> "Ai love LLM" |
| `split_random_word` | <class 'bool'> | `False` | whether to open the augmentation method of splitting words randomly with whitespaces in the original texts. e.g. "I love LLM" --> "I love LL M" |
| `keyboard_error_char` | <class 'bool'> | `False` | whether to open the augmentation method of simulating the keyboard error for characters in the original texts. e.g. "I love LLM" --> "I ;ov4 LLM" |
| `ocr_error_char` | <class 'bool'> | `False` | whether to open the augmentation method of simulating the OCR error for characters in the original texts. e.g. "I love LLM" --> "I 10ve LLM" |
| `delete_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of deleting random characters from the original texts. e.g. "I love LLM" --> "I oe LLM" |
| `swap_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of swapping random contiguous characters in the original texts. e.g. "I love LLM" --> "I ovle LLM" |
| `insert_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of inserting random characters into the original texts. e.g. "I love LLM" --> "I ^lKove LLM" |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/nlpaug_en_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_nlpaug_en_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)