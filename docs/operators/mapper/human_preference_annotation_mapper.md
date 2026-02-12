# human_preference_annotation_mapper

Operator for human preference annotation using Label Studio.

This operator formats and presents pairs of answers to a prompt for human evaluation. It uses a default or custom Label Studio configuration to display the prompt and answer options. The operator processes the annotations to determine the preferred answer, updating the sample with the chosen and rejected answers. The operator requires specific keys in the samples for the prompt and answer options. If these keys are missing, it logs warnings and uses placeholder text. The annotated results are processed to update the sample with the chosen and rejected answers.

使用Label Studio进行人类偏好标注的算子。

该算子格式化并呈现一对答案供人类评估。它使用默认或自定义的Label Studio配置来显示提示和答案选项。算子处理注释以确定首选答案，并更新带有选择和拒绝答案的样本。算子需要样本中具有提示和答案选项的特定键。如果缺少这些键，它会记录警告并使用占位符文本。注释结果被处理以更新带有选择和拒绝答案的样本。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `label_config_file` | <class 'str'> | `None` | Path to the label config file |
| `answer1_key` | <class 'str'> | `'answer1'` | Key for the first answer |
| `answer2_key` | <class 'str'> | `'answer2'` | Key for the second answer |
| `prompt_key` | <class 'str'> | `'prompt'` | Key for the prompt/question |
| `chosen_key` | <class 'str'> | `'chosen'` | Key for the chosen answer |
| `rejected_key` | <class 'str'> | `'rejected'` | Key for the rejected answer |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/annotation/human_preference_annotation_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/annotation/test_human_preference_annotation_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)