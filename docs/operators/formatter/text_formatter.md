# text_formatter


The class is used to load and format text-type files.

e.g. `['.txt', '.pdf', '.cpp', '.docx']`


该类用于加载和格式化文本类型的文件。

例如：`['.txt', '.pdf', '.cpp', '.docx']`

Type 算子类型: **formatter**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `dataset_path` |  | `''` | a dataset file or a dataset directory |
| `suffixes` |  | `None` | files with specified suffixes to be processed |
| `add_suffix` |  | `False` | Whether to add file suffix to dataset meta info |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/format/text_formatter.py)
- [unit test 单元测试](../../../tests/format/test_text_formatter.py)
- [Return operator list 返回算子列表](../../Operators.md)