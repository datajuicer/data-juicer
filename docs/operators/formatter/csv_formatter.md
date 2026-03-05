# csv_formatter

The class is used to load and format csv-type files.

Default suffixes is `['.csv']`

该类用于加载和格式化 CSV 类型的文件。

默认后缀为 `['.csv']`

Type 算子类型: **formatter**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `dataset_path` |  | `''` | a dataset file or a dataset directory |
| `suffixes` |  | `None` | files with specified suffixes to be processed |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/format/csv_formatter.py)
- [unit test 单元测试](../../../tests/format/test_csv_formatter.py)
- [Return operator list 返回算子列表](../../Operators.md)