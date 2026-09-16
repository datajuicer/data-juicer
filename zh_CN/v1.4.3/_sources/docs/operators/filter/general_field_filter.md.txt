# general_field_filter

Filter to keep samples based on a general field filter condition.

The filter condition is a string that can include logical operators (and/or) and chain comparisons. For example: "10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3". The condition is evaluated for each sample, and only samples that meet the condition are kept. The result of the filter condition is stored in the sample's stats under the key 'general_field_filter_condition'. If the filter condition is empty or already computed, the sample is not re-evaluated.

基于通用字段过滤条件来保留样本。

过滤条件是一个字符串，可以包含逻辑运算符（and/or）和链式比较。例如："10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3"。该条件会对每个样本进行评估，只有满足条件的样本才会被保留。过滤条件的结果存储在样本的 stats 字段中，键名为 'general_field_filter_condition'。如果过滤条件为空或已经计算过，则不会重新评估样本。

Type 算子类型: **filter**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_condition` | <class 'str'> | `''` | The filter condition as a string. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/general_field_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_general_field_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)