# in_context_influence_filter

Filter to keep texts based on their in-context influence on a validation set.

This operator calculates the in-context influence of each sample by comparing perplexities with and without the sample as context. The influence score is computed as the ratio of these perplexities. If `valid_as_demo` is True, the score is L(A|Q) / L(A|task_desc, Q_v, A_v, Q). Otherwise, it is L(A_v|Q) / L(A_v|task_desc, Q, A, Q_v). The operator retains samples whose in-context influence score is within a specified range. The in-context influence score is stored in the 'in_context_influence' field of the sample's stats. The validation set must be prepared using the `prepare_valid_feature` method if not provided during initialization.

根据文本在验证集上的上下文影响过滤保留文本。

该算子通过比较有无样本作为上下文时的困惑度来计算每个样本的上下文影响。影响分数计算为这些困惑度的比率。如果 `valid_as_demo` 为 True，分数为 L(A|Q) / L(A|task_desc, Q_v, A_v, Q)。否则，它为 L(A_v|Q) / L(A_v|task_desc, Q, A, Q_v)。该算子保留上下文影响分数在指定范围内的样本。上下文影响分数存储在样本统计信息的 'in_context_influence' 字段中。如果初始化时未提供，验证集必须使用 `prepare_valid_feature` 方法准备。

Type 算子类型: **filter**

Tags 标签: gpu, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. If None, 'self.prepare_valid_feature' should be manually called before applying the filter. |
| `task_desc` | <class 'str'> | `None` | The description of the validation task. |
| `valid_as_demo` | <class 'bool'> | `False` | If true, score =  L(A\|Q) / L(A\|task_desc, Q_v, A_v, Q); If false, score = L(A_v\|Q) L(A_v\|task_desc, Q, A, Q_v). |
| `n_shot` | typing.Optional[int] | `None` | The number of shots in validation. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/in_context_influence_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_in_context_influence_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)