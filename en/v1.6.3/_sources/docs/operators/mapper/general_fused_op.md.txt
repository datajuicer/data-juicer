# general_fused_op

An explicitly fused operator designed to execute multiple sequential operations (OPs) on the same batch, enabling fine-grained control over data processing.

This operator allows for the chaining of multiple data processing steps, such as mappers and filters, into a single pass. It processes each batch of samples sequentially through the defined operations, ensuring that all specified transformations are applied in order. The operator supports both mappers, which transform data, and filters, which remove or keep samples based on computed statistics. Context variables can be passed between operations if needed. The accelerator is set to 'cuda' if any of the fused operations use it. The number of processes is determined by the minimum value among all fused operations. After processing, any temporary context variables, such as those used for video containers, are cleaned up.

一个显式融合的算子，旨在对同一批次数据执行多个顺序操作（OP），从而实现对数据处理的细粒度控制。

该算子允许将多个数据处理步骤（如映射器和过滤器）串联为单次遍历。它按顺序将每个样本批次依次通过所定义的操作，确保所有指定的转换按序应用。该算子同时支持映射器（用于转换数据）和过滤器（基于计算出的统计信息移除或保留样本）。如有需要，操作之间可传递上下文变量。如果任一融合操作使用了 'cuda'，则加速器将被设为 'cuda'。进程数量由所有融合操作中的最小值决定。处理完成后，任何临时上下文变量（例如用于视频容器的变量）都将被清理。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_size` | <class 'int'> | `1` | the batch size of the input samples. |
| `fused_op_list` | typing.Optional[typing.List] | `None` | a list of OPs to be fused. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/op_fusion.py)
- [unit test 单元测试](../../../tests/ops/test_op_fusion.py)
- [Return operator list 返回算子列表](../../Operators.md)