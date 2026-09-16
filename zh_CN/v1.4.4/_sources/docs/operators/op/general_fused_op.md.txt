# general_fused_op

An explicitly fused operator designed to execute multiple sequential operations (OPs) on the same batch, enabling fine-grained control over data processing.

This operator allows for the chaining of multiple data processing steps, such as mappers and filters, into a single pass. It processes each batch of samples sequentially through the defined operations, ensuring that all specified transformations are applied in order. The operator supports both mappers, which transform data, and filters, which remove or keep samples based on computed statistics. Context variables can be passed between operations if needed. The accelerator is set to 'cuda' if any of the fused operations use it. The number of processes is determined by the minimum value among all fused operations. After processing, any temporary context variables, such as those used for video containers, are cleaned up.

一个显式融合的算子，旨在对同一批次执行多个顺序操作（OP），从而实现对数据处理的细粒度控制。

该算子允许将多个数据处理步骤（如映射器和过滤器）链接成一次传递。它按定义的操作顺序逐批处理样本，确保所有指定的变换按顺序应用。该算子支持映射器（用于变换数据）和过滤器（基于计算的统计信息移除或保留样本）。如果需要，可以在操作之间传递上下文变量。如果任何融合操作使用了CUDA，则加速器被设置为'cuda'。进程数由所有融合操作中的最小值决定。处理后，会清理任何临时上下文变量，例如用于视频容器的变量。

Type 算子类型: **op**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_size` | <class 'int'> | `1` | the batch size of the input samples. |
| `fused_op_list` | typing.Optional[typing.List] | `None` | a list of OPs to be fused. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/op_fusion.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)