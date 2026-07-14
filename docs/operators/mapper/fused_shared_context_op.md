# fused_shared_context_op

Runs mapper and filter operators sequentially in one batch stage while sharing a temporary per-sample context.

The shared context lets compatible operators reuse expensive intermediate values, such as split lines, extracted words, decoded images/audio/video, and sampled video frames. The context is batch-local: it is never returned as an output column, and owned PyAV containers are closed when their row is filtered, when processing fails, or when the batch completes.

在同一个批处理阶段中顺序执行 Mapper 和 Filter，并共享按样本保存的临时上下文。

共享上下文可让兼容算子复用分行结果、分词结果、已解码的图像/音频/视频及采样视频帧等开销较大的中间值。上下文只在当前批次内有效，不会作为结果列返回；其中持有的 PyAV 容器会在样本被过滤、处理失败或批次完成时关闭。

Type 算子类型: **mapper**

Tags 标签: cpu, cuda, text, image, audio, video

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_size` | `int` | `1` | Outer batch size of the fused stage. Must be positive. |
| `fused_op_list` | `Optional[List[Dict]]` | `None` | Ordered list of standard Data-Juicer operator configs. |

## Usage 使用方式

```yaml
process:
  - fused_shared_context_op:
      batch_size: 128
      fused_op_list:
        - average_line_length_filter:
            min_len: 10
        - maximum_line_length_filter:
            max_len: 10000
```

Both filters above use the same cached line split. Filters may drop rows, and later sub-ops receive the filtered batch.

上述两个 Filter 会复用同一次文本分行结果。Filter 可以删除样本，后续子算子会收到过滤后的批次。

## Safety contract 安全约束

- Only Mapper and Filter sub-ops are supported.
- Every returned top-level batch column must have the same row count.
- The input must not already contain the reserved `Fields.context` column.
- Context keys are shared only within one batch; there is no cross-batch cache.
- Sub-ops that share a context key must use the same source semantics and compatible parameters.
- Do not put a source-mutating Mapper between a context producer and consumer unless that Mapper also updates or removes the affected cached value. For example, cached lines are stale after changing the source text.
- GPU/resource scheduling belongs to the outer fused operator. Set its resource options to cover all inner operators used by the stage.

- 仅支持 Mapper 和 Filter 子算子。
- 返回批次的所有顶层列必须具有相同的样本数。
- 输入数据不得预先包含保留列 `Fields.context`。
- 上下文仅在单个批次内共享，不支持跨批次缓存。
- 共享同一上下文键的子算子必须采用一致的数据源语义及兼容参数。
- 除非 Mapper 同时更新或删除受影响的缓存值，否则不要把修改数据源的 Mapper 放在上下文生产者与消费者之间。例如，修改原文本后，先前缓存的分行结果将失效。
- GPU 和资源调度由外层融合算子负责；外层资源参数应覆盖阶段内所有子算子的需要。

`fused_shared_context_op` differs from `general_fused_op` by making context ownership, alignment, and cleanup an explicit contract. It remains a manually configured optimization and does not perform automatic fusion planning.

`fused_shared_context_op` 与 `general_fused_op` 的区别是：它显式约定了上下文的所有权、行对齐及资源清理行为。它仍是手工配置的优化，不会自动规划算子融合。

## 🔗 related links 相关链接

- [source code 源代码](../../../data_juicer/ops/fused_shared_context_op.py)
- [unit test 单元测试](../../../tests/ops/test_fused_shared_context_op.py)
- [Return operator list 返回算子列表](../../Operators.md)
