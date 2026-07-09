# combined_logical_filter

A combined filter operator that applies multiple filter operators with logical operations (AND/OR).

This is a composition operator that combines multiple filter operators and applies a logical operation (AND or OR) to their results. It's more explicit and self-documenting than using separate filters in sequence.

组合型过滤算子，将多个过滤算子组合并应用逻辑运算（AND/OR）。

这是一个组合算子，将多个过滤算子组合并对其结果应用逻辑运算（AND 或 OR）。比在序列中使用单独的过滤器更明确和自文档化。

Type 算子类型: **filter**

Tags 标签: cpu, gpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_ops` | <class 'list'> | `[]` | A list of filter operator configurations. Each item should be a dict with operator name as key and its parameters as value. Example: [{"text_length_filter": {"min_len": 10, "max_len": 100}}, {"language_id_score_filter": {"lang": "zh", "min_score": 0.8}}] |
| `logical_op` | <class 'str'> | `'and'` | The logical operator to combine filter results. Can be "and" or "or" (case-insensitive). When "and" is used, a sample is kept only if all filters return True. When "or" is used, a sample is kept if any filter returns True. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示

### Example 1: AND operation (默认)
```python
CombinedLogicalFilter(
    filter_ops=[
        {"text_length_filter": {"min_len": 10, "max_len": 100}},
        {"language_id_score_filter": {"lang": "zh", "min_score": 0.8}}
    ],
    logical_op="and"
)
```

This configuration will keep only samples that:
- Have text length between 10 and 100 characters **AND**
- Have Chinese language score >= 0.8

此配置将只保留同时满足以下条件的样本：
- 文本长度在10到100个字符之间 **且**
- 中文语言得分 >= 0.8

### Example 2: OR operation
```python
CombinedLogicalFilter(
    filter_ops=[
        {"text_length_filter": {"min_len": 10, "max_len": 50}},
        {"text_length_filter": {"min_len": 100, "max_len": 200}}
    ],
    logical_op="or"
)
```

This configuration will keep samples that:
- Have text length between 10 and 50 characters **OR**
- Have text length between 100 and 200 characters

此配置将保留满足以下任一条件的样本：
- 文本长度在10到50个字符之间 **或**
- 文本长度在100到200个字符之间

### Example 3: Configuration file usage (配置文件使用示例)

In a YAML configuration file:

```yaml
process:
  - combined_logical_filter:
      filter_ops:
        - text_length_filter:
            min_len: 10
            max_len: 1000
        - language_id_score_filter:
            lang: 'zh'
            min_score: 0.8
      logical_op: 'and'
```

## 💡 Design Rationale 设计理由

This operator is designed as a **composition operator** rather than a simple filter. It explicitly combines multiple filters with a logical operation, making the intent clear and self-documenting.

这个算子被设计为一个**组合算子**而不是简单的过滤器。它明确地将多个过滤器与逻辑运算组合，使意图清晰且自文档化。

### Comparison with Sequential Filters 与串联过滤器的对比

**Sequential Filters 串联过滤器**:
```yaml
process:
  - text_length_filter: {min_len: 10, max_len: 50}
  - text_length_filter: {min_len: 100, max_len: 200}
```
- ❌ **Implicit AND only**: Filters are applied sequentially, each filtering the result of the previous one. This is equivalent to AND logic.
- ❌ **Cannot implement OR**: There's no way to keep samples that satisfy **either** condition.
- ❌ **Unclear intent**: The logical relationship is implicit and not self-documenting.
- ⚠️ **Performance**: Each filter processes data independently, with intermediate data reduction and reorganization.

- ❌ **仅隐式 AND**：过滤器按顺序应用，每个过滤器过滤前一个的结果。这相当于 AND 逻辑。
- ❌ **无法实现 OR**：无法保留满足**任一**条件的样本。
- ❌ **意图不明确**：逻辑关系是隐式的，不自文档化。
- ⚠️ **性能**：每个过滤器独立处理数据，需要中间数据缩减和重组。

**Combined Logical Filter 组合逻辑过滤器**:
```yaml
process:
  - combined_logical_filter:
      filter_ops:
        - text_length_filter: {min_len: 10, max_len: 50}
        - text_length_filter: {min_len: 100, max_len: 200}
      logical_op: 'or'  # ✅ Now possible!
```
- ✅ **Explicit logic**: AND/OR relationship is clearly specified.
- ✅ **OR support**: Can now implement OR operations that sequential version impossible.
- ✅ **Self-documenting**: The intent is clear from the configuration.
- ✅ **Performance optimized**: All filters compute stats in one batch, avoiding intermediate data reorganization.

- ✅ **显式逻辑**：AND/OR 关系明确指定。
- ✅ **OR 支持**：现在可以实现隐式串联版本无法实现的 OR 操作。
- ✅ **自文档化**：意图从配置中清晰可见。
- ✅ **性能优化**：所有过滤器在一次批处理中计算统计信息，避免中间数据重组。

### Comparison with FusedFilter 与 FusedFilter 的对比

**FusedFilter (Automatic Fusion) 自动融合**:
- Framework automatically fuses consecutive filters that share intermediate variables
- Optimized for performance (reduces memory and speeds up processing)
- Default logical operation is AND
- User cannot explicitly control which filters are fused or the logical operation
- Best for: Performance optimization when filters share intermediate variables

- 框架自动融合共享中间变量的连续过滤器
- 针对性能优化（减少内存并加速处理）
- 默认逻辑运算是 AND
- 用户无法显式控制哪些过滤器被融合或逻辑运算
- 适用于：共享中间变量的过滤器性能优化

**Combined Logical Filter (Explicit Control) 显式控制**:
- User explicitly specifies which filters to combine
- Supports both AND and OR operations
- User has full control over the logical relationship
- Works with any filters, regardless of intermediate variables
- Best for: When you need OR operations or explicit control over filter combinations

- 用户显式指定要组合的过滤器
- 支持 AND 和 OR 操作
- 用户完全控制逻辑关系
- 适用于任何过滤器，不依赖中间变量
- 适用于：需要 OR 操作或对过滤器组合进行显式控制的场景

### When to Use Each 何时使用哪个

| Scenario 场景 | Recommended Approach 推荐方式 | Reason 原因 |
|--------------|---------------------------|-----------|
| Multiple filters with AND logic, sharing intermediate vars | FusedFilter (automatic) | Performance optimization |
| 多个过滤器使用 AND 逻辑，共享中间变量 | FusedFilter（自动） | 性能优化 |
| Need OR operation | combined_logical_filter | Only way to implement OR |
| 需要 OR 操作 | combined_logical_filter | 实现 OR 的唯一方式 |
| Explicit control over filter combination | combined_logical_filter | Full user control |
| 对过滤器组合的显式控制 | combined_logical_filter | 完全的用户控制 |
| Complex filter logic | combined_logical_filter | Clear and self-documenting |
| 复杂的过滤器逻辑 | combined_logical_filter | 清晰且自文档化 |

## 🚀 Performance Advantages 性能优势

Compared to sequential filters, `combined_logical_filter` provides several performance benefits:

与串联过滤器相比，`combined_logical_filter` 提供以下性能优势：

### 1. Optimized Statistics Computation 优化的统计信息计算

**Sequential Filters 串联过滤器**:
```
Dataset → Filter1.compute_stats → Filter1.process → Reduced Dataset1
       → Filter2.compute_stats → Filter2.process → Reduced Dataset2
```
- Each filter computes stats independently
- Dataset is reduced after each filter
- Multiple data reorganizations required
- May duplicate some statistics computation

- 每个过滤器独立计算统计信息
- 每个过滤器后数据集被缩减
- 需要多次数据重组
- 可能重复某些统计信息计算

**Combined Logical Filter 组合逻辑过滤器**:
```
Dataset → All Filters.compute_stats (single batch)
       → All Filters.process (parallel evaluation)
       → Logical combination → Final Dataset
```
- All statistics computed in one batch pass
- No intermediate dataset reduction
- Single data reorganization
- Shared statistics computation

- 所有统计信息在一次批处理中计算完成
- 无中间数据集缩减
- 单次数据重组
- 共享统计信息计算

### 2. Batch Processing Optimization 批处理优化

- All filters' `process_batched` methods execute on the same batch
- Uses numpy's vectorized `logical_and`/`logical_or` operations
- Avoids multiple data passes and reorganizations
- Better memory locality and cache efficiency

- 所有过滤器的 `process_batched` 方法在同一批数据上执行
- 使用 numpy 的向量化 `logical_and`/`logical_or` 操作
- 避免多次数据传递和重组
- 更好的内存局部性和缓存效率

### 3. OR Operation  OR 操作 

**This is the most important advantage**: OR operations were completely impossible with sequential filters. Now you can:

**这是最重要的优势**：OR 操作在串联过滤器中完全无法实现。现在您可以：

- Keep samples that satisfy **any** of multiple conditions
- Implement complex filtering logic (e.g., "short text OR long text, but not medium")
- Combine different filter types with OR logic

- 保留满足**任一**多个条件的样本
- 实现复杂的过滤逻辑（例如，"短文本或长文本，但不是中等长度"）
- 将不同类型的过滤器与 OR 逻辑组合

## ⚠️ Notes 注意事项

1. **Filter Type Requirement**: All operators in `filter_ops` must be Filter instances. Other operator types (Mapper, etc.) are not supported.

   **过滤器类型要求**：`filter_ops` 中的所有算子必须是 Filter 实例。不支持其他算子类型（Mapper 等）。

2. **Processing Modes**: The operator automatically handles both batched and single-sample processing modes. It will fall back to single-sample processing for non-batched operators.

   **处理模式**：该算子自动处理批处理和单样本处理模式。对于非批处理算子，它将回退到单样本处理。

3. **Stats Computation**: All filters compute their stats before processing. The operator ensures this by calling `compute_stats_batched` for all filters first, which is more efficient than sequential computation.

   **统计信息计算**：所有过滤器在处理前计算其统计信息。该算子通过首先为所有过滤器调用 `compute_stats_batched` 来确保这一点，这比顺序计算更高效。

4. **Ray Compatibility**: This operator is compatible with Ray executor. It properly handles CUDA-accelerated filters and context variables.

   **Ray 兼容性**：该算子兼容 Ray 执行器。它正确处理 CUDA 加速的过滤器和上下文变量。

5. **Performance**: For best performance, use batched filter operators. The operator will automatically optimize batch processing.

   **性能**：为了获得最佳性能，请使用批处理过滤器算子。该算子将自动优化批处理。

## 🔄 Ray Mode Compatibility Ray 模式兼容性

This operator is compatible with Ray executor:

该算子兼容 Ray 执行器：

- ✅ Supports batched processing in Ray mode
- ✅ Handles CUDA-accelerated filters correctly
- ✅ Properly manages context variables for intermediate operations
- ✅ Works with Ray's distributed processing

- ✅ 在 Ray 模式下支持批处理
- ✅ 正确处理 CUDA 加速的过滤器
- ✅ 正确管理中间操作的上下文变量
- ✅ 与 Ray 的分布式处理兼容

## 📝 Real-World Use Cases 实际使用场景

### Use Case 1: OR Operation 场景1：OR 操作

**Scenario**: Keep samples that are either short (10-50 chars) or long (100-200 chars), but exclude medium-length text.

**场景**：保留短文本（10-50 字符）或长文本（100-200 字符），但排除中等长度文本。

```yaml
process:
  - combined_logical_filter:
      filter_ops:
        - text_length_filter: {min_len: 10, max_len: 50}
        - text_length_filter: {min_len: 100, max_len: 200}
      logical_op: 'or'
```

**Why this was impossible before**: Sequential filters can only implement AND logic. The first filter would remove all samples outside its range, leaving nothing for the second filter to process.

**为什么Default Sequential Version 无法实现**：串联过滤器只能实现 AND 逻辑。第一个过滤器会移除其范围外的所有样本，第二个过滤器无法处理。

### Use Case 2: Multi-Criteria Filtering 场景2：多条件过滤

**Scenario**: Keep samples that are in Chinese OR have high quality scores.

**场景**：保留中文样本或高质量得分样本。

```yaml
process:
  - combined_logical_filter:
      filter_ops:
        - language_id_score_filter: {lang: 'zh', min_score: 0.8}
        - llm_quality_score_filter: {min_score: 0.9}
      logical_op: 'or'
```

### Use Case 3: Performance Optimization 场景3：性能优化

**Scenario**: Multiple filters with AND logic, but you want explicit control and better performance than sequential filters.

**场景**：多个过滤器使用 AND 逻辑，但您想要显式控制和比串联过滤器更好的性能。

```yaml
process:
  - combined_logical_filter:
      filter_ops:
        - text_length_filter: {min_len: 10, max_len: 1000}
        - language_id_score_filter: {lang: 'zh', min_score: 0.8}
        - alphanumeric_filter: {min_ratio: 0.5}
      logical_op: 'and'
```

**Benefits**: All statistics computed in one pass, no intermediate data reduction, better performance.

**优势**：所有统计信息在一次遍历中计算，无中间数据缩减，性能更好。

## 🚀 Future Enhancements 未来增强

Potential improvements for future versions:

未来版本的潜在改进：

1. **Nested Logical Operations**: Support for complex nested logical expressions (e.g., (A AND B) OR (C AND D))

   **嵌套逻辑运算**：支持复杂的嵌套逻辑表达式（例如，(A AND B) OR (C AND D)）

2. **Performance Optimization**: Further optimization for large-scale batch processing

   **性能优化**：进一步优化大规模批处理

3. **Integration with FusedFilter**: Automatic detection and optimization when combined with framework's automatic fusion

   **与 FusedFilter 集成**：与框架的自动融合结合时自动检测和优化

4. **Validation**: Enhanced validation and error messages for invalid configurations

   **验证**：增强的验证和错误消息，用于无效配置

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/combined_logical_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_combined_logical_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)
