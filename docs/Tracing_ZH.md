# 数据追踪

本文档描述 DataJuicer 的追踪系统，用于跟踪数据处理过程中样本级别的变化。

## 概述

Tracer 记录每个算子在处理管道中如何修改、过滤或去重各个样本。这对以下场景非常有用：

- **调试** — 理解特定样本为何被修改或删除
- **质量保证** — 验证算子是否按预期工作
- **审计** — 维护数据转换的记录

## 配置

### 基本设置

```yaml
open_tracer: false        # 启用/禁用追踪
op_list_to_trace: []      # 要追踪的算子列表（空 = 所有算子）
trace_num: 10             # 每个算子最多收集的样本数
trace_keys: []            # 追踪输出中包含的额外字段
```

### 命令行

```bash
# 启用所有算子的追踪
dj-process --config config.yaml --open_tracer true

# 仅追踪特定算子
dj-process --config config.yaml --open_tracer true \
    --op_list_to_trace clean_email_mapper,words_num_filter

# 每个算子收集更多样本
dj-process --config config.yaml --open_tracer true --trace_num 50

# 在追踪输出中包含额外字段
dj-process --config config.yaml --open_tracer true \
    --trace_keys sample_id,source_file
```

## 输出结构

追踪结果存储在工作目录的 `trace/` 子目录中：

```
{work_dir}/
└── trace/
    ├── sample_trace-clean_email_mapper.jsonl
    ├── sample_trace-words_num_filter.jsonl
    ├── duplicate-document_deduplicator.jsonl
    └── ...
```

每个追踪文件为 JSONL 格式（每行一个 JSON 对象），内容因算子类型而异。

## 追踪的算子类型

### Mapper 追踪

对于 Mapper 算子，Tracer 记录文本内容发生变化的样本。每条记录包含：

| 字段 | 描述 |
|------|------|
| `original_text` | Mapper 处理前的文本 |
| `processed_text` | Mapper 处理后的文本 |
| *trace_keys 字段* | 配置的 `trace_keys` 对应的值 |

输出示例（`sample_trace-clean_email_mapper.jsonl`）：
```json
{"original_text":"联系我们 user@example.com 获取详情。","processed_text":"联系我们  获取详情。"}
{"original_text": "邮箱：admin@test.org", "processed_text": "邮箱："}
```

仅收集文本实际发生变化的样本，未变化的样本会被跳过。

### Filter 追踪

对于 Filter 算子，Tracer 记录被**过滤掉**（删除）的样本。每条记录包含完整的样本数据。

输出示例（`sample_trace-words_num_filter.jsonl`）：
```json
{"text": "Too short.", "__dj__stats__": {"words_num": 2}}
{"text": "Also brief.", "__dj__stats__": {"words_num": 2}}
```

仅收集被过滤掉的样本，通过过滤器的样本会被跳过。

### Deduplicator 追踪

对于 Deduplicator 算子，Tracer 记录近似重复的样本对。每条记录包含：

| 字段 | 描述 |
|------|------|
| `dup1` | 重复对中的第一个样本 |
| `dup2` | 重复对中的第二个样本 |

输出示例（`duplicate-document_deduplicator.jsonl`）：
```json
{"dup1": "这是一段重复的文本。", "dup2": "这是一段重复的文本。"}
```

## 样本收集行为

Tracer 使用高效的**样本级收集**方式：

1. 每个算子在处理过程中最多收集 `trace_num` 个样本
2. 收集到足够样本后提前停止
3. 在默认模式下，收集是**线程安全**的，使用多进程锁
4. 在 Ray 模式下，每个 Worker 有自己的 Tracer 实例（无需加锁）

这种设计最大限度地减少了性能开销——Tracer 不会比较整个数据集，而是在处理过程中实时捕获变化。

## trace_keys

`trace_keys` 选项允许在追踪输出中包含原始样本的额外字段。这对于识别哪些样本受到影响非常有用：

```yaml
open_tracer: true
trace_keys:
  - sample_id
  - source_file
```

使用此配置，Mapper 追踪条目将包含：
```json
{
  "sample_id": "doc_00042",
  "source_file": "corpus_part1.jsonl",
  "original_text": "原始内容...",
  "processed_text": "处理后的内容..."
}
```

## API 参考

### Tracer（默认模式）

```python
from data_juicer.core.tracer import Tracer

tracer = Tracer(
    work_dir="./outputs",
    op_list_to_trace=["clean_email_mapper", "words_num_filter"],
    show_num=10,
    trace_keys=["sample_id"]
)

# 检查某个算子是否需要追踪
tracer.should_trace_op("clean_email_mapper")  # True

# 检查是否已收集足够的样本
tracer.is_collection_complete("clean_email_mapper")  # False

# 收集 Mapper 样本
tracer.collect_mapper_sample(
    op_name="clean_email_mapper",
    original_sample={"text": "邮箱：a@b.com"},
    processed_sample={"text": "邮箱："},
    text_key="text"
)

# 收集 Filter 样本
tracer.collect_filter_sample(
    op_name="words_num_filter",
    sample={"text": "太短"},
    should_keep=False
)
```

### RayTracer（分布式模式）

```python
from data_juicer.core.tracer.ray_tracer import RayTracer

# RayTracer 是一个 Ray Actor — 通过 Ray 创建
tracer = RayTracer.remote(
    work_dir="./outputs",
    op_list_to_trace=None,  # 追踪所有算子
    show_num=10,
    trace_keys=["sample_id"]
)

# 远程方法调用
ray.get(tracer.collect_mapper_sample.remote(
    op_name="clean_email_mapper",
    original_sample={"text": "邮箱：a@b.com"},
    processed_sample={"text": "邮箱："},
    text_key="text"
))

# 最终化并导出所有追踪结果
ray.get(tracer.finalize_traces.remote())
```

### 辅助函数

`data_juicer.core.tracer` 模块提供了模式无关的辅助函数：

```python
from data_juicer.core.tracer import (
    should_trace_op,
    check_tracer_collect_complete,
    collect_for_mapper,
    collect_for_filter,
)

# 这些函数自动处理默认模式和 Ray 模式
should_trace_op(tracer_instance, "clean_email_mapper")
check_tracer_collect_complete(tracer_instance, "clean_email_mapper")
collect_for_mapper(tracer_instance, "op_name", original, processed, "text")
collect_for_filter(tracer_instance, "op_name", sample, should_keep=False)
```

## 性能考虑

### 开销

- 当 `trace_num` 较小时（默认：10），追踪的额外开销极小
- 一旦某个算子收集了 `trace_num` 个样本，就不再进行进一步收集
- 主要成本是 Mapper 中原始文本与处理后文本的比较

### 建议

| 场景 | 推荐设置 |
|------|----------|
| 开发/调试 | `open_tracer: true`，`trace_num: 10-50` |
| 生产运行 | `open_tracer: false` |
| 审计特定算子 | `open_tracer: true`，`op_list_to_trace: [特定算子]` |
| 大规模追踪 | `open_tracer: true`，`trace_num: 100`，指定 `op_list_to_trace` |

## 故障排除

**没有生成追踪文件：**
```bash
# 验证追踪器是否启用
grep "open_tracer" config.yaml

# 检查追踪目录是否存在
ls -la ./outputs/{work_dir}/trace/
```

**追踪文件为空：**
- 对于 Mapper：算子可能没有修改任何样本
- 对于 Filter：算子可能没有过滤掉任何样本
- 检查日志中是否有类似 "Datasets before and after op [X] are all the same" 的警告

**追踪文件中样本太少：**
- 增加 `trace_num` 以收集更多样本
- 数据集中变化/过滤的样本可能少于 `trace_num`
