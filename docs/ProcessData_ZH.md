# 数据处理

本指南覆盖使用 Data-Juicer 运行数据处理流水线的完整流程——CLI 与 Python API 两种方式。

> 如果你还没跑过第一条流水线，建议先看[快速上手](tutorial/QuickStart_ZH.md)。
> 完整的全局参数列表请参见[全局配置参数速查](GlobalConfig_ZH.md)。

---

## CLI 方式

### 基本用法

```bash
dj-process --config my-recipe.yaml
```

Data-Juicer 读取菜谱文件，默认按 `process` 列表的顺序依次执行算子，将结果写入 `export_path`。

### 命令行覆盖

任何菜谱中的参数都可以在命令行直接覆盖，无需修改 YAML：

```bash
dj-process --config recipe.yaml --np 8 --export_path ./out/result.parquet
dj-process --config recipe.yaml --language_id_score_filter.lang=en
```

### 自动安装算子依赖

```bash
dj-install --config my-recipe.yaml
```

工具根据菜谱中的算子列表扫描源码，并预装识别到的依赖。扫描范围和运行环境准备步骤见[安装文档](tutorial/Installation_ZH.md)。

---

## Python API 方式

Python API 提供了比 YAML 菜谱更灵活的控制——适合在训练脚本、Notebook 或自动化流水线中嵌入数据处理。

### 方式一：加载菜谱

```python
from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
executor = DefaultExecutor(cfg)
dataset = executor.run()
```

如果你已经在内存中持有 dataset 对象（比如从上游拼接或采样得到），可以跳过菜谱中的数据源定义，只复用其算子流水线：

```python
dataset = executor.run(dataset=my_dataset, skip_export=True)
```

### 方式二：直接实例化算子

不写 YAML，直接在 Python 中组装算子链：

```python
from data_juicer.ops import load_ops
from data_juicer.core import NestedDataset

# 从字典配置加载算子（与 YAML process 列表格式一致）
ops = load_ops([
    {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
    {'text_length_filter': {'min_len': 10, 'max_len': 50000}},
    {'document_minhash_deduplicator': {'tokenization': 'space', 'window_size': 5}},
])

# 加载数据集
dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))

# 链式处理
dataset = dataset.process(ops)
```

### 方式三：精确控制单个算子

当你需要条件分支、循环或中间检查时：

```python
from data_juicer.ops.filter import LanguageIDScoreFilter, TextLengthFilter
from data_juicer.ops.deduplicator import DocumentMinhashDeduplicator
from data_juicer.core import NestedDataset

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))

# 第一步：语言过滤
lang_filter = LanguageIDScoreFilter(lang='zh', min_score=0.8)
dataset = lang_filter.run(dataset=dataset)

print(f"语言过滤后: {len(dataset)} 条")

# 第二步：条件去重——仅当数据量超过阈值时执行
if len(dataset) > 10000:
    dedup = DocumentMinhashDeduplicator(tokenization='space', window_size=5)
    dataset = dedup.run(dataset=dataset)
    print(f"去重后: {len(dataset)} 条")

# 第三步：长度过滤
length_filter = TextLengthFilter(min_len=10, max_len=50000)
dataset = length_filter.run(dataset=dataset)
```

### 方式四：动态组合算子

根据数据特征动态选择算子——适合编程式批处理或自动化工作流：

```python
from data_juicer.ops import load_ops
from data_juicer.core import NestedDataset

dataset = NestedDataset(NestedDataset.from_json('input.jsonl'))

# 根据数据特征决定处理策略
sample = dataset[0]
ops_config = []

# 如果有多语言数据，加语言过滤
if any(key in sample for key in ['text']):
    ops_config.append({'language_id_score_filter': {'lang': 'zh', 'min_score': 0.5}})

# 如果有图像字段，加图像尺寸过滤
if 'images' in sample and sample['images']:
    ops_config.append({'image_shape_filter': {'min_width': 256, 'min_height': 256}})

# 统一清洗
ops_config.append({'clean_html_mapper': {}})
ops_config.append({'text_length_filter': {'min_len': 10}})

ops = load_ops(ops_config)
dataset = dataset.process(ops)
```

---

## 算子执行顺序最佳实践

默认执行流程按照 `process` 列表**从上到下串行执行**。顺序影响性能和结果：

1. **低成本过滤先行**：文本长度、语言识别等轻量算子尽早减少数据量
2. **去重放中段**：去重需全局状态，在初步过滤后、精细处理前执行
3. **高成本算子靠后**：GPU 推理类算子只处理已过滤的子集

```yaml
process:
  # 低成本
  - text_length_filter: { min_len: 10, max_len: 50000 }
  - language_id_score_filter: { lang: zh, min_score: 0.5 }
  # 去重
  - document_minhash_deduplicator: { tokenization: space, window_size: 5 }
  # 高成本
  - clean_html_mapper: {}
  - perplexity_filter: { lang: en, max_ppl: 1500 }
```

---

## 性能调优

### 算子融合

融合兼容算子以减少重复处理；`default` 和 Ray 执行路径均支持算子融合。实际吞吐收益取决于菜谱与数据，需要实测：

```yaml
op_fusion: true
fusion_strategy: probe   # probe：分组并按探测速度排序；greedy：按融合组安排顺序
```

融合会按兼容性分组，并可能调整菜谱中的算子顺序。`default` 执行器和普通 Analyzer 使用 `probe` 时，默认取当前数据集的前 1,000 条测速，不足 1,000 条则全部使用。每个算子将这批样本复制为其实际进程数对应的份数，再执行测速。Ray 执行器按融合分组安排算子顺序。

### GPU Mapper 融合

多个连续 GPU Mapper 融合为一次 GPU 调用：

```yaml
op_fusion: true
mapper_fusion: true
adaptive_batch_size: true
```

`default` 执行器通过 `adaptive_batch_size` 探测并调整批处理算子的批大小。

### 数据采样试跑

源数据至少有 1,000 条时，使用默认执行器取 1,000 条样本验证菜谱：将原菜谱中的 `dataset_path` 替换为结构化的 `dataset` 配置，并保留其 `process` 列表。

```yaml
dataset:
  max_sample_num: 1000
  configs:
    - type: local
      path: path/to/your/dataset.jsonl
```

将路径替换为自己的数据集。`max_sample_num` 指定采样条数；小样本试跑时，将它设为不超过源数据条数的值。预算超过源数据条数时，会重复采样以补足预算。

采样在数据加载完成后执行。需要同时减少读取量时，可预先生成小样本文件并用它作为输入。正式处理全部数据时，移除 `max_sample_num`。

---

## 检查点与断点续跑

```yaml
use_checkpoint: true
```

默认执行器通过 `job_id` 和工作目录基路径定位检查点。首次运行时指定一个固定 ID，中断后保持输入、菜谱和工作目录配置一致，重复运行该命令即可恢复：

```bash
dj-process --config your-recipe.yaml --use_checkpoint true --job_id recipe-checkpoint
```

断点位于解析后的 `<cfg.work_dir>/ckpt`，其中 `work_dir` 包含 `job_id`。`use_checkpoint` 会禁用数据缓存，并与 `op_fusion` 互斥。

对于 `ray_partitioned` 模式有更精细的策略，失败后可通过 `--resume <job_id>` 恢复。详见[全局配置](GlobalConfig_ZH.md)。

---

## 追踪与调试

```yaml
open_tracer: true
trace_num: 10
```

Tracer 在工作目录输出每个算子的 before/after 对比——帮助理解算子行为。详见[追踪文档](Tracing_ZH.md)。

---

## 下一步

- [数据分析](AnalyzeData_ZH.md)——运行前先了解数据分布
- [数据集配置](DatasetCfg_ZH.md)——输入格式、数据混合、远程数据集
- [全局配置参数速查](GlobalConfig_ZH.md)——所有参数的完整列表
- [分布式处理](Distributed_ZH.md)——使用 Ray 扩展到集群
- [算子库](Operators.md)——浏览 200+ 可用算子
