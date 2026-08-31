# 数据分析

在确定过滤阈值之前，通常需要先了解数据集的统计概况。`dj-analyze` 会计算算子产出的所有统计量的分布和相关性，帮助你做出数据驱动的阈值决策。

> 完整的全局参数列表请参见[全局配置参数速查](GlobalConfig_ZH.md)。

---

## CLI 方式

### 基本用法

对已有菜谱运行分析器：

```bash
dj-analyze --config path/to/your-recipe.yaml
```

### 自动模式

无需编写专门的分析菜谱——自动使用全部可产出统计信息的 Filter 来分析数据集子集：

```bash
dj-analyze --auto --dataset_path your-dataset.jsonl --auto_num 1000
```

- `--auto_num`：采样分析的样本数量，默认 1000。适合快速了解数据分布。

---

## Python API 方式

### 基本用法：加载配置运行

```python
from data_juicer.config import init_configs
from data_juicer.core import Analyzer

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
analyzer = Analyzer(cfg)
dataset = analyzer.run()

# 分析结果在 analyzer.overall_result 中
print(analyzer.overall_result)
```

### 对已有 dataset 分析

将 Hugging Face Dataset 包装为 `NestedDataset`，即可使用 Analyzer 的数据处理接口：

```python
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs

cfg = init_configs(args=[
    '--config', 'my-recipe.yaml',
    '--export_path', './analysis-output/stats.jsonl',
])
analyzer = Analyzer(cfg)

# 传入已加载的数据集
dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))
analyzed = analyzer.run(dataset=dataset)
```

### 在内存中使用统计量

调用 Filter 并设置 `reduce=False`，为每条数据计算统计量，保留全部输入行，并返回可供脚本继续使用的数据集。此调用的数据集和分析报告导出由调用方负责；启用数据集缓存时会使用磁盘缓存。

```python
from data_juicer.core import NestedDataset
from data_juicer.ops.filter import TextLengthFilter
from data_juicer.utils.constant import Fields, StatsKeys

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))
analyzed = TextLengthFilter().run(dataset=dataset, reduce=False)
stats = analyzed[Fields.stats]
avg_len = sum(s[StatsKeys.text_len] for s in stats) / len(stats) if len(stats) else 0
print(f"平均文本长度: {avg_len:.2f}")
```

### 手动构建分析流程

当你需要完全控制分析逻辑时，可以直接使用底层分析组件：

```python
from data_juicer.ops.filter import LanguageIDScoreFilter, TextLengthFilter
from data_juicer.analysis import OverallAnalysis, ColumnWiseAnalysis
from data_juicer.core import NestedDataset

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))

# 手动计算统计量（仅 compute_stats，不过滤）
filters = [
    TextLengthFilter(min_len=0, max_len=999999),
    LanguageIDScoreFilter(lang='zh', min_score=0.0),
]

for f in filters:
    dataset = f.run(dataset=dataset, reduce=False)  # 仅计算统计，不执行过滤

# 运行分析
output_dir = './my-analysis'
overall = OverallAnalysis(dataset, output_dir)
result = overall.analyze()
print(result)

column_wise = ColumnWiseAnalysis(dataset, output_dir, overall_result=result)
column_wise.analyze()
```

### 动态选择分析维度

根据数据模态编程式选择分析算子——适合自动化工作流：

```python
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs

dataset = NestedDataset(NestedDataset.from_json('input.jsonl'))
sample = dataset[0]

# 根据数据模态构建分析菜谱
process_config = []

# 文本统计
if 'text' in sample:
    process_config.extend([
        {'text_length_filter': {'min_len': 0, 'max_len': 999999}},
        {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.0}},
        {'alphanumeric_filter': {'min_ratio': 0.0}},
    ])

# 图像统计
if 'images' in sample and sample['images']:
    process_config.extend([
        {'image_shape_filter': {'min_width': 0, 'min_height': 0}},
        {'image_aspect_ratio_filter': {'min_ratio': 0.0, 'max_ratio': 999}},
    ])

cfg = init_configs(args=[
    '--auto',
    '--dataset_path', 'input.jsonl',
    '--export_path', './analysis/stats.jsonl',
], allow_auto=True)
cfg.process = process_config
# auto 模式最多分析 auto_num 条数据（默认 1000）。

analyzer = Analyzer(cfg)
analyzed = analyzer.run(dataset=dataset)
```

---

## 分析输出

分析器生成以下内容：

- **整体统计表**：各统计量的 count、mean、std、min、max
- **分布图表**：每个统计量的直方图
- **相关性分析**：统计量之间的相关性热力图

图表与总体统计表保存在 `analyzer.analysis_path`（即 `<cfg.work_dir>/analysis`）中；解析后的 `work_dir` 包含 `job_id`。统计数据集则由 `export_path` 决定导出位置。

`Analyzer.run(..., skip_export=True)` 会保存统计数据集，并跳过总体统计表和图表的导出。Analyzer 初始化时创建工作目录并备份配置。

---

## 哪些算子参与分析

Analyzer 只处理两类算子：
- **Filter 算子**中能在 `__dj__stats__` 字段产出统计信息的（大多数 Filter 都可以）
- **Tagging 算子**中能在 `__dj__meta__` 字段产出标签的

注册器标记：
- `NON_STATS_FILTERS`：不能产出统计信息的 Filter
- `TAGGING_OPS`：能产出标签的算子

---

## 分布式分析

配置 `executor_type: ray` 后自动使用 `RayAnalyzer`，通过 Ray 原生聚合算子计算整体统计：

```bash
dj-analyze --config demos/analyze_simple/ray_analyzer.yaml
```

> RayAnalyzer 不产出逐列分布图和相关性分析。详见[分布式处理](Distributed_ZH.md)。

---

## 字体设置

分析结果图表中如出现 "Glyph missing" 警告：

```bash
export ANALYZER_FONT="Heiti SC"  # 默认值，支持中文
```

---

## 下一步

- 根据分析结果调整阈值？使用 [Web Playground](Playground_ZH.md) 交互式调优
- 确认阈值后运行处理？参见[处理数据](ProcessData_ZH.md)
- 大规模分析？参见[分布式处理](Distributed_ZH.md)
