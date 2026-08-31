# 快速上手

本指南带你完成一次完整的 Data-Juicer 数据处理：安装、编写菜谱、运行流水线、查看输出。

> **提示：** 部分算子在首次使用时会下载模型权重（例如 `language_id_score_filter` 会下载 fastText 语言识别模型）。首次运行可能多等几分钟，后续运行可以复用本地模型缓存。

---

## 1. 安装 Data-Juicer

先按[安装文档](Installation_ZH.md)准备 Python、Git 和 uv。以下命令使用 Linux/macOS shell。

克隆仓库并从源码安装。本指南会用到仓库中的示例菜谱和样例数据：

```bash
git clone https://github.com/datajuicer/data-juicer.git --depth 1
cd data-juicer
uv venv
source .venv/bin/activate  # Linux / macOS
uv pip install -e ".[nlp]"
```

验证 CLI 可用：

```bash
dj-process --help
```

> **提示：** 如果不需要示例文件，`uv pip install "py-data-juicer[nlp]"` 可直接从 PyPI 安装相同的包。完整的安装方式（场景化 extras、Docker 等）请参见[安装文档](Installation_ZH.md)。

---

## 2. 了解输入数据

Data-Juicer 开箱支持 JSONL、Parquet、CSV/TSV、纯文本等多种格式。本指南使用内置的样例数据集 [`demos/data/demo-dataset.jsonl`](https://github.com/datajuicer/data-juicer/blob/main/demos/data/demo-dataset.jsonl)，内容如下：

```json
{"text": "Today is Sunday and it's a happy day!", "meta": {"src": "Arxiv"}}
{"text": "Do you need a cup of coffee?", "meta": {"src": "code"}}
{"text": "你好，请问你是谁", "meta": {"src": "customized"}}
{"text": "Sur la plateforme MT4, plusieurs manières...", "meta": {"src": "Oscar"}}
{"text": "欢迎来到阿里巴巴！", "meta": {"src": "customized"}}
{"text": "This paper proposed a novel method on LLM pretraining.", "meta": {"src": "customized"}}
```

每行一个 JSON 对象，文本默认在 `"text"` 字段，其余字段作为元数据保留。

> 输入格式、数据混合、远程数据集（如 Hugging Face）等高级配置请参见[数据集配置指南](../DatasetCfg_ZH.md)。对于 arXiv tar 包、Stack Exchange 7z 等需要额外解压/转换的原始数据，可使用[预处理工具](../../tools/preprocess/README_ZH.md)将其转为 Data-Juicer 可直接读取的格式。

---

## 3. 编写菜谱

**菜谱**是一个 YAML 配置文件，声明运行哪些算子、以什么顺序运行。这里使用内置的示例菜谱 [`demos/process_simple/process.yaml`](https://github.com/datajuicer/data-juicer/blob/main/demos/process_simple/process.yaml)：

```yaml
# 全局参数
project_name: 'demo-process'
dataset_path: './demos/data/demo-dataset.jsonl'
np: 4

export_path: './outputs/demo-process/demo-processed.jsonl'

# 要应用的算子
process:
  - language_id_score_filter:
      lang: 'zh'
      min_score: 0.8
```

`process` 下的每项是一个算子，按列出顺序依次执行——前一个算子的输出是后一个的输入。这个菜谱只保留语言置信度较高的中文样本。

> 菜谱的完整语法请参见[全局配置参数速查](../GlobalConfig_ZH.md)。200+ 内置算子的完整列表请参见[算子库](../Operators.md)。

---

## 4. 运行流水线

将菜谱传给 `dj-process` CLI：

```bash
dj-process --config demos/process_simple/process.yaml
```

Data-Juicer 加载数据集，按顺序执行每个算子，将过滤后的结果写入 `export_path`。运行时会在控制台打印各算子的处理统计。

命令行可以覆盖菜谱中的任何参数，无需修改 YAML：

```bash
dj-process --config demos/process_simple/process.yaml --language_id_score_filter.lang=en \
  --export_path ./outputs/demo-process/demo-processed-en.jsonl
```

> 英文结果保存在 `demo-processed-en.jsonl`，下一节查看中文结果文件 `demo-processed.jsonl`。其他菜谱可通过 `dj-install --config your-recipe.yaml` 预装算子源码中识别到的依赖，详见[安装文档](Installation_ZH.md)。

---

## 5. 查看输出

处理后的数据集在你配置的 `export_path` 路径：

```bash
cat ./outputs/demo-process/demo-processed.jsonl
```

你应该只看到通过过滤器的中文样本——英文和法文行已被移除。

想在正式运行前了解数据集的质量分布？使用分析器：

```bash
dj-analyze --config demos/process_simple/process.yaml
```

> 分析器的完整用法（自动模式、分布式分析、自定义指标）请参见[数据分析指南](../AnalyzeData_ZH.md)。交互式拖动滑块调优过滤阈值请参见 [Web Playground](../Playground_ZH.md)。

---

## 6. 在 Python 中使用（可选）

如果你想在训练脚本或 Notebook 中嵌入 Data-Juicer：

```python
from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
executor = DefaultExecutor(cfg)
executor.run()
```

也支持单算子链式调用：

```python
dataset = dataset.process([op1, op2])
```

> 编程接口的完整用法请参见[处理数据指南](../ProcessData_ZH.md)。

---

## 下一步

你已经跑通了第一条流水线。根据需要探索以下方向：

| 方向 | 说明 | 链接 |
|------|------|------|
| **处理数据** | CLI 与 Python API 完整用法、性能调优 | [处理数据指南](../ProcessData_ZH.md) |
| **数据分析** | 运行前用分析器了解数据分布 | [数据分析指南](../AnalyzeData_ZH.md) |
| **算子库** | 浏览 200+ 算子，覆盖文本/图像/音频/视频 | [算子总览](../Operators.md) |
| **可视化调优** | 拖动滑块调整过滤阈值，即时预览效果 | [Web Playground](../Playground_ZH.md) |
| **分布式处理** | 使用 Ray 扩展到多机集群 | [分布式处理](../Distributed_ZH.md) |
| **数据沙盒** | 小数据快速实验，数据-模型协同优化闭环 | [DJ-Sandbox](https://datajuicer.github.io/data-juicer-sandbox/zh_CN/main/index_ZH.html) |
| **导出与缓存** | 控制输出格式、加速重复运行 | [导出](../Export_ZH.md) · [缓存](../Cache_ZH.md) |
| **自定义算子** | 编写自己的算子并贡献代码 | [开发者指南](../DeveloperGuide_ZH.md) |
| **DJ-Cookbook** | 社区菜谱合集与教程资源 | [DJ-Cookbook](DJ-Cookbook_ZH.md) |
