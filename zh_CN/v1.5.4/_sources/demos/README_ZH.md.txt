# 演示

此文件夹包含一些演示样例，帮助用户轻松体验 Data-Juicer 的各种功能和工具。

## 用法

使用 `demos` 子目录下的 `app.py` 来执行演示样例。

```shell
cd <subdir_of_demos>
streamlit run app.py
```

## 可用的演示

- 数据集样例 (`data`)
  - 该文件夹包含一些样例数据集。

- 初探索 (`overview_scan`)
  - 该示例介绍了 Data-Juicer 的基本概念和功能，例如特性、配置系统，算子等等。

- 数据处理回路 (`data_process_loop`)
  - 该示例用来分析和处理数据集，并给出处理前后数据集的统计信息比对。

- 词法多样性可视化 (`data_visualization_diversity`)
  - 该示例可以用来分析 CFT 数据集的动词-名词结构，并绘制成sunburst层级环形图表。

- 算子效果可视化 (`data_visualization_op_effect`)
  - 该示例可以分析数据集的统计信息，并根据这些统计信息可以显示出每个 `Filter` 算子在不同阈值下的效果。

- 统计信息可视化 (`data_visualization_statistics`)
  - 该示例可以分析数据集，并获得多达13种统计信息。

- 处理 CFT 中文数据 (`process_cft_zh_data`)
  - 以 Alpaca-CoT 的部分中文数据为例，演示了 LLM 中指令跟随微调数据和有监督微调数据的分析和处理流程。

- 处理预训练科学文献类数据 (`process_sci_data`)
  - 以 arXiv 的部分数据为例，演示了如何处理 LLM 预训练中的科学文献类数据的分析和处理流程。

- 处理预训练代码类数据 (`process_code_data`)
  - 以 Stack-Exchange 的部分数据为例，演示了如何处理 LLM 预训练中的代码类数据的分析和处理流程。

- 文本质量打分器 (`tool_quality_classifier`)
  - 该示例提供了3种文本质量打分器，对数据集进行打分评估。

- 按语言分割数据集 (`tool_dataset_splitting_by_language`)
  - 该示例按照语言将数据集拆分为不同的子数据集。

- 数据混合 (`data_mixture`)
  - 该示例从多份数据集中进行采样并混合为一个新的数据集。

- 分区和检查点 (`partition_and_checkpoint`)
  - 该演示展示了带分区、检查点和事件日志的分布式处理。它演示了新的作业管理功能，包括资源感知分区、全面的事件日志记录和处理快照工具，用于监控作业进度。
