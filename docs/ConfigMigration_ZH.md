# 全局配置迁移说明

## 已移除的配置

以下配置在内置执行器中没有运行效果，现已从配置解析器移除。升级时请从 YAML 和命令行中删除；包含这些键的旧配置会在校验阶段报错。

| 已移除配置 | 当前配置或行为 |
| --- | --- |
| `intermediate_storage.*`（全部八个字段） | `ray_partitioned` 检查点使用 Parquet 和写入器的默认压缩设置；通过 `checkpoint.enabled`、`checkpoint.strategy`、`checkpoint.n_ops`、`checkpoint.op_names` 控制保存时机。 |
| `preserve_intermediate_data` | 退出执行器运行上下文时清理临时目录；检查点单独保存，用于续跑。 |
| `partition_size`、`max_partition_size_mb` | 手动分区使用 `partition.mode: manual` 和 `partition.num_of_partitions`；自动规划使用 `partition.mode: auto` 和 `partition.target_size_mb`。新配置分别表达分区数量和规划目标，旧数值不能直接照搬。 |
| `resource_optimization.auto_configure` | 使用 `partition.mode: auto` 开启自动分区规划。 |
| `max_log_size_mb`、`backup_count` | 内置日志器没有可配置的文件大小轮转和备份数量策略。`setup_logger()` 接受 `save_dir`、`filename`、`level` 等实际声明的参数。 |

删除这些配置不会改变现有检查点保存、续跑和临时目录清理行为。保留任意中间文件、选择中间文件格式和压缩方式、按天数或任务结果保留文件，目前没有对应的新开关。

历史嵌套字段 `partition.size` 和 `partition.max_size_mb` 同样不受 YAML/CLI 支持，本次也移除了执行器中残留的读取和回退属性。手动模式使用 `partition.num_of_partitions`；自动模式依据优化器建议和集群资源计算分区数量。优化失败或返回无效的建议样本数时，保留配置的分区数量，再应用集群约束。

`checkpoint.strategy: every_partition` 也会在解析时被拒绝：此前解析器接受此值，但执行器将其回退为 `every_op`。支持的策略为 `every_op`、`every_n_ops`、`manual` 和 `disabled`。`checkpoint.n_ops`、`partition.target_size_mb` 和非空的 `override_num_blocks` 必须为正整数。

## 可配置的现有功能

`load_jsonl_lenient` 现可通过 YAML 和 CLI 设置，在默认执行器和 Analyzer 中启用已有的宽松 JSONL 加载器，跳过损坏行。环境变量 `DATA_JUICER_JSONL_LENIENT=1` 也可启用该功能。支持的文件类型见[数据集配置](DatasetCfg_ZH.md)。

`use_dag` 控制执行计划生成和 DAG 监控。默认 `null` 保持执行器的原有行为：`ray`、`ray_partitioned` 开启，`default` 关闭。可用 `true` 或 `false` 覆盖。

## 读取选项

数据处理、分析和直接调用 `DatasetBuilder` 统一应用全局读取选项：

- `load_dataset_kwargs` 为默认执行器和 Analyzer 提供 HuggingFace 读取默认参数，例如 Parquet 的 `columns`、CSV 的 `delimiter`。
- `read_options` 配置 Ray 路径中的 PyArrow JSON 读取，包括本地 JSON 输入；`override_num_blocks` 指定 Ray 读取请求的 block 数。
- 显式传给 `DatasetBuilder.load_dataset(...)` 的同名参数优先于全局默认值。`generated_dataset_config` 继续使用其自身的 formatter 构造参数。

`data_probe_algo`、`data_probe_ratio` 保留供外部 Data-Juicer Sandbox 的模型探测使用；`hpo_config` 保留供 HPO 工具使用。

自动分区分析现使用 `text_keys` 的第一个字段，支持嵌套路径；空列表表示不将文本计入模态和文本长度分析。全局配置示例中的通知、标注配置说明也已改为实际实现使用的算子参数位置。

## 程序化优化器接口

`ModalityConfig` 现保留模态、回退样本数、建议样本数上限和描述。已移除不参与计算的 `max_partition_size_mb`、`memory_multiplier`、`complexity_multiplier`。优化器根据实际算子计算处理复杂度的逻辑保持不变。

`get_partition_recommendations()` 继续返回计算建议和分析信息，其中 `modality_configs` 的各项保留 `default_size`、`max_size`、`description`，移除过时的 `max_size_mb`。直接构造 `ModalityConfig`、读取已移除属性或返回键的代码需要调整。计算得到的 `recommended_max_size_mb` 估算值仍保留，用户侧的规划目标仍为 `partition.target_size_mb`。
