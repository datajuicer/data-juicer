# 全局配置参数速查

本页列出 Data-Juicer 菜谱中常用全局参数及其默认值。这些参数在菜谱 YAML 顶层设置，也可通过命令行 `--参数名 值` 覆盖。完整参数列表请运行 `dj-process --help` 查看。

> 算子参数不在此列——请参考[算子提要](Operators.md)或各算子详情页。

各参数的说明包含适用的执行器、工具及所需的功能开关。

---

## 项目与路径

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `project_name` | str | `hello_world` | 项目名称，用于标识输出目录和日志 |
| `dataset_path` | str | `""` | 输入数据集路径，支持权重混合格式：`<w1> path1 <w2> path2` |
| `dataset` | list/dict | `[]` | 高级数据集配置（本地/远程），详见[数据集配置](DatasetCfg_ZH.md) |
| `export_path` | str | `./outputs/hello_world/hello_world.jsonl` | 输出文件路径 |
| `work_dir` | str | `None` | 工作目录基路径（默认取 export_path 的父目录）；解析后追加 job_id |
| `temp_dir` | str | `None` | 临时文件目录（禁用缓存时使用） |

---

## 执行引擎

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `executor_type` | str | `default` | 执行引擎：`default`（本地多进程）/ `ray` / `ray_partitioned` |
| `np` | int | `4` | 并行 worker 进程数 |
| `ray_address` | str | `"auto"` | Ray 集群地址（仅 ray 模式） |

---

## 输入与格式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text_keys` | str/list | `"text"` | 文本字段名 |
| `image_key` | str | `"images"` | 图像路径列表字段名 |
| `audio_key` | str | `"audios"` | 音频路径列表字段名 |
| `video_key` | str | `"videos"` | 视频路径列表字段名 |
| `suffixes` | str/list | `[]` | 限制加载的文件后缀（空=自动检测） |
| `load_dataset_kwargs` | dict | `{}` | 传递给 `datasets.load_dataset()` 的额外参数 |
| `read_options` | dict | `{}` | `ray` 执行器的 PyArrow 读取选项 |

使用默认执行器处理少量输入时，可设置 `dataset.max_sample_num` 或提前准备子集。参见[数据采样试跑](ProcessData_ZH.md)。

### 分析与工具专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_probe_ratio` | float | `1.0` | Sandbox 模型推理探测的抽样比例，传给 `sample_data()` |
| `data_probe_algo` | str | `uniform` | Sandbox 模型推理探测的抽样算法 |
| `hpo_config` | str | `None` | [HPO 工具](../data_juicer/tools/hpo/README_ZH.md)的搜索空间配置 |
| `auto_num` | int | `1000` | `dj-analyze --auto` 的分析样本数上限 |

[Data-Juicer Sandbox](https://github.com/datajuicer/data-juicer-sandbox) 在模型推理探测中使用上述抽样参数。通过 Python API 抽样时，可调用 `executor.sample_data(sample_ratio=cfg.data_probe_ratio, sample_algo=cfg.data_probe_algo)`，再处理返回的子集。

---

## 导出

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `export_type` | str | `None` | 导出格式（省略时从 export_path 后缀推断） |
| `export_shard_size` | int | `0` | 分片大小（字节），0 = 单文件 |
| `export_in_parallel` | bool | `false` | 并行导出到单文件 |
| `export_extra_args` | dict | `{}` | 格式特定的额外参数 |
| `export_aws_credentials` | dict | `null` | S3 导出凭证 |
| `keep_stats_in_res_ds` | bool | `false` | 输出中保留算子计算的统计字段 |
| `keep_hashes_in_res_ds` | bool | `false` | 输出中保留去重计算的哈希字段 |

详见[导出文档](Export_ZH.md)。

---

## 性能优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `op_fusion` | bool | `false` | 在 default 或 Ray 执行器中融合兼容算子，减少重复处理 |
| `fusion_strategy` | str | `probe` | 融合策略：`probe` 按分组和探测速度安排顺序；`greedy` 按融合分组安排顺序。测速适用于 default 执行器和普通 Analyzer |
| `mapper_fusion` | bool | `true` | 融合连续 GPU Mapper（需 op_fusion 开启） |
| `mapper_fusion_vram_limit` | float | `0.9` | 融合 Mapper 聚合显存上限 |
| `adaptive_batch_size` | bool | `false` | `default` 执行器中批处理算子的自适应批大小 |
| `turbo` | bool | `false` | Turbo 模式（batch_size=1 时最大化速度） |

---

## 缓存与检查点

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_cache` | bool | `true` | 使用 HuggingFace datasets 缓存 |
| `ds_cache_dir` | str | `None` | 自定义缓存目录（覆盖 `HF_DATASETS_CACHE`） |
| `cache_compress` | str | `None` | 缓存压缩：`gzip` / `zstd` / `lz4` |
| `use_checkpoint` | bool | `false` | 启用 default 执行器检查点（自动禁用 cache，与 op_fusion 互斥） |

### ray_partitioned 检查点

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint.enabled` | bool | `true` | 启用分区检查点 |
| `checkpoint.strategy` | str | `every_n_ops` | 策略：`every_op` / `every_partition` / `every_n_ops` / `manual` / `disabled` |
| `checkpoint.n_ops` | int | `5` | `every_n_ops` 策略的间隔 |
| `checkpoint.op_names` | list | `[]` | `manual` 策略下检查点的算子名列表 |

---

## 任务管理与恢复

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `job_id` | str | `None` | 自定义 job ID（用于恢复和追踪） |
| `resume` | str | `None` | 恢复指定 job ID 的任务（仅 ray_partitioned） |
| `event_logging.enabled` | bool | `true` | 启用事件日志 |
| `event_log_dir` | str | `None` | 事件日志目录（推荐快速存储） |
| `checkpoint_dir` | str | `None` | `ray_partitioned` 执行器的检查点目录；default 执行器的检查点位于 `<work_dir>/ckpt` |

---

## 追踪与监控

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `open_tracer` | bool | `false` | 启用样本追踪（记录算子前后变化） |
| `op_list_to_trace` | list | `[]` | 需追踪的算子名（空=全部） |
| `trace_num` | int | `10` | 每算子展示的变化样本数 |
| `trace_keys` | list | `[]` | 追踪的字段名列表 |
| `open_monitor` | bool | `false` | 启用资源监控（CPU/内存/GPU） |
| `open_insight_mining` | bool | `false` | 启用算子洞察挖掘（统计/标签变化追踪） |
| `op_list_to_mine` | list | `[]` | 参与洞察挖掘的算子（空=全部产出统计的算子） |

---

## 预留配置字段

下列字段为预留项，内置执行器尚未使用。日志输出配置见[作业管理](JobManagement_ZH.md)，分区文件管理见[中间存储](PartitionAndCheckpoint_ZH.md)。

| 字段 | 类别 |
|------|------|
| `max_log_size_mb`、`backup_count` | 日志轮转与保留 |
| `intermediate_storage.*`、`preserve_intermediate_data` | 中间存储 |
| `resource_optimization.auto_configure` | 资源配置 |

---

## 加密

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `decrypt_after_reading` | bool | `false` | 读取时自动解密输入文件 |
| `encrypt_before_export` | bool | `false` | 导出时自动加密输出文件 |
| `encryption_key_path` | str | `None` | Fernet 密钥文件路径（或环境变量 `DJ_ENCRYPTION_KEY`） |

---

## 容错

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_op_error` | bool | `true` | 跳过算子中因异常样本引发的错误 |

---

## 多模态特殊 Token

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_special_token` | `<__dj__image>` | 文本中代表图像的占位符 |
| `audio_special_token` | `<__dj__audio>` | 文本中代表音频的占位符 |
| `video_special_token` | `<__dj__video>` | 文本中代表视频的占位符 |
| `eoc_special_token` | `<\|__dj__eoc\|>` | 文本中 chunk 结束标记 |

---

## 算子运行环境管理（仅 Ray 模式）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_common_dep_num_to_combine` | int | `-1` | 合并算子运行环境的最小公共依赖数（-1=不合并） |
| `conflict_resolve_strategy` | str | `split` | 依赖冲突策略：`split` / `overwrite` / `latest` |
