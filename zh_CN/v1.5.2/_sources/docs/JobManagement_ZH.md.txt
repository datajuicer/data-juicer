# 作业管理

DataJuicer 提供用于监控和管理处理作业的工具。

## 处理快照

从事件日志和 DAG 结构分析作业状态。

```bash
# JSON 输出
python -m data_juicer.utils.job.snapshot /path/to/job_dir

# 人类可读输出
python -m data_juicer.utils.job.snapshot /path/to/job_dir --human-readable
```

输出包括：
- 作业状态和进度百分比
- 分区完成计数
- 操作指标
- 检查点覆盖率
- 时间信息

## 资源感知分区

系统根据集群资源和数据特征自动优化分区大小。

```yaml
partition:
  mode: "auto"
  target_size_mb: 256  # 目标分区大小（可配置）
```

优化器会：
1. 检测 CPU、内存和 GPU 资源
2. 采样数据以确定模态和内存使用
3. 计算目标为配置大小的分区（默认 256MB）
4. 确定最佳工作节点数量

## 日志

日志按作业组织，支持轮转和保留：

```
{job_dir}/
├── events_{timestamp}.jsonl   # 机器可读事件
├── logs/
│   ├── log.txt                # 主日志
│   ├── log_DEBUG.txt          # 调试日志
│   ├── log_ERROR.txt          # 错误日志
│   └── log_WARNING.txt        # 警告日志
└── job_summary.json           # 摘要（完成时）
```

配置日志：
```python
from data_juicer.utils.logger_utils import setup_logger

setup_logger(
    save_dir="./outputs",
    filename="log.txt",
    max_log_size_mb=100,
    backup_count=5
)
```

## API 参考

### ProcessingSnapshotAnalyzer

```python
from data_juicer.utils.job.snapshot import ProcessingSnapshotAnalyzer

analyzer = ProcessingSnapshotAnalyzer(job_dir)
snapshot = analyzer.generate_snapshot()
json_data = analyzer.to_json_dict(snapshot)
```

### ResourceDetector

```python
from data_juicer.core.executor.partition_size_optimizer import ResourceDetector

local = ResourceDetector.detect_local_resources()
cluster = ResourceDetector.detect_ray_cluster()
workers = ResourceDetector.calculate_optimal_worker_count()
```

### PartitionSizeOptimizer

```python
from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

optimizer = PartitionSizeOptimizer(cfg)
recommendations = optimizer.get_partition_recommendations(dataset, pipeline)
```

## 故障排除

检查作业状态：
```bash
python -m data_juicer.utils.job.snapshot /path/to/job
```

分析事件：
```bash
cat /path/to/job/events_*.jsonl | head -20
```

检查资源：
```python
from data_juicer.core.executor.partition_size_optimizer import ResourceDetector
print(ResourceDetector.detect_local_resources())
```
