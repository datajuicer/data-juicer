# 作业管理与监控

Data-Juicer 提供全面的作业管理和监控功能，帮助您跟踪、分析和优化数据处理工作流。

## 概述

作业管理系统包括：

- **处理快照工具**：详细的作业状态和进度分析
- **资源感知分区**：分布式处理的自动优化
- **增强日志系统**：集中化日志管理，支持轮转和保留
- **作业监控工具**：处理作业的实时跟踪

## 处理快照工具

处理快照工具基于 `events.jsonl` 和 DAG 结构提供 Data-Juicer 作业处理状态的全面分析。

### 功能特性

- **JSON 输出**：机器可读格式，便于自动化和集成
- **进度跟踪**：详细的分区和操作进度
- **检查点分析**：检查点状态和可恢复性信息
- **时间分析**：从作业摘要或事件中获取精确时间
- **资源利用**：分区和操作级别的统计信息

### 使用方法

#### 基本快照
```bash
python -m data_juicer.utils.job.snapshot outputs/partition-checkpoint-eventlog/20250809_040053_a001de
```

#### 人类可读输出
```bash
python -m data_juicer.utils.job.snapshot outputs/partition-checkpoint-eventlog/20250809_040053_a001de --human-readable
```

### JSON 输出结构

```json
{
  "job_info": {
    "job_id": "20250809_040053_a001de",
    "executor_type": "ray_partitioned",
    "status": "completed",
    "config_file": ["configs/demo/partition-checkpoint-eventlog.yaml"],
    "work_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de",
    "resumption_command": "dj-process --config [Path_fr(...)] --job_id 20250809_040053_a001de",
    "error_message": null
  },
  "overall_status": "completed",
  "overall_progress": {
    "overall_percentage": 100.0,
    "partition_percentage": 100.0,
    "operation_percentage": 100.0
  },
  "timing": {
    "start_time": 1754712053.496651,
    "end_time": 1754712325.323669,
    "duration_seconds": 271.82701802253723,
    "duration_formatted": "4m 31s",
    "job_summary_duration": 271.82701802253723,
    "timing_source": "job_summary"
  },
  "progress_summary": {
    "total_partitions": 18,
    "completed_partitions": 18,
    "failed_partitions": 0,
    "in_progress_partitions": 0,
    "partition_progress_percentage": 100.0,
    "total_operations": 144,
    "completed_operations": 144,
    "failed_operations": 0,
    "checkpointed_operations": 0,
    "operation_progress_percentage": 100.0
  },
  "checkpointing": {
    "strategy": "every_op",
    "last_checkpoint_time": 1754712320.123456,
    "checkpointed_operations_count": 72,
    "resumable": true,
    "checkpoint_progress": {
      "percentage": 50.0,
      "checkpointed_operations": [...],
      "checkpoint_coverage": 0.5
    },
    "checkpoint_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/checkpoints"
  },
  "partition_progress": {
    "0": {
      "status": "completed",
      "sample_count": 20000,
      "creation_start_time": 1754712074.356004,
      "creation_end_time": 1754712074.366004,
      "processing_start_time": 1754712074.366004,
      "processing_end_time": 1754712074.456004,
      "current_operation": null,
      "completed_operations": ["clean_links_mapper", "clean_email_mapper", ...],
      "failed_operations": [],
      "checkpointed_operations": [],
      "error_message": null,
      "progress_percentage": 100.0
    }
  },
  "operation_progress": {
    "p0_op0_clean_links_mapper": {
      "operation_name": "clean_links_mapper",
      "operation_idx": 0,
      "status": "completed",
      "start_time": 1754712074.356004,
      "end_time": 1754712074.366004,
      "duration": 0.01,
      "input_rows": 20000,
      "output_rows": 19363,
      "checkpoint_time": null,
      "error_message": null,
      "progress_percentage": 100.0
    }
  },
  "file_paths": {
    "event_log_file": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/events.jsonl",
    "event_log_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/logs",
    "checkpoint_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/checkpoints",
    "metadata_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/metadata",
    "backed_up_config_path": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/partition-checkpoint-eventlog.yaml"
  },
  "metadata": {
    "snapshot_generated_at": "2025-08-09T13:33:54.770298",
    "events_analyzed": 367,
    "dag_plan_loaded": true,
    "job_summary_loaded": true,
    "job_summary_used": true
  }
}
```

## 资源感知分区

资源感知分区系统根据可用的集群资源和数据特征自动优化分区大小和工作节点数量。

### 功能特性

- **自动资源检测**：分析本地和集群资源
- **数据驱动优化**：采样数据以确定最佳分区大小
- **模态感知**：针对文本、图像、音频、视频和多模态数据的不同优化策略
- **64MB 目标**：优化分区以目标 64MB 每个分区
- **工作节点数量优化**：自动确定最佳 Ray 工作节点数量

### 配置

在配置中启用资源优化：

```yaml
# 资源优化配置
resource_optimization:
  auto_configure: true  # 启用自动优化

# 手动配置（当 auto_configure: false 时使用）
# partition:
#   size: 10000  # 每个分区的样本数
#   max_size_mb: 128  # 最大分区大小（MB）
# np: 2  # Ray 工作节点数量
```

### 优化过程

1. **资源检测**：分析 CPU、内存、GPU 和集群资源
2. **数据采样**：采样数据集以了解数据特征
3. **模态分析**：确定数据模态并应用适当的优化
4. **分区计算**：计算最佳分区大小，目标 64MB
5. **工作节点优化**：确定最佳 Ray 工作节点数量
6. **应用**：将优化应用到处理管道

## 增强日志系统

增强日志系统提供集中化日志管理，支持轮转和保留策略。

### 功能特性

- **集中化日志**：所有日志通过 `logger_utils.py` 管理
- **日志轮转**：基于文件大小的自动轮转
- **保留策略**：可配置的保留和清理
- **压缩**：轮转日志的自动压缩
- **多级别**：不同日志级别的单独日志文件

### 配置

```python
from data_juicer.utils.logger_utils import setup_logger

# 设置带轮转和保留的日志记录器
setup_logger(
    save_dir="./outputs",
    filename="log.txt",
    max_log_size_mb=100,  # 100MB 时轮转
    backup_count=5        # 保留 5 个备份文件
)
```

### 日志结构

```
outputs/
├── job_20250809_040053_a001de/
│   ├── events.jsonl          # 事件日志（JSONL 格式）
│   ├── logs/                 # 日志目录
│   │   ├── events.log        # 事件日志（人类可读）
│   │   ├── log.txt           # 主日志文件
│   │   ├── log_DEBUG.txt     # 调试级别日志
│   │   ├── log_ERROR.txt     # 错误级别日志
│   │   └── log_WARNING.txt   # 警告级别日志
│   ├── checkpoints/          # 检查点目录
│   ├── partitions/           # 分区目录
│   └── job_summary.json      # 作业摘要
```

## 作业管理工具

### 作业工具

```python
from data_juicer.utils.job import JobUtils, create_snapshot

# 创建作业工具
job_utils = JobUtils("./outputs")

# 列出运行中的作业
running_jobs = job_utils.list_running_jobs()

# 加载事件日志
events = job_utils.load_event_logs()

# 创建处理快照
snapshot = create_snapshot("./outputs/job_20250809_040053_a001de")
```

### 事件分析

系统跟踪各种事件类型：

- **作业事件**：`job_start`、`job_complete`
- **分区事件**：`partition_creation_start`、`partition_creation_complete`、`partition_start`、`partition_complete`、`partition_failed`
- **操作事件**：`op_start`、`op_complete`、`op_failed`
- **检查点事件**：`checkpoint_save`
- **DAG 事件**：`dag_build_start`、`dag_build_complete`、`dag_execution_plan_saved`

## 最佳实践

### 1. 启用资源优化

对于生产工作负载，始终启用资源优化：

```yaml
resource_optimization:
  auto_configure: true
```

### 2. 监控作业进度

使用快照工具监控长时间运行的作业：

```bash
# 检查作业状态
python -m data_juicer.utils.job.snapshot /path/to/job/directory

# 获取详细分析
python -m data_juicer.utils.job.snapshot /path/to/job/directory --human-readable
```

### 3. 配置日志

设置适当的日志轮转和保留：

```python
setup_logger(
    save_dir="./outputs",
    max_log_size_mb=100,
    backup_count=5
)
```

### 4. 使用检查点

为长时间运行的作业启用检查点：

```yaml
checkpoint:
  enabled: true
  strategy: "every_op"  # 或 "every_partition"、"every_n_ops"
```

### 5. 监控资源使用

快照工具提供详细的资源利用信息：

- 分区级别的进度和时间
- 操作级别的性能指标
- 检查点覆盖率和可恢复性
- 整体作业效率统计

## 集成示例

### 自动化脚本

```python
import json
import subprocess
from pathlib import Path

def monitor_job(job_dir: str):
    """监控 Data-Juicer 作业并返回状态。"""
    result = subprocess.run([
        "python", "-m", "data_juicer.utils.job.snapshot", job_dir
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        snapshot = json.loads(result.stdout)
        return {
            "status": snapshot["overall_status"],
            "progress": snapshot["overall_progress"]["overall_percentage"],
            "duration": snapshot["timing"]["duration_formatted"],
            "resumable": snapshot["checkpointing"]["resumable"]
        }
    else:
        return {"error": result.stderr}

# 使用
status = monitor_job("./outputs/job_20250809_040053_a001de")
print(f"作业状态: {status['status']}, 进度: {status['progress']:.1f}%")
```

### 仪表板集成

JSON 输出格式便于与监控仪表板集成：

```python
def get_job_metrics(job_dir: str):
    """提取仪表板显示的关键指标。"""
    snapshot = create_snapshot(job_dir)
    
    return {
        "job_id": snapshot.job_id,
        "status": snapshot.overall_status.value,
        "progress": {
            "partitions": f"{snapshot.completed_partitions}/{snapshot.total_partitions}",
            "operations": f"{snapshot.completed_operations}/{snapshot.total_operations}"
        },
        "timing": {
            "duration": snapshot.total_duration,
            "start_time": snapshot.job_start_time
        },
        "checkpointing": {
            "resumable": snapshot.resumable,
            "strategy": snapshot.checkpoint_strategy
        }
    }
```

## 故障排除

### 常见问题

1. **作业无法启动**：检查资源可用性和配置
2. **性能缓慢**：启用资源优化并检查分区大小
3. **内存问题**：减少分区大小或启用检查点
4. **日志文件增长**：配置日志轮转和保留策略

### 调试命令

```bash
# 检查作业状态
python -m data_juicer.utils.job.snapshot /path/to/job

# 分析事件
python -c "import json; events = [json.loads(line) for line in open('/path/to/job/events.jsonl')]; print(f'总事件数: {len(events)}')"

# 检查资源使用
python -c "from data_juicer.core.executor.partition_size_optimizer import ResourceDetector; print(ResourceDetector.detect_local_resources())"
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

# 检测本地资源
local_resources = ResourceDetector.detect_local_resources()

# 检测 Ray 集群
cluster_resources = ResourceDetector.detect_ray_cluster()

# 计算最佳工作节点数量
optimal_workers = ResourceDetector.calculate_optimal_worker_count()
```

### PartitionSizeOptimizer

```python
from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

optimizer = PartitionSizeOptimizer()
recommendations = optimizer.get_partition_recommendations(dataset, modality)
```

这个全面的作业管理系统提供了您有效监控、优化和故障排除 Data-Juicer 处理作业所需的工具。
