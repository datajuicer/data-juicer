# Branch Execution

Branch execution allows you to execute multiple processing branches in parallel from a common checkpoint, automatically deduplicating the common processing steps.

分支执行允许您从公共检查点并行执行多个处理分支，自动去重公共处理步骤。

## Overview 概述

The branch executor:
1. Executes `common_process` once and saves a checkpoint
2. Executes each branch in parallel from the checkpoint
3. Each branch has its own `export_path`

分支执行器：
1. 执行一次 `common_process` 并保存检查点
2. 从检查点并行执行每个分支
3. 每个分支都有自己的 `export_path`

## Configuration 配置

```yaml
executor_type: 'branch'

# Common process executed once
common_process:
  - miner_u_mapper:
      # configuration
  - document_deduplicator:
      # configuration

# Branches executed in parallel
branches:
  - name: "branch1"
    process:
      - data_augmentation_mapper:
          # configuration
    export_path: './outputs/branch1.jsonl'
  
  - name: "branch2"
    process:
      - html_enhancement_mapper:
          # configuration
    export_path: './outputs/branch2.jsonl'
```

## Use Cases 使用场景

### Example: Multiple Enhancement Paths

You have a common preprocessing pipeline (minerU → deduplication), and then want to apply different enhancements in parallel:

您有一个公共的预处理流程（minerU → 去重），然后想要并行应用不同的增强：

```yaml
common_process:
  - miner_u_mapper: {...}
  - document_deduplicator: {...}

branches:
  - name: "data_augmentation"
    process:
      - data_augmentation_mapper: {...}
    export_path: './outputs/augmented.jsonl'
  
  - name: "html_enhancement"
    process:
      - html_enhancement_mapper: {...}
    export_path: './outputs/html_enhanced.jsonl'
```

This will:
1. Execute minerU and deduplication once
2. Use the result as input for both branches
3. Execute data augmentation and HTML enhancement in parallel
4. Export each branch to its own file

这将：
1. 执行一次 minerU 和去重
2. 使用结果作为两个分支的输入
3. 并行执行数据增强和 HTML 增强
4. 将每个分支导出到自己的文件

## Benefits 优势

1. **Automatic Deduplication**: Common steps are executed only once
   自动去重：公共步骤只执行一次

2. **Parallel Execution**: Branches run in parallel for better performance
   并行执行：分支并行运行以获得更好的性能

3. **Clear Configuration**: Branch structure is explicit and self-documenting
   清晰的配置：分支结构明确且自文档化

4. **Flexible Export**: Each branch can export to different paths
   灵活的导出：每个分支可以导出到不同的路径

## Execution mode / Ray 与 Ray 模式

Two backends for running branches in parallel:

- **`backend: thread`** (default): Common process and each branch run **locally**. Branches are executed in parallel with a **thread pool**. No extra dependency.
- **`backend: ray`**: Common process can run on the driver (default) or on a Ray worker when `run_common_on_ray: true` (pure Ray). Each branch runs as a **Ray remote task**. Requires `pip install ray` and (optionally) a Ray cluster.

**注意**：`executor_type: 'ray'` 是整条线性流水线用 Ray，与分支执行器不同。分支执行器用 `executor_type: branch`，再通过 `branch.backend: ray` 让各分支以 Ray 任务运行。

### Enabling Ray backend 启用 Ray 后端

```yaml
executor_type: 'branch'
branch:
  backend: ray   # default is "thread"
common_process: [...]
branches: [...]
```

Ray is auto-initialized if not already. In Ray mode, `run()` returns `dict[branch_name] -> None` (results only on disk at each `export_path`).

### Pure Ray: common on Ray too 纯 Ray：common 也上 Ray

When all work should run on the cluster (no heavy common process on the driver), set `run_common_on_ray: true`:

```yaml
branch:
  backend: ray
  run_common_on_ray: true   # common process runs as a Ray task; branches also on Ray
```

**Requirement**: `dataset_path` and `work_dir` must be **accessible from every Ray worker** (e.g. NFS, S3, or a path on shared storage). The common task loads the dataset from `dataset_path` and writes checkpoints under `work_dir`; branches read the common result via object store or disk.

**要求**：`dataset_path` 与 `work_dir` 须对**所有 Ray worker 可访问**（如 NFS、S3 或共享存储路径）。

## MVP example 最小可运行示例

Minimal structure (thread backend): set `executor_type: branch`, `common_process`, and `branches` with per-branch `process` and `export_path`. Top-level `export_path` is unused (each branch has its own). Run:

```bash
python tools/process_data.py --config <your_branch_config>.yaml
# or: dj process --config <your_branch_config>.yaml
```

For **Ray backend**, add under `branch:` set `backend: ray`. For **pure Ray** (common on Ray too), add `run_common_on_ray: true`; then `dataset_path` and `work_dir` must be worker-accessible. Full example configs (thread / ray / ray-pure) are maintained in [dj-hub](https://github.com/datajuicer/dj-hub).

## Branch options (optional)

Under top-level key `branch` you can tune execution (WIP, see Issue #915):

```yaml
branch:
  backend: thread       # "thread" (default) or "ray"
  run_common_on_ray: false  # when backend=ray, if true run common as a Ray task (pure Ray)
  parallel: true        # run branches in parallel (default: true; ignored when backend=ray)
  fail_fast: true       # stop on first branch failure (default: true)
  max_workers: null     # thread pool size when backend=thread; null = one per branch
  retries: 0            # per-branch retry count (default: 0)
```

- With `backend: ray`, each branch runs as a Ray remote task; common dataset is shared via object store when possible, else via disk. Branch configs are minimal. When `run_common_on_ray: true`, the common process also runs as one Ray task (dataset_path and work_dir must be worker-accessible).

## Implementation Details 实现细节

- Common process checkpoints are saved in `{work_dir}/.branch_ckpt/common/`
- **Ray backend**: common runs on driver by default, or on a Ray task when `run_common_on_ray: true`. Common result is then put in object store (or saved to disk) and consumed by branch tasks. Branch configs are minimal. On first failure, remaining tasks are cancelled when `fail_fast` is true. `retries` resubmits only failed branches.
- Branch checkpoints are saved in `{work_dir}/.branch_ckpt/{branch_name}/`
- Each branch uses a separate DefaultExecutor instance (locally for thread backend; inside each Ray task for Ray backend)
- **Thread backend**: branches run in parallel by default (thread pool); use `branch.parallel: false` for sequential
- **Ray backend**: return value is `{branch_name: None}` (results only on disk at each `export_path`)

- 公共流程检查点保存在 `{work_dir}/.branch_ckpt/common/`
- **Ray 后端**：优先用 object store 传 common 数据集；失败则落盘到 `{work_dir}/.branch_ckpt/common/dataset/`；仅传最小 config；支持 fail_fast 取消与 retries 重试
- 分支检查点保存在 `{work_dir}/.branch_ckpt/{branch_name}/`
- 每个分支使用单独的 DefaultExecutor 实例（thread 在本地，ray 在各自 Ray 任务内）
- 默认线程池并行；Ray 后端下分支以 Ray 任务运行，返回值仅包含导出路径
