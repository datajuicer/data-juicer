# Global Configuration Reference

This page lists common global parameters available in a Data-Juicer recipe YAML, along with their defaults. These are set at the top level of the YAML and can be overridden via `--param value` on the command line. For the complete parameter list, run `dj-process --help`.

> Operator-specific parameters are not covered here—see the [Operator Schemas](Operators.md) or individual operator detail pages.

Each description identifies the applicable executor or tool and any required feature switches.

---

## Project & Paths

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | str | `hello_world` | Project name (used in output paths and logs) |
| `dataset_path` | str | `""` | Input dataset path; supports weighted mixing: `<w1> path1 <w2> path2` |
| `dataset` | list/dict | `[]` | Advanced dataset config (local/remote), see [Dataset Configuration](DatasetCfg.md) |
| `export_path` | str | `./outputs/hello_world/hello_world.jsonl` | Output file path |
| `work_dir` | str | `None` | Working-directory base (defaults to export_path's parent); resolution appends job_id |
| `temp_dir` | str | `None` | Temp file directory (used when cache is disabled) |

---

## Executor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `executor_type` | str | `default` | Engine: `default` (local multiprocess) / `ray` / `ray_partitioned` |
| `np` | int | `4` | Number of parallel worker processes |
| `ray_address` | str | `"auto"` | Ray cluster address (ray mode only) |

---

## Input & Format

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_keys` | str/list | `"text"` | Text field name(s) |
| `image_key` | str | `"images"` | Field for image path list |
| `audio_key` | str | `"audios"` | Field for audio path list |
| `video_key` | str | `"videos"` | Field for video path list |
| `suffixes` | str/list | `[]` | File suffixes to load (empty = auto-detect) |
| `load_dataset_kwargs` | dict | `{}` | Extra kwargs for `datasets.load_dataset()` |
| `read_options` | dict | `{}` | PyArrow read options for the `ray` executor |

To process a smaller input with the default executor, use `dataset.max_sample_num` or prepare a subset first. See [Sampling Dry Run](ProcessData.md).

### Analysis and tool-specific parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_probe_ratio` | float | `1.0` | Sampling ratio passed to `sample_data()` by the Sandbox model-inference probe |
| `data_probe_algo` | str | `uniform` | Sampling algorithm for the Sandbox model-inference probe |
| `hpo_config` | str | `None` | Search-space configuration for the [HPO tool](../data_juicer/tools/hpo/README.md) |
| `auto_num` | int | `1000` | Maximum samples analyzed with `dj-analyze --auto` |

[Data-Juicer Sandbox](https://github.com/datajuicer/data-juicer-sandbox) uses these sampling parameters for model-inference probes. To sample through the Python API, call `executor.sample_data(sample_ratio=cfg.data_probe_ratio, sample_algo=cfg.data_probe_algo)` before processing the returned subset.

---

## Export

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_type` | str | `None` | Export format (inferred from path suffix if omitted) |
| `export_shard_size` | int | `0` | Shard size in bytes; 0 = single file |
| `export_in_parallel` | bool | `false` | Parallel export to a single file |
| `export_extra_args` | dict | `{}` | Format-specific extra arguments |
| `export_aws_credentials` | dict | `null` | AWS credentials for S3 export |
| `keep_stats_in_res_ds` | bool | `false` | Keep computed stats fields in output |
| `keep_hashes_in_res_ds` | bool | `false` | Keep computed hash fields in output |

See [Export](Export.md) for details.

---

## Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `op_fusion` | bool | `false` | Fuse compatible operators in default or Ray execution to reduce repeated processing |
| `fusion_strategy` | str | `probe` | Fusion strategy: `probe` orders by fusion group and measured speed; `greedy` orders by fusion group. Speed probing applies to the default executor and standard Analyzer |
| `mapper_fusion` | bool | `true` | Fuse consecutive GPU Mappers (requires op_fusion) |
| `mapper_fusion_vram_limit` | float | `0.9` | Max aggregate VRAM fraction for fused mappers |
| `adaptive_batch_size` | bool | `false` | Adaptive batch sizes for batched operators in the `default` executor |
| `turbo` | bool | `false` | Turbo mode (maximize speed at batch_size=1) |

---

## Cache & Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cache` | bool | `true` | Use HuggingFace datasets cache |
| `ds_cache_dir` | str | `None` | Custom cache directory (overrides `HF_DATASETS_CACHE`) |
| `cache_compress` | str | `None` | Cache compression: `gzip` / `zstd` / `lz4` |
| `use_checkpoint` | bool | `false` | Enable default-executor checkpointing (disables cache; mutually exclusive with op_fusion) |

### ray_partitioned Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint.enabled` | bool | `true` | Enable partition checkpointing |
| `checkpoint.strategy` | str | `every_n_ops` | Strategy: `every_op` / `every_partition` / `every_n_ops` / `manual` / `disabled` |
| `checkpoint.n_ops` | int | `5` | Interval for `every_n_ops` strategy |
| `checkpoint.op_names` | list | `[]` | Operator names to checkpoint for `manual` strategy |

---

## Job Management & Resumption

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `job_id` | str | `None` | Custom job ID for tracking and resumption |
| `resume` | str | `None` | Resume a job by ID (ray_partitioned only) |
| `event_logging.enabled` | bool | `true` | Enable event logging |
| `event_log_dir` | str | `None` | Event log directory (fast storage recommended) |
| `checkpoint_dir` | str | `None` | Checkpoint directory for `ray_partitioned`; the default executor stores checkpoints in `<work_dir>/ckpt` |

---

## Tracing & Monitoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `open_tracer` | bool | `false` | Enable sample tracing (records before/after for each op) |
| `op_list_to_trace` | list | `[]` | Operators to trace (empty = all) |
| `trace_num` | int | `10` | Number of changed samples shown per operator |
| `trace_keys` | list | `[]` | Fields to include in trace output |
| `open_monitor` | bool | `false` | Enable resource monitoring (CPU/memory/GPU) |
| `open_insight_mining` | bool | `false` | Enable op-wise insight mining (stat/tag change tracking) |
| `op_list_to_mine` | list | `[]` | Operators for insight mining (empty = all that produce stats) |

---

## Reserved Configuration Fields

The following fields are reserved and are not used by the built-in executors. See [Job Management](JobManagement.md) for log output configuration and [Intermediate Storage](PartitionAndCheckpoint.md) for partition file management.

| Fields | Category |
|--------|----------|
| `max_log_size_mb`, `backup_count` | Log rotation and retention |
| `intermediate_storage.*`, `preserve_intermediate_data` | Intermediate storage |
| `resource_optimization.auto_configure` | Resource configuration |

---

## Encryption

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decrypt_after_reading` | bool | `false` | Decrypt input files on read |
| `encrypt_before_export` | bool | `false` | Encrypt output files on write |
| `encryption_key_path` | str | `None` | Path to Fernet key file (or env var `DJ_ENCRYPTION_KEY`) |

---

## Error Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_op_error` | bool | `true` | Skip errors caused by unexpected invalid samples |

---

## Multimodal Special Tokens

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_special_token` | `<__dj__image>` | Placeholder for images in text |
| `audio_special_token` | `<__dj__audio>` | Placeholder for audio in text |
| `video_special_token` | `<__dj__video>` | Placeholder for video in text |
| `eoc_special_token` | `<\|__dj__eoc\|>` | End-of-chunk marker in text |

---

## Operator Environment Management (Ray only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_common_dep_num_to_combine` | int | `-1` | Min common deps to merge op envs (-1 = no merging) |
| `conflict_resolve_strategy` | str | `split` | Conflict resolution: `split` / `overwrite` / `latest` |
