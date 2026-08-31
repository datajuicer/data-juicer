# Global Configuration Migration

## Removed settings

The following settings had no runtime effect in the built-in executors. They
have been removed from the configuration parser. Delete them from YAML files
and command-line arguments when upgrading; old recipes containing these keys
now fail configuration validation.

| Removed setting | Current configuration or behavior |
| --- | --- |
| `intermediate_storage.*` (all eight fields) | `ray_partitioned` checkpoints use Parquet with the writer's default compression. Use `checkpoint.enabled`, `checkpoint.strategy`, `checkpoint.n_ops`, and `checkpoint.op_names` to control checkpoint creation. |
| `preserve_intermediate_data` | Temporary run directories are cleaned on exit from the executor's run context. Checkpoints are stored separately for resumption. |
| `partition_size`, `max_partition_size_mb` | Use `partition.mode: manual` with `partition.num_of_partitions`, or `partition.mode: auto` with `partition.target_size_mb`. These express partition counts and planning targets, rather than translating the old values directly. |
| `resource_optimization.auto_configure` | Automatic partition planning is selected by `partition.mode: auto`. |
| `max_log_size_mb`, `backup_count` | The built-in logger has no configurable size-based rotation or backup-count policy. `setup_logger()` accepts `save_dir`, `filename`, `level`, and its other documented arguments. |

Removing these settings does not change checkpoint creation, resumption, or
temporary-directory cleanup. There is no replacement setting for preserving
arbitrary intermediate files, choosing their format/compression, or retaining
files by age or job outcome.

`checkpoint.strategy: every_partition` is also rejected: it was accepted by
the parser but fell back to `every_op` in the executor. The supported strategies
are `every_op`, `every_n_ops`, `manual`, and `disabled`.
`checkpoint.n_ops`, `partition.target_size_mb`, and a non-null
`override_num_blocks` must be positive integers.

## Available controls

`load_jsonl_lenient` is now accepted by YAML and CLI configuration. In the
default executor and Analyzer it selects the existing lenient JSONL reader,
which skips malformed lines. `DATA_JUICER_JSONL_LENIENT=1` also enables that
reader. See [dataset configuration](DatasetCfg.md) for supported file types.

`use_dag` selects execution-plan generation and DAG monitoring. Its default
`null` preserves executor defaults: enabled for `ray` and `ray_partitioned`,
disabled for `default`. Set it to `true` or `false` to override.

## Reader options

Dataset loading applies global reader options consistently across execution,
analysis, and direct `DatasetBuilder` calls:

- `load_dataset_kwargs` provides HuggingFace reader defaults for the default
  executor and Analyzer, such as Parquet `columns` or CSV `delimiter`.
- `read_options` configures PyArrow JSON reading in Ray, including local JSON
  input. `override_num_blocks` controls the requested Ray read block count.
- Explicit `DatasetBuilder.load_dataset(...)` keyword arguments override the
  corresponding global defaults. `generated_dataset_config` continues to use
  its own formatter constructor arguments.

`data_probe_algo` and `data_probe_ratio` remain available for the external
Data-Juicer Sandbox model probe. `hpo_config` remains an HPO-tool setting.

Automatic partition analysis now uses the first configured `text_keys` field,
including nested paths. An empty list excludes text from modality and text-length
analysis. Notification and annotation settings are configured on individual operators,
where their implementations consume them; the global example points to that scope.
