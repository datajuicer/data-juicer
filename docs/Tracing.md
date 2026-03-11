# Data Tracing

This document describes DataJuicer's tracing system for tracking sample-level changes during data processing.

## Overview

The Tracer records how each operator modifies, filters, or deduplicates individual samples in the processing pipeline. This is useful for:

- **Debugging** — Understanding why specific samples were modified or removed
- **Quality Assurance** — Verifying operators are working as expected
- **Auditing** — Maintaining records of data transformations

## Configuration

### Basic Settings

```yaml
open_tracer: false        # Enable/disable tracing
op_list_to_trace: []      # List of operators to trace (empty = all operators)
trace_num: 10             # Maximum number of samples to collect per operator
trace_keys: []            # Additional fields to include in trace output
```

### Command Line

```bash
# Enable tracing for all operators
dj-process --config config.yaml --open_tracer true

# Trace only specific operators
dj-process --config config.yaml --open_tracer true \
    --op_list_to_trace clean_email_mapper,words_num_filter

# Collect more samples per operator
dj-process --config config.yaml --open_tracer true --trace_num 50

# Include additional fields in trace output
dj-process --config config.yaml --open_tracer true \
    --trace_keys sample_id,source_file
```

## Output Structure

Trace results are stored in the `trace/` subdirectory of the work directory:

```
{work_dir}/
└── trace/
    ├── sample_trace-clean_email_mapper.jsonl
    ├── sample_trace-words_num_filter.jsonl
    ├── duplicate-document_deduplicator.jsonl
    └── ...
```

Each trace file is in JSONL format (one JSON object per line), with content varying by operator type.

## Traced Operator Types

### Mapper Tracing

For Mapper operators, the Tracer records samples where text content changes. Each record contains:

| Field | Description |
|-------|-------------|
| `original_text` | Text before Mapper processing |
| `processed_text` | Text after Mapper processing |
| *trace_keys fields* | Values corresponding to configured `trace_keys` |

Example output (`sample_trace-clean_email_mapper.jsonl`):
```json
{"original_text":"Contact us at user@example.com for details.","processed_text":"Contact us at  for details."}
{"original_text": "Email: admin@test.org", "processed_text": "Email: "}
```

Only samples with actual text changes are collected; unchanged samples are skipped.

### Filter Tracing

For Filter operators, the Tracer records samples that are **filtered out** (removed). Each record contains the complete sample data.

Example output (`sample_trace-words_num_filter.jsonl`):
```json
{"text": "Too short.", "__dj__stats__": {"words_num": 2}}
{"text": "Also brief.", "__dj__stats__": {"words_num": 2}}
```

Only samples that fail the filter are collected; samples passing the filter are skipped.

### Deduplicator Tracing

For Deduplicator operators, the Tracer records pairs of near-duplicate samples. Each record contains:

| Field | Description |
|-------|-------------|
| `dup1` | First sample in the duplicate pair |
| `dup2` | Second sample in the duplicate pair |

Example output (`duplicate-document_deduplicator.jsonl`):
```json
{"dup1": "This is a duplicate text.", "dup2": "This is a duplicate text."}
```

## Sample Collection Behavior

The Tracer uses an efficient **sample-level collection** approach:

1. Each operator collects at most `trace_num` samples during processing
2. Collection stops early once enough samples are gathered
3. In default mode, collection is **thread-safe** using multiprocess locks
4. In Ray mode, each Worker has its own Tracer instance (no locking needed)

This design minimizes performance overhead — the Tracer does not compare the entire dataset, but captures changes in real-time during processing.

## trace_keys

The `trace_keys` option allows including additional fields from original samples in the trace output. This is useful for identifying which samples were affected:

```yaml
open_tracer: true
trace_keys:
  - sample_id
  - source_file
```

With this configuration, Mapper trace entries will include:
```json
{
  "sample_id": "doc_00042",
  "source_file": "corpus_part1.jsonl",
  "original_text": "Original content...",
  "processed_text": "Processed content..."
}
```

## API Reference

### Tracer (Default Mode)

```python
from data_juicer.core.tracer import Tracer

tracer = Tracer(
    work_dir="./outputs",
    op_list_to_trace=["clean_email_mapper", "words_num_filter"],
    show_num=10,
    trace_keys=["sample_id"]
)

# Check if an operator should be traced
tracer.should_trace_op("clean_email_mapper")  # True

# Check if enough samples have been collected
tracer.is_collection_complete("clean_email_mapper")  # False

# Collect Mapper sample
tracer.collect_mapper_sample(
    op_name="clean_email_mapper",
    original_sample={"text": "Email: a@b.com"},
    processed_sample={"text": "Email: "},
    text_key="text"
)

# Collect Filter sample
tracer.collect_filter_sample(
    op_name="words_num_filter",
    sample={"text": "too short"},
    should_keep=False
)
```

### RayTracer (Distributed Mode)

```python
from data_juicer.core.tracer.ray_tracer import RayTracer

# RayTracer is a Ray Actor — created via Ray
tracer = RayTracer.remote(
    work_dir="./outputs",
    op_list_to_trace=None,  # Trace all operators
    show_num=10,
    trace_keys=["sample_id"]
)

# Remote method calls
ray.get(tracer.collect_mapper_sample.remote(
    op_name="clean_email_mapper",
    original_sample={"text": "Email: a@b.com"},
    processed_sample={"text": "Email: "},
    text_key="text"
))

# Finalize and export all trace results
ray.get(tracer.finalize_traces.remote())
```

### Helper Functions

The `data_juicer.core.tracer` module provides mode-agnostic helper functions:

```python
from data_juicer.core.tracer import (
    should_trace_op,
    check_tracer_collect_complete,
    collect_for_mapper,
    collect_for_filter,
)

# These functions automatically handle default mode and Ray mode
should_trace_op(tracer_instance, "clean_email_mapper")
check_tracer_collect_complete(tracer_instance, "clean_email_mapper")
collect_for_mapper(tracer_instance, "op_name", original, processed, "text")
collect_for_filter(tracer_instance, "op_name", sample, should_keep=False)
```

## Performance Considerations

### Overhead

- When `trace_num` is small (default: 10), the additional overhead of tracing is minimal
- Once an operator has collected `trace_num` samples, no further collection occurs
- The main cost is comparing original and processed text in Mappers

### Recommendations

| Scenario | Recommended Settings |
|----------|----------------------|
| Development/Debugging | `open_tracer: true`, `trace_num: 10-50` |
| Production Runs | `open_tracer: false` |
| Auditing Specific Operators | `open_tracer: true`, `op_list_to_trace: [specific operators]` |
| Large-scale Tracing | `open_tracer: true`, `trace_num: 100`, specify `op_list_to_trace` |

## Troubleshooting

**No trace files generated:**
```bash
# Verify tracer is enabled
grep "open_tracer" config.yaml

# Check if trace directory exists
ls -la ./outputs/{work_dir}/trace/
```

**Trace files are empty:**
- For Mapper: The operator may not have modified any samples
- For Filter: The operator may not have filtered out any samples
- Check logs for warnings like "Datasets before and after op [X] are all the same"

**Too few samples in trace files:**
- Increase `trace_num` to collect more samples
- There may be fewer than `trace_num` changed/filtered samples in the dataset
