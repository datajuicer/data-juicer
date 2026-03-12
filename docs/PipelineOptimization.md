# Pipeline Optimization Framework

This document describes Data-Juicer's pipeline optimization framework, which provides intelligent query planning and execution optimization for data processing pipelines.

## Overview

The Pipeline Optimization Framework transforms user-defined processing pipelines into optimized execution plans. It operates at a higher level than Ray's built-in optimizations, providing domain-specific optimizations that understand data processing semantics.

**Current capabilities:**
- **Operator Reordering** - Run selective/cheap operations first
- **Operator Fusion** - Combine filters/mappers that share intermediate variables

**Planned capabilities:**
- **Filter Pushdown** - Push predicates to data sources to reduce I/O (requires DataConnector integration)
- **Cost-based Optimization** - Use statistics for optimization decisions

## Ray's Built-in Optimizations vs. Data-Juicer Optimization Framework

Understanding the division of responsibilities between Ray Data's optimizations and Data-Juicer's optimization framework is crucial for understanding why both layers are necessary.

### What Ray Data Already Optimizes

Ray Data provides powerful built-in optimizations at the **execution level**:

| Optimization | Description | How It Works |
|-------------|-------------|--------------|
| **Operator Fusion** | Combines consecutive map operations | Ray automatically fuses `map()` calls to reduce serialization overhead |
| **Pipelining** | Overlaps compute and I/O | While one block is being processed, the next is being read |
| **Automatic Parallelization** | Distributes work across workers | Data is partitioned into blocks and processed in parallel |
| **Lazy Execution** | Builds execution plan before running | Operations are not executed until materialization (e.g., `take()`, `write()`) |
| **Memory Management** | Handles backpressure and spilling | Automatically spills to disk when memory is constrained |
| **Block Coalescing** | Merges small blocks | Reduces scheduling overhead for many small blocks |

**Example of Ray's automatic fusion:**

```python
# These three map calls are automatically fused by Ray
ds = ray.data.read_parquet("data.parquet")
ds = ds.map(lambda x: {**x, "len": len(x["text"])})
ds = ds.map(lambda x: {**x, "words": x["text"].split()})
ds = ds.map(lambda x: {**x, "word_count": len(x["words"])})
# Ray executes as single fused operation, not three separate passes
```

### What Ray Cannot Optimize

Ray operates at the **execution level** without understanding the **semantics** of your data processing operations. It cannot:

| Limitation | Why It Matters | Example |
|-----------|----------------|---------|
| **Reorder operations** | Doesn't know which filters are selective | Can't move `text_length_filter` (removes 70%) before `perplexity_filter` (expensive) |
| **Push filters to data source** | Doesn't understand data source capabilities | Can't convert `text_length_filter` to SQL `WHERE LENGTH(text) > 100` |
| **Share intermediate variables** | Doesn't know ops compute same things | Can't share loaded images between `image_size_filter` and `image_aspect_ratio_filter` |
| **Understand operator costs** | Doesn't know LLM inference is expensive | Can't prioritize cheap filters before expensive model calls |
| **Project only needed columns** | Doesn't analyze column dependencies | Reads all columns even if pipeline only uses `text` |

### The Two-Layer Optimization Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA-JUICER OPTIMIZATION LAYER                       │
│                     (Semantic/Logical Optimization)                     │
│                                                                         │
│  Input:  User's YAML config with operators                             │
│  Output: Optimized operator list + modified data source query          │
│                                                                         │
│  Optimizations:                                                         │
│  • Filter pushdown to data source (ODPS, Parquet, etc.)               │
│  • Projection pushdown (only read needed columns)                      │
│  • Operator reordering (selective/cheap first)                         │
│  • Semantic fusion (share loaded images, computed stats)               │
│  • Cost-based decisions using domain knowledge                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       RAY OPTIMIZATION LAYER                            │
│                     (Execution/Physical Optimization)                   │
│                                                                         │
│  Input:  Optimized operator list from Data-Juicer                      │
│  Output: Efficient distributed execution                                │
│                                                                         │
│  Optimizations:                                                         │
│  • Automatic map fusion (reduce serialization)                         │
│  • Pipelining (overlap I/O and compute)                                │
│  • Parallelization (distribute across workers)                         │
│  • Memory management (backpressure, spilling)                          │
│  • Block scheduling (locality, load balancing)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Illustrative Example: How Both Layers Work Together

Consider this pipeline processing a large dataset from ODPS:

```yaml
dataset_path: "odps://project/web_crawl"
process:
  - text_length_filter: {min_len: 100, max_len: 10000}
  - language_id_score_filter: {lang: 'en', min_score: 0.8}
  - perplexity_filter: {max_ppl: 1000}
  - image_size_filter: {min_width: 200}
  - image_aspect_ratio_filter: {min_ratio: 0.5, max_ratio: 2.0}
  - clean_email_mapper: {}
```

**Without Data-Juicer Optimization (Ray only):**

```
1. Read entire dataset from ODPS (all columns)
2. Execute filters in user-specified order:
   - text_length_filter: processes full dataset
   - language_id_score_filter: model inference on full dataset
   - perplexity_filter: model inference on full dataset
   - image_size_filter: loads all images
   - image_aspect_ratio_filter: loads all images AGAIN
3. Ray fuses the map operations within each filter
```

**With Data-Juicer Optimization + Ray:**

```
1. Filter Pushdown (future): Push text_length_filter to ODPS
   → SELECT text, images FROM web_crawl WHERE LENGTH(text) BETWEEN 100 AND 10000
   → Reduces data transfer

2. Operator Reordering:
   - Run cheap filters (text_length, image_size) before expensive ones
   - Expensive model inference runs on smaller dataset

3. Operator Fusion: Fuse image filters
   - image_size_filter + image_aspect_ratio_filter → load images ONCE

4. Ray executes the optimized plan with its own optimizations
```

**Potential Benefits** (actual gains depend on workload characteristics):

| Optimization | Potential Benefit | When It Helps Most |
|--------------|-------------------|-------------------|
| Filter Pushdown | Reduced I/O | Database/cloud sources with selective filters |
| Operator Reordering | Less wasted compute | Pipelines with expensive ops and selective cheap filters |
| Operator Fusion | Reduced redundant work | Operations sharing intermediate data (e.g., loaded images) |

> **Note**: Actual performance improvements vary significantly based on dataset characteristics, filter selectivity, and infrastructure. Benchmarking on your specific workload is recommended.

### When Data-Juicer Optimization May Help

| Scenario | Potential Value | Why |
|----------|----------------|-----|
| Large datasets | Higher | I/O and compute reduction more impactful |
| ODPS/database sources | Higher | Filter pushdown possible (when implemented) |
| Multimodal data (images/video) | Higher | Avoid redundant media loading via fusion |
| Pipelines with model inference | Higher | Reordering can reduce expensive inference calls |
| Simple text pipelines | Moderate | Reordering may still help |
| Small datasets (<1GB) | Lower | Optimization overhead may exceed benefit |

### Complementary, Not Competing

The two optimization layers are **complementary**:

```
User Config
    │
    ▼
┌─────────────────────────┐
│  Data-Juicer Optimizer  │  ← "WHAT to execute" (logical plan)
│  - Reorder operators    │
│  - Push to data source  │
│  - Fuse semantically    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    Ray Data Engine      │  ← "HOW to execute" (physical plan)
│  - Parallelize          │
│  - Pipeline I/O         │
│  - Manage memory        │
└─────────────────────────┘
    │
    ▼
  Results
```

Data-Juicer optimization **reduces the work** that needs to be done.
Ray optimization **efficiently executes** whatever work remains.

Both layers working together achieve the best performance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Config (YAML)                          │
│  dataset_path: "odps://project/table"                               │
│  process:                                                           │
│    - text_length_filter: {min_len: 100}                            │
│    - language_id_score_filter: {lang: 'en'}                        │
│    - clean_email_mapper: {}                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Pipeline AST Builder                           │
│  Converts config to Abstract Syntax Tree representation            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Optimization Passes                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │  Predicate  │ │  Operator   │ │  Operator   │ │    Cost     │  │
│  │  Pushdown   │ │  Reordering │ │   Fusion    │ │  Estimation │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Optimized Execution Plan                         │
│  1. Data Source Query (with pushed predicates)                     │
│  2. Reordered operators                                            │
│  3. Fused operator groups                                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Executor                                    │
│  DefaultExecutor / RayExecutor / PartitionedRayExecutor            │
│  (See PartitionAndCheckpoint.md for execution details)             │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline AST

The Pipeline AST (Abstract Syntax Tree) represents the processing pipeline as a tree structure, enabling analysis and transformation.

### Core Data Structures

```python
class OpType(Enum):
    ROOT = "root"
    FILTER = "filter"
    MAPPER = "mapper"
    DEDUPLICATOR = "deduplicator"
    SELECTOR = "selector"

class OpNode:
    """Represents a single operation in the pipeline."""
    name: str                    # Operation name (e.g., "text_length_filter")
    op_type: OpType              # Operation type
    config: Dict[str, Any]       # Operation configuration
    children: List[OpNode]       # Child nodes
    parent: Optional[OpNode]     # Parent node

    # Metadata for optimization (see "Extended Metadata" section)
    pushable_predicate: Optional[Predicate]
    required_columns: Set[str]
    produced_columns: Set[str]
    estimated_cost: float
    estimated_selectivity: float

class PipelineAST:
    """Abstract Syntax Tree for a processing pipeline."""
    root: OpNode

    def build_from_config(self, config: List[Dict]) -> None: ...
    def optimize(self, strategies: List[OptimizationStrategy]) -> PipelineAST: ...
    def to_operators(self) -> List[Operator]: ...
```

### Extended Metadata for Optimization

Each operator should provide metadata to enable optimization:

```python
class Operator:
    # Required for projection pushdown
    def required_columns(self) -> Set[str]:
        """Columns this operator reads."""
        return {self.text_key}

    def produced_columns(self) -> Set[str]:
        """Columns this operator adds or modifies."""
        return set()

    # Required for filter pushdown
    def get_pushable_predicate(self) -> Optional[Predicate]:
        """Return predicate if this filter can be pushed to data source."""
        return None

    # Required for cost-based optimization
    def estimated_cost(self) -> float:
        """Estimated cost per sample (0.0-1.0 scale)."""
        return 0.1  # Default: low cost

    def estimated_selectivity(self) -> float:
        """Estimated fraction of samples that pass (for filters)."""
        return 1.0  # Default: keeps all samples

    # Required for operator fusion
    def get_intermediate_vars(self) -> Set[str]:
        """Intermediate variables this operator computes/uses."""
        return set()

    # Required for model optimization
    def required_models(self) -> Set[str]:
        """Models this operator requires."""
        return set()
```

## Optimization Strategies

### 1. Filter Pushdown

Push filter predicates to data sources to reduce I/O.

**Applicable filters:**

| Filter | Pushdown SQL | Notes |
|--------|-------------|-------|
| `text_length_filter` | `LENGTH(text) BETWEEN min AND max` | Direct mapping |
| `words_num_filter` | `word_count BETWEEN min AND max` | Requires pre-computed column |
| `specified_field_filter` | `field = value` | Direct mapping |
| `specified_numeric_field_filter` | `field BETWEEN min AND max` | Direct mapping |
| `suffix_filter` | `path LIKE '%.ext'` | Pattern matching |

**Non-pushable filters:**
- `language_id_score_filter` - Requires model inference
- `perplexity_filter` - Requires model inference
- `image_*_filter` - Requires loading binary data
- `*_deduplicator` - Requires global state

**Implementation:**

```python
class FilterPushdownStrategy(OptimizationStrategy):
    def optimize(self, ast: PipelineAST, connector: DataConnector) -> PipelineAST:
        pushable = []
        remaining = []

        for node in ast.get_filter_nodes():
            predicate = node.op.get_pushable_predicate()
            if predicate and connector.supports_predicate(predicate):
                pushable.append(predicate)
            else:
                remaining.append(node)

        # Modify data source query
        connector.push_predicates(pushable)

        # Return AST with remaining filters
        return ast.with_nodes(remaining)
```

### 2. Projection Pushdown

Only read columns that are needed by the pipeline.

```python
class ProjectionPushdownStrategy(OptimizationStrategy):
    def optimize(self, ast: PipelineAST, connector: DataConnector) -> PipelineAST:
        # Analyze column dependencies
        required = set()
        for node in ast.traverse():
            required |= node.op.required_columns()

        # Push to data source
        connector.select_columns(required)

        return ast
```

### 3. Operator Reordering

Reorder operators to minimize work:
1. Run high-selectivity filters first (filters that remove most data)
2. Run cheap operators before expensive ones
3. Respect data dependencies

```python
class OperatorReorderingStrategy(OptimizationStrategy):
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        # Group by type while respecting dependencies
        filters = ast.get_filter_nodes()
        mappers = ast.get_mapper_nodes()

        # Sort filters by (selectivity ASC, cost ASC)
        # Lower selectivity = removes more data = run first
        filters.sort(key=lambda n: (
            n.op.estimated_selectivity(),
            n.op.estimated_cost()
        ))

        # Interleave: filters first, then mappers
        return ast.reorder(filters + mappers)
```

**Example:**

```yaml
# Before optimization
process:
  - perplexity_filter: {}          # Cost: HIGH, Selectivity: 0.8
  - text_length_filter: {min: 100} # Cost: LOW,  Selectivity: 0.3
  - clean_email_mapper: {}

# After optimization
process:
  - text_length_filter: {min: 100} # Run first: cheap, removes 70%
  - perplexity_filter: {}          # Run on 30% of data
  - clean_email_mapper: {}
```

#### Probing-Aided Reordering

Static heuristics categorize operations as "cheap" or "expensive" based on operation type, but this can miss actual runtime differences. **Probing** measures actual costs by running each operation on a small sample.

**What Probing Measures:**
- **Runtime**: Execution time per sample (ms)
- **Selectivity**: Ratio of samples that pass (for filters)

These are combined into an effective cost: `effective_cost = runtime × selectivity`

This formula favors placing highly selective filters (that remove more data) earlier in the pipeline.

**Configuration:**

```yaml
enable_optimizer: true
optimizer_strategies: ['op_reorder']
optimizer_probe_enabled: true      # Enable probing (default: true)
optimizer_probe_samples: 100       # Sample size (default: 100)
```

**Benchmark Results (C4 dataset, 356K samples, 5 filters):**

| Configuration | Time | Speedup | Notes |
|--------------|------|---------|-------|
| Baseline (no optimization) | 73.5s | - | Original order |
| Static reorder (heuristics) | 73.3s | ~0% | No change - all filters classified as "cheap" |
| Probed reorder (measured) | 53.1s | **28%** | Reordered based on actual measured costs |

**Why Static Heuristics Failed:**

All 5 filters were in the `CHEAP_FILTERS` category (same static cost), so no reordering occurred. But actual runtimes varied significantly:

| Filter | Static Cost | Actual Runtime |
|--------|-------------|----------------|
| text_length_filter | 1 (cheap) | 3.6s |
| special_characters_filter | 1 (cheap) | 5.2s |
| alphanumeric_filter | 1 (cheap) | 8.8s |
| words_num_filter | 1 (cheap) | 12.4s |
| character_repetition_filter | 1 (cheap) | 21.6s |

Probing discovered these differences and reordered accordingly, moving the expensive `character_repetition_filter` to run last on fewer samples.

**Probing Overhead:** ~0.15s for 100 samples (negligible compared to 20s savings)

### 4. Operator Fusion

Combine operators that share intermediate variables to avoid redundant computation.

**Fusible groups by intermediate variable:**

| Intermediate Variable | Operators |
|----------------------|-----------|
| `lines` (text.split('\n')) | `average_line_length_filter`, `maximum_line_length_filter` |
| `words` (text.split()) | `words_num_filter`, `word_repetition_filter`, `stopwords_filter` |
| `loaded_images` | All `image_*` filters and mappers |
| `loaded_audios` | All `audio_*` filters and mappers |
| `loaded_videos` | All `video_*` filters and mappers |
| `sampled_frames` | Video frame-based operations |

**Implementation:**

```python
class OperatorFusionStrategy(OptimizationStrategy):
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        # Group consecutive filters by intermediate vars
        groups = self._group_by_intermediate_vars(ast.get_filter_nodes())

        fused_nodes = []
        for group in groups:
            if len(group) > 1:
                # Create fused filter
                fused = FusedFilter(
                    name=f"fused_filter({','.join(n.name for n in group)})",
                    filters=[n.op for n in group]
                )
                fused_nodes.append(OpNode(fused))
            else:
                fused_nodes.append(group[0])

        return ast.with_nodes(fused_nodes)
```

### 5. Model Sharing

Share loaded models across operators that use the same model.

```python
class ModelSharingStrategy(OptimizationStrategy):
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        # Group operators by required model
        model_groups = defaultdict(list)
        for node in ast.traverse():
            for model in node.op.required_models():
                model_groups[model].append(node)

        # Create shared model context for groups
        for model, nodes in model_groups.items():
            if len(nodes) > 1:
                shared_context = ModelContext(model)
                for node in nodes:
                    node.op.set_model_context(shared_context)

        return ast
```

### 6. Common Subexpression Elimination

Identify and cache common computations.

```python
class CSEStrategy(OptimizationStrategy):
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        # Find operators computing same intermediate vars
        computations = defaultdict(list)
        for node in ast.traverse():
            for var in node.op.get_intermediate_vars():
                computations[var].append(node)

        # For vars computed multiple times, add caching
        for var, nodes in computations.items():
            if len(nodes) > 1:
                # First occurrence computes and caches
                nodes[0].op.enable_caching(var)
                # Subsequent occurrences read from cache
                for node in nodes[1:]:
                    node.op.use_cached(var)

        return ast
```

## Data Connector Interface

Data connectors provide the interface for optimization strategies to interact with data sources.

```python
class DataConnector(ABC):
    """Abstract interface for data source connectors."""

    @abstractmethod
    def get_schema(self) -> Schema:
        """Get the schema of the data source."""
        pass

    @abstractmethod
    def get_statistics(self) -> TableStatistics:
        """Get statistics for cost estimation."""
        pass

    @abstractmethod
    def supports_predicate(self, predicate: Predicate) -> bool:
        """Check if a predicate can be pushed to this source."""
        pass

    @abstractmethod
    def push_predicates(self, predicates: List[Predicate]) -> None:
        """Push filter predicates to the data source."""
        pass

    @abstractmethod
    def select_columns(self, columns: Set[str]) -> None:
        """Set columns to read (projection pushdown)."""
        pass

    @abstractmethod
    def get_partitions(self) -> List[PartitionInfo]:
        """Get partition information for partition pruning."""
        pass

    @abstractmethod
    def execute(self) -> Iterator[Dict]:
        """Execute the query and return results."""
        pass
```

### ODPS Connector Example

```python
class ODPSConnector(DataConnector):
    """Connector for Alibaba ODPS (MaxCompute)."""

    def __init__(self, table: str, project: str, **credentials):
        self.table = table
        self.project = project
        self.odps = ODPS(**credentials)
        self._predicates = []
        self._columns = None

    def supports_predicate(self, predicate: Predicate) -> bool:
        # ODPS supports SQL-like predicates
        return predicate.type in {
            'length', 'range', 'equality', 'like', 'in'
        }

    def push_predicates(self, predicates: List[Predicate]) -> None:
        self._predicates.extend(predicates)

    def select_columns(self, columns: Set[str]) -> None:
        self._columns = columns

    def execute(self) -> Iterator[Dict]:
        sql = self._build_sql()
        return self.odps.execute_sql(sql)

    def _build_sql(self) -> str:
        columns = ', '.join(self._columns) if self._columns else '*'
        where = ' AND '.join(self._predicate_to_sql(p) for p in self._predicates)

        sql = f"SELECT {columns} FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        return sql

    def _predicate_to_sql(self, pred: Predicate) -> str:
        if pred.type == 'length':
            return f"LENGTH({pred.column}) BETWEEN {pred.min} AND {pred.max}"
        elif pred.type == 'range':
            return f"{pred.column} BETWEEN {pred.min} AND {pred.max}"
        elif pred.type == 'equality':
            return f"{pred.column} = '{pred.value}'"
        # ... more predicate types
```

### Supported Connectors

| Connector | Filter Pushdown | Projection | Partition Pruning |
|-----------|----------------|------------|-------------------|
| `LocalFileConnector` | No | Yes (Parquet) | No |
| `ParquetConnector` | Yes (row groups) | Yes | Yes |
| `ODPSConnector` | Yes (SQL) | Yes | Yes |
| `HuggingFaceConnector` | No | Yes | No |
| `S3Connector` | No | Yes (Parquet) | Yes (prefix) |

## Configuration

### Enabling Optimization

```yaml
# New recommended way
enable_optimizer: true
optimizer_strategies:
  - filter_pushdown      # Push filters to data source
  - projection_pushdown  # Only read needed columns
  - op_reorder          # Reorder by selectivity/cost
  - filter_fusion       # Fuse filters sharing intermediate vars
  - model_sharing       # Share models across operators

# Legacy (deprecated, auto-mapped to above)
op_fusion: true
fusion_strategy: probe  # Maps to: op_reorder + filter_fusion
```

### Strategy-specific Configuration

```yaml
optimizer_config:
  filter_pushdown:
    enabled: true
    # Only push filters with selectivity below threshold
    selectivity_threshold: 0.5

  op_reorder:
    enabled: true
    # Use runtime statistics if available
    use_runtime_stats: true

  filter_fusion:
    enabled: true
    # Minimum filters to fuse
    min_group_size: 2
```

## Cost Model

The optimizer uses a cost model to make decisions.

### Cost Factors

| Factor | Description | Unit |
|--------|-------------|------|
| `io_cost` | Cost of reading data | bytes |
| `cpu_cost` | Cost of computation | operations |
| `memory_cost` | Memory usage | bytes |
| `network_cost` | Data transfer cost | bytes |

### Operator Cost Estimation

```python
class CostModel:
    # Base costs per operation type (relative scale 0-1)
    BASE_COSTS = {
        'text_length_filter': 0.01,
        'words_num_filter': 0.02,
        'language_id_score_filter': 0.5,
        'perplexity_filter': 0.8,
        'image_size_filter': 0.3,
        'embedding_mapper': 0.9,
        'deduplicator': 1.0,
    }

    def estimate_cost(self, op: Operator, stats: TableStatistics) -> float:
        base = self.BASE_COSTS.get(op.name, 0.1)

        # Adjust for data characteristics
        if 'image' in op.name and stats.has_images:
            base *= stats.avg_image_size / 1_000_000  # MB

        return base * stats.row_count
```

## Integration with Execution

The optimization framework integrates with executors via the `OptimizationManager`:

```python
class OptimizationManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.strategies = self._load_strategies()
        self.connector = self._create_connector()

    def optimize(self, ops: List[Operator]) -> List[Operator]:
        # Build AST
        ast = PipelineAST.from_operators(ops)

        # Apply optimization strategies
        for strategy in self.strategies:
            ast = strategy.optimize(ast, self.connector)

        # Convert back to operators
        return ast.to_operators()
```

For execution details (partitioning, checkpointing, event logging), see [PartitionAndCheckpoint.md](./PartitionAndCheckpoint.md).

## Performance Impact

### Expected Optimization Impact by Data Size

| Dataset Size | Filter Pushdown | Op Reorder | Fusion |
|-------------|-----------------|------------|--------|
| < 1 GB | Low | Low-Medium | Low |
| 1-100 GB | Medium-High | Medium-High | Medium |
| > 100 GB | High | High | Medium |

> **Note**: These are expected relative impacts, not measured benchmarks. Actual performance depends on workload characteristics such as filter selectivity, operator costs, and data source capabilities. We encourage users to benchmark on their specific use cases and share results.

## Future Work

1. **Adaptive Optimization** - Collect runtime statistics and re-optimize
2. **Incremental Processing** - Only process changed data
3. **Multi-query Optimization** - Optimize across multiple pipelines
4. **GPU-aware Scheduling** - Optimize for GPU/CPU placement
5. **Distributed Join Optimization** - For deduplication across partitions

## Related Documentation

- [PartitionAndCheckpoint.md](./PartitionAndCheckpoint.md) - Execution, checkpointing, and fault tolerance
- [Distributed.md](./Distributed.md) - Distributed processing with Ray
- [JobManagement.md](./JobManagement.md) - Job lifecycle management
