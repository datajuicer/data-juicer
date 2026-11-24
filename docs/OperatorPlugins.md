# Operator Plugins


## Overview

Data-Juicer supports discovering and loading operators from external Python packages via the standard [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) mechanism. This allows operators to be distributed, versioned, and installed independently (e.g. on PyPI or a private index) while still appearing in the global `OPERATORS` registry and being usable in any recipe by name.

## How It Works

When `data_juicer.ops` is imported, `load_op_plugins()` scans all installed distributions for entry points declared under the `data_juicer.ops` group and imports the referenced modules. Importing a plugin module executes its module-level `@OPERATORS.register_module(...)` decorators, registering operators into the global registry before `init_configs()` reads it.

Key properties:

- **Automatic discovery** — once a plugin package is installed, its operators become available without additional configuration.
- **Fault isolation** — if a plugin fails to import (e.g. due to a missing dependency), it is skipped with a warning and does not affect the rest of the pipeline.
- **Backward compatibility** — when no plugin is installed, behavior remains unchanged.

## Plugins vs. `custom_operator_paths`

Data-Juicer offers two complementary ways to use out-of-tree operators:

| | Operator Plugins (entry points) | `custom_operator_paths` |
|---|---|---|
| Distribution | Installable package (PyPI / private index) | Local `.py` file or package directory |
| Discovery | Automatic on `pip install` | Explicit path in CLI / YAML |
| Best for | Reusable, versioned, shared operators | Quick local or one-off operators |
| Configuration | None required | `--custom-operator-paths` or `custom_operator_paths:` in YAML |

## Writing an Operator Plugin

### 1. Package Layout

```
my-dj-ops/
├── pyproject.toml
└── my_dj_ops/
    └── __init__.py        # defines & registers operators
```

### 2. Implement and Register Operators

Operators follow the same conventions as built-in ones: inherit from a base class (`Mapper`, `Filter`, `Deduplicator`, etc.) and register with `@OPERATORS.register_module(<op_name>)`. Heavy dependencies should be loaded lazily via `LazyLoader` rather than imported at module level.

```python
# my_dj_ops/__init__.py
from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("my_upper_mapper")
class MyUpperMapper(Mapper):
    """Uppercases the text of each sample."""

    _batched_op = True

    def process_batched(self, samples):
        samples[self.text_key] = [t.upper() for t in samples[self.text_key]]
        return samples
```

### 3. Declare the Entry Point

In `pyproject.toml`, expose the module under the `data_juicer.ops` group. The entry point value must reference a module (or object) whose import triggers the `@OPERATORS.register_module` calls — pointing to the package `__init__` is the simplest approach.

```toml
[project]
name = "my-dj-ops"
version = "0.1.0"
dependencies = ["py-data-juicer"]

[project.entry-points."data_juicer.ops"]
my_dj_ops = "my_dj_ops"
```

### 4. Install and Use

```bash
pip install -e .        # or: pip install my-dj-ops
```

Then reference the operator by name in any recipe. Both the default and Ray executors are supported:

```yaml
process:
  - my_upper_mapper: {}
```

## Notes and Best Practices

- **Operator name uniqueness**: the registered name (e.g. `my_upper_mapper`) must not collide with a built-in operator or another plugin.
- **Depend on `py-data-juicer`**: list it in your plugin's `dependencies` so that base classes and the registry are available.
- **Keep heavy dependencies lazy**: declare heavy ML libraries in your plugin's `dependencies`, but load them at runtime via `LazyLoader`, following the same convention used by core operators.
- **GPU and unforkable operators**: plugins can use `_accelerator = "cuda"` or `use_cuda()` as usual; the executor selects the appropriate multiprocessing context based on these attributes.
