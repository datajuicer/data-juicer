# 算子插件



## 概述

Data-Juicer 支持通过 Python 标准的 [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) 机制发现并加载来自外部包的算子。这使得算子可以作为独立的 Python 包进行分发、版本管理和安装（例如发布到 PyPI 或私有索引），同时仍然注册到全局 `OPERATORS` 注册表中，并可在任意 recipe 中通过名称引用。

## 工作原理

在 `import data_juicer.ops` 时，`load_op_plugins()` 会扫描所有已安装发行包中声明在 `data_juicer.ops` 分组下的 entry points，并导入其指向的模块。导入插件模块会执行其模块级的 `@OPERATORS.register_module(...)` 装饰器，在 `init_configs()` 读取注册表之前完成算子注册。

关键特性：

- **自动发现**：插件包安装后，其算子即可使用，无需额外配置。
- **故障隔离**：若某个插件导入失败（例如缺少依赖），仅记录警告并跳过，不影响其余流程的执行。
- **向后兼容**：未安装任何插件时，行为保持不变。

## 插件 vs. `custom_operator_paths`

Data-Juicer 提供两种互补的方式来使用库外算子：

| | 算子插件（entry points） | `custom_operator_paths` |
|---|---|---|
| 分发方式 | 可安装的包（PyPI / 私有索引） | 本地 `.py` 文件或包目录 |
| 发现方式 | 安装后自动发现 | 在 CLI / YAML 中显式指定路径 |
| 适用场景 | 可复用、可版本化、需要共享的算子 | 快速的本地或一次性算子 |
| 配置 | 无需额外配置 | `--custom-operator-paths` 或 YAML 中的 `custom_operator_paths:` |

## 开发算子插件

### 1. 包结构

```
my-dj-ops/
├── pyproject.toml
└── my_dj_ops/
    └── __init__.py        # 定义并注册算子
```

### 2. 实现并注册算子

算子的编写方式与内置算子一致：继承基类（`Mapper`、`Filter`、`Deduplicator` 等），并使用 `@OPERATORS.register_module(<op_name>)` 装饰器注册。重型依赖应通过 `LazyLoader` 延迟加载，避免在模块导入时直接引入。

```python
# my_dj_ops/__init__.py
from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("my_upper_mapper")
class MyUpperMapper(Mapper):
    """将每个样本的文本转为大写。"""

    _batched_op = True

    def process_batched(self, samples):
        samples[self.text_key] = [t.upper() for t in samples[self.text_key]]
        return samples
```

### 3. 声明 Entry Point

在 `pyproject.toml` 中，将模块暴露到 `data_juicer.ops` 分组下。entry point 的取值需指向一个模块（或对象），其导入会触发 `@OPERATORS.register_module` 调用——指向包的 `__init__` 是最简单的做法。

```toml
[project]
name = "my-dj-ops"
version = "0.1.0"
dependencies = ["py-data-juicer"]

[project.entry-points."data_juicer.ops"]
my_dj_ops = "my_dj_ops"
```

### 4. 安装并使用

```bash
pip install -e .        # 或：pip install my-dj-ops
```

然后即可在任意 recipe 中按名称引用该算子，default 和 Ray 执行器均支持：

```yaml
process:
  - my_upper_mapper: {}
```

## 注意事项与最佳实践

- **算子名称唯一性**：注册的算子名（如 `my_upper_mapper`）不得与内置算子或其他插件冲突。
- **声明 `py-data-juicer` 依赖**：在插件的 `dependencies` 中列出，以确保基类与注册表可用。
- **重型依赖保持延迟加载**：将重型 ML 库写入插件的 `dependencies`，但在运行时通过 `LazyLoader` 加载，与核心算子保持相同的约定。
- **GPU 与不可 fork 算子**：插件中可照常使用 `_accelerator = "cuda"` 或 `use_cuda()`，执行器会根据这些属性选择相应的多进程上下文。
