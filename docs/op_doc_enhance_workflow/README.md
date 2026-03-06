## Operator Documentation Enhancement Workflow

## 算子文档增强工作流

This workflow automates the generation and enhancement of operator (OP) documentation for Data-Juicer. It combines **runtime example capture**, **LLM-powered docstring rewriting**, and **bilingual translation** to produce rich, standardized Markdown documentation for every operator.

本工作流用于自动生成和增强 Data-Juicer 的算子（OP）文档。它结合了**运行时示例捕获**、**LLM 驱动的 docstring 重写**和**双语翻译**，为每个算子生成丰富、标准化的 Markdown 文档。

---

## Overview 概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Workflow Pipeline                           │
│                                                                 │
│  Layer 1 — Example Capture (数据采集层)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  pytest tests/ops/ --capture-op-examples                  │  │
│  │      ↓  conftest_capture.py + capture_examples.py         │  │
│  │  examples.jsonl  (runtime I/O snapshots, streaming)       │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↓                                                       │
│  Layer 2 — Doc Generation (文档生成层)                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  generate_op_details.py                                   │  │
│  │      ├── Load examples via utils/example_loader.py        │  │
│  │      ├── Scan all operators via OPSearcher                │  │
│  │      ├── (Optional) Rewrite docstrings via LLM            │  │
│  │      ├── Translate descriptions to Chinese via LLM        │  │
│  │      └── Render Markdown via Jinja2 template              │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↓                                                       │
│  docs/operators/{type}/{op_name}.md  (output)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure 目录结构

```
docs/op_doc_enhance_workflow/
├── conftest_capture.py          # Pytest plugin: activate capture via --capture-op-examples
├── capture_examples.py          # Runtime hook: monkey-patch operators to capture I/O
├── examples.jsonl               # Unified example storage (JSONL, generated)
├── generate_op_details.py       # Main entry: generate all operator Markdown docs
├── rewrite_op_docstrings.py     # LLM-based docstring rewriting
├── runner.py                    # [DEPRECATED] Old AST-based example runner
├── examples.json                # [DEPRECATED] Old nested-dict example storage
├── templates/
│   └── op_doc.md.j2             # Jinja2 template for operator docs
└── utils/
    ├── example_ir.py            # ExampleIR dataclass & MediaAsset
    ├── example_loader.py        # Load, normalize, select & render captured examples
    ├── llm_service.py           # LLM calls: translation & example selection
    ├── md_parser.py             # Parse existing operator Markdown files
    ├── model.py                 # LLM model wrapper (qwen3-max via API)
    ├── prompts.py               # Centralized LLM prompt templates
    └── view_model.py            # Render ExampleIR to HTML for Markdown
```

---

## Prerequisites 前置条件

1. **Install Data-Juicer** with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. **Configure LLM API**: The workflow uses `qwen3-max` by default (configured in `utils/model.py`). Ensure the corresponding API key / environment variables are set according to Data-Juicer's model configuration.

3. **Install additional dependencies**:

   ```bash
   pip install jinja2 fire
   ```

---

## Usage 使用方法

### Step 1: Capture Runtime Examples 捕获运行时示例

Run the operator unit tests with the `--capture-op-examples` flag to record real input/output data:

使用 `--capture-op-examples` 标志运行算子单元测试，以记录真实的输入/输出数据：

```bash
pytest tests/ops/ \
    -p docs.op_doc_enhance_workflow.conftest_capture \
    --capture-op-examples
```

This produces an `examples.jsonl` file containing the runtime I/O snapshots for each operator's test cases. The capture automatically resumes from existing data — already-captured test cases are skipped.

这将生成 `examples.jsonl` 文件，包含每个算子测试用例的运行时 I/O 快照。捕获会自动从已有数据恢复——已捕获的测试用例会被跳过。

### Step 2: Generate Operator Documentation 生成算子文档

Use the main entry script `generate_op_details.py` to generate all operator docs:

使用主入口脚本 `generate_op_details.py` 生成所有算子文档：

```bash
# Basic usage — generate docs using examples.jsonl
python docs/op_doc_enhance_workflow/generate_op_details.py gen

# With LLM-powered docstring rewriting
python docs/op_doc_enhance_workflow/generate_op_details.py gen --rewrite_docstring=True

# With LLM-powered example explanations
python docs/op_doc_enhance_workflow/generate_op_details.py gen --explain_examples=True

# Specify a custom captured examples file
python docs/op_doc_enhance_workflow/generate_op_details.py gen \
    --captured_examples_path=/path/to/examples.jsonl

# Enable all LLM enhancements
python docs/op_doc_enhance_workflow/generate_op_details.py gen \
    --rewrite_docstring=True \
    --explain_examples=True
```

Generated docs are written to `docs/operators/{type}/{op_name}.md`.

生成的文档将输出到 `docs/operators/{type}/{op_name}.md`。

### Step 3 (Optional): Rewrite Docstrings Only 仅重写 Docstring

To rewrite operator docstrings in source code without generating full docs:

仅重写源代码中的算子 docstring，而不生成完整文档：

```python
from rewrite_op_docstrings import update_op_docstrings_with_names

# Rewrite docstrings for specific operators
update_op_docstrings_with_names(["word_num_filter", "image_blur_mapper"])
```

---

## `gen()` Parameters 参数说明

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rewrite_docstring` | `bool` | `False` | Use LLM to rewrite operator class docstrings for clarity. 使用 LLM 重写算子类的 docstring 以提高清晰度。 |
| `explain_examples` | `bool` | `False` | Use LLM to generate bilingual explanations for each example. 使用 LLM 为每个示例生成双语解释。 |
| `captured_examples_path` | `str` | `captured_examples.json` | Path to the captured examples JSON/JSONL file. 捕获的示例 JSON/JSONL 文件路径。 |

---

## Generated Document Structure 生成文档结构

Each operator doc follows the Jinja2 template (`templates/op_doc.md.j2`) and includes:

每个算子文档遵循 Jinja2 模板，包含以下部分：

1. **Title & Description** — Bilingual operator description (English + Chinese)
2. **Type & Tags** — Operator type (e.g., `filter`, `mapper`) and associated tags
3. **Parameter Configuration** — Table of all `__init__` parameters with types, defaults, and descriptions
4. **Effect Demonstration** — Real input/output examples rendered as HTML cards with:
   - Operator instantiation code
   - Input data (text, images, videos, audios, metadata)
   - Output data
   - Optional bilingual explanation
5. **Related Links** — Links to source code, unit tests, and the operator list

---

## Key Design Details 关键设计细节

### Example Selection Logic 示例选择逻辑

- If no captured examples exist for an operator, existing examples from the Markdown are preserved.
- If the existing Markdown already has ≥ 2 examples whose methods all appear in the captured data, they are kept as-is.
- Parallel / numpy variant test methods (containing `parallel` or `np`) are automatically skipped.
- At most **2 examples** are selected per operator.

### Bilingual Translation 双语翻译

- English descriptions are batch-translated to Simplified Chinese via LLM.
- Batches are split when total text exceeds 5,000 characters.
- Terminology rules: `operator` → `算子`, `Hugging Face` / `token` remain in English.

### LLM Configuration LLM 配置

- Default model: **qwen3-max** (configurable in `utils/model.py`)
- The model is accessed via Data-Juicer's `model_utils` API layer
- All prompts are centralized in `utils/prompts.py` for easy customization

### Excluded Operators 排除的算子

The following operators are excluded from example generation due to special requirements:

以下算子因特殊需求被排除在示例生成之外：

- `llm_task_relevance_filter`
- `in_context_influence_filter`
- `text_embd_similarity_filter`
- `audio_add_gaussian_noise_mapper`
- `image_blur_mapper`
- `image_captioning_from_gpt4v_mapper`

---

## Architecture 架构分层

The workflow is organized into three decoupled layers:

工作流分为三个解耦的层：

| Layer | Files | Responsibility |
|-------|-------|----------------|
| **Layer 1: Capture** | `conftest_capture.py`, `capture_examples.py` | Run tests, monkey-patch operators, capture I/O to `examples.jsonl` |
| **Layer 2: Generation** | `generate_op_details.py`, `utils/example_loader.py` | Scan operators, load & process examples, (optional) LLM rewrite/translate, render Markdown |

---

## Troubleshooting 常见问题

- **"No captured examples found"**: Run `pytest tests/ops/ -p docs.op_doc_enhance_workflow.conftest_capture --capture-op-examples` first to generate the data.
- **Migrating from old format**: If you have an `examples.json` file, run `python docs/op_doc_enhance_workflow/migrate_legacy_examples.py` to convert it.
- **LLM translation failures**: Check API key configuration and network connectivity. The workflow retries up to 3 times on failure.
- **Missing operator in output**: Ensure the operator is registered via `@OPERATORS.register_module()` decorator so that `OPSearcher` can discover it.
