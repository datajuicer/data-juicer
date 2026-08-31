# Installation Guide

This guide covers all the ways to install Data-Juicer: from PyPI, from source, or via Docker. Pick the scenario that matches your work — a minimal install is enough to get started, and you can add capabilities later.

## 1. Prerequisites

Ensure your environment meets the following conditions:

- Python >= 3.10
- Git (for source installation)
- uv (recommended package installer)

Install `uv` if you do not have it yet:

```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

Run the following `uv pip` and CLI commands in an activated virtual environment. Create and activate one in a Linux/macOS shell:

```bash
uv venv
source .venv/bin/activate
```

## 2. Basic Installation

Data-Juicer is available on PyPI:

```bash
uv pip install py-data-juicer
```

This will install the core data processing capabilities, which provide:

- Data loading and manipulation
- File system operations
- Parallel processing
- Basic I/O and utilities

> 💡 **Which install do I need?**
> The basic install covers text-oriented recipes built from the operators that need no extra models. If an operator complains about a missing package, add the matching extra from the next section — or let Data-Juicer install it on demand (see [Section 5](#5-installation-for-specific-ops)).

## 3. Scenario-based Installation

Extras let you install only the capabilities you need. For the exact package list behind each extra, see [pyproject.toml](../../pyproject.toml).

**Core ML & DL** — includes PyTorch, Transformers, VLLM, etc.

```bash
# Generic ML/DL capabilities
uv pip install "py-data-juicer[generic]"
```

**Domain-Specific Features**

```bash
# Computer Vision
uv pip install "py-data-juicer[vision]"

# Natural Language Processing
uv pip install "py-data-juicer[nlp]"

# Audio Processing
uv pip install "py-data-juicer[audio]"
```

**Additional Components**

```bash
# Distributed Computing
uv pip install "py-data-juicer[distributed]"

# AI Services & APIs
uv pip install "py-data-juicer[ai_services]"
```

**Development Tools**

```bash
# Development & Testing
uv pip install "py-data-juicer[dev]"
```

### Common Combinations

Extras compose, so a typical setup is one line:

```bash
# Text processing
uv pip install "py-data-juicer[generic,nlp]"

# Vision processing
uv pip install "py-data-juicer[generic,vision]"

# Full processing pipeline
uv pip install "py-data-juicer[generic,nlp,vision,distributed]"

# Everything except sandbox
uv pip install "py-data-juicer[all]"
```

## 4. Installation From Source

Install from source to use the latest features and updates:

```bash
# Clone repository
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
uv venv
source .venv/bin/activate
uv pip install -e .

# You can install specific domain as well
uv pip install -e ".[vision]"
```

This will install Data-Juicer in editable mode, so your local edits take effect without reinstalling. It also gives you the `tools/`, `demos/`, and `app.py` entry points that the [Quick Start](QuickStart.md) refers to.

> 💡 **Note**: Use `-e` when installing from source. Some tools and demos are only available in a source checkout.

## 5. Installation for Specific OPs

As the number of operators grows, installing every dependency becomes heavy. Instead of `uv pip install -e ".[all]"`, you can install only what a given recipe needs:

- **Automatic minimal dependency installation** — Data-Juicer installs missing dependencies at runtime as operators need them. This runs immediately, but may lead to dependency conflicts.

- **Manual minimal dependency installation** — resolve the dependencies for one specific recipe ahead of time:

  ```shell
  # only for installation from source
  python tools/dj_install.py --config path_to_your_data-juicer_config_file

  # use command line tool
  dj-install --config path_to_your_data-juicer_config_file
  ```

  The tool scans direct imports and LazyLoader declarations in the configured operators' source files. Helper-module dependencies and model weights may be loaded when operators run. For offline use, run the required recipe in advance and cache its dependencies and models.

## 6. Installation Using Docker

You can either pull the pre-built image or build it yourself.

**Pull from DockerHub:**

```shell
docker pull datajuicer/data-juicer:<version_tag>
```

If you cannot reach DockerHub, use a registry mirror:

```shell
docker pull <other_registry_mirror>/datajuicer/data-juicer:<version_tag>
```

**Or build the image** including the latest `data-juicer` with the provided [Dockerfile](../../Dockerfile):

```shell
docker build -t datajuicer/data-juicer:<version_tag> .
```

> 💡 **Note**: `<version_tag>` follows the release version tag, e.g. `v0.2.0`.

Once the container is running, use the same CLI commands (`dj-process`, `dj-analyze`, etc.) as on a native installation.

## 7. Verify the Installation

```python
import data_juicer as dj
print(dj.__version__)
```

If this prints a version, you are ready to run your first job — continue to the [Quick Start](QuickStart.md).

## Frequently Asked Questions

### Q1: Which extras should I install?

**A**: Install only what you need and combine extras as required (e.g. `[generic,nlp]`). Use `all` when you want everything except the sandbox. If you are unsure, start minimal and let `dj-install` resolve a specific recipe for you.

### Q2: A video-related operator fails to run

**A**: Video operators require **FFmpeg** to be installed and reachable via `$PATH`:

- Install it with a package manager (e.g. `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS), or from the [official FFmpeg site](https://ffmpeg.org/download.html).
- Verify it by running `ffmpeg` from your terminal.

### Q3: How do I install the sandbox?

**A**: The sandbox is installed separately from the `all` extra because it targets experimental, data-model co-development workflows. See the [sandbox documentation](https://datajuicer.github.io/data-juicer-sandbox/en/main/index.html).

### Q4: Where do I get help?

**A**: Check the documentation and existing [issues](https://github.com/datajuicer/data-juicer/issues) first, then open a GitHub issue if needed. You can also join the community channels linked in the [README](../../README.md) for discussions.
