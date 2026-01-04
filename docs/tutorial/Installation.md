# Installation Guide

## Prerequisites

- Python >= 3.10 & <= 3.12
- Git (for source installation)
- uv (recommended package installer)
  - uv can be installed by:

```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

## Basic DJ Installation

Data-Juicer is now available on PyPI:

```bash
uv pip install py-data-juicer
```

The minimal installation includes core data processing capabilities, which provides:
- Data loading and manipulation
- File system operations
- Parallel processing
- Basic I/O and utilities


## Scenario-based Installation
For component details, plz refer to [pyproject.toml](../../pyproject.toml).

**Core ML & DL**
```bash
# Generic ML/DL capabilities
uv pip install "py-data-juicer[generic]"
```
Includes: PyTorch, Transformers, VLLM, etc.

**Domain-Specific Features**

```bash
# Computer Vision
uv pip install "py-data-juicer[vision]"

# Natural Language Processing
uv pip install "py-data-juicer[nlp]"

# Audio Processing
uv pip install "py-data-juicer[audio]"

**Additional Components**

```bash
# Distributed Computing
uv pip install "py-data-juicer[distributed]"

# AI Services & APIs
uv pip install "py-data-juicer[ai_services]"

**Development Tools**
```bash
# Development & Testing
uv pip install "py-data-juicer[dev]"
```

## Common Installation Patterns

**1. Text Processing Setup**
```bash
uv pip install "py-data-juicer[generic,nlp]"
```

**2. Vision Processing Setup**
```bash
uv pip install "py-data-juicer[generic,vision]"
```

**3. Full Processing Pipeline**
```bash
uv pip install "py-data-juicer[generic,nlp,vision,distributed]"
```


**4. Complete Installation**
```bash
# Install all features (except sandbox)
uv pip install "py-data-juicer[all]"
```


## Installation From Source

If you want to use the latest features and updates, Data-Juicer can be installed from source as well:
```bash
# Clone repository
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
uv pip install -e .

# You can install specific domain as well
uv pip install -e ".[vision]"
```

**Note**: It's suggested to used `-e` to enable editable mode when installing from source.


## Installation for Specific OPs
Besides the scenarios-based installation, we also provide OP-based and recipe-based manners.

- Install dependencies for specific OPs

With the growth of the number of OPs, the dependencies of all OPs become very heavy. Instead of using the command `pip install -v -e .[all]` to install all dependencies,
we provide two alternative, lighter options:

  - Automatic Minimal Dependency Installation: During the execution of Data-Juicer, minimal dependencies will be automatically installed. This allows for immediate execution, but may potentially lead to dependency conflicts.

  - Manual Minimal Dependency Installation: To manually install minimal dependencies tailored to a specific execution configuration, run the following command:
    ```shell
    # only for installation from source
    python tools/dj_install.py --config path_to_your_data-juicer_config_file

    # use command line tool
    dj-install --config path_to_your_data-juicer_config_file
    ```

## Installation Using Docker

- You can
  - either pull our pre-built image from DockerHub:
    ```shell
    docker pull datajuicer/data-juicer:<version_tag>
    ```
  
    - if you can not connect ot DockerHub, please use other registry mirrors (you can find some from the Internet):
    ```shell
    docker pull <other_registry_mirror>/datajuicer/data-juicer:<version_tag>
    ```

  - or run the following command to build the docker image including the
    latest `data-juicer` with provided [Dockerfile](../../Dockerfile):

    ```shell
    docker build -t datajuicer/data-juicer:<version_tag> .
    ```

  - The format of `<version_tag>` is like `v0.2.0`, which is the same as the release version tag.

## Notes & Troubleshooting

0. **installation check**
   
```python
import data_juicer as dj
print(dj.__version__)
```

1. **Modular Installation**
   - Install only what you need
   - Combine components as required
   - Use `all` for complete installation

2. **Sandbox Environment**
   - Separate installation for experimental features
   - Will be provided as micro-services in future

3. **For Video-related Operators**
   - Before using video-related operators, **FFmpeg** should be installed and accessible via the $PATH environment variable.
   - You can install FFmpeg using package managers(e.g. sudo apt install ffmpeg on Debian/Ubuntu, brew install ffmpeg on OS X) or visit the [official ffmpeg link](https://ffmpeg.org/download.html).
   - Check if your environment path is set correctly by running the ffmpeg command from the terminal.

4. **Getting Help**  
   - Plz check documentation/issues first
   - Create GitHub issues when necessary
   - Join community channels for discussions