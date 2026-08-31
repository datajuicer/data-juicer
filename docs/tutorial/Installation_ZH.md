# 安装

本指南涵盖 Data-Juicer 的所有安装方式：通过 PyPI、从源码，或使用 Docker。请选择与你的工作场景匹配的方式——最小安装就足以开始使用，后续可以随时按需添加能力。

## 1. 前置条件

请确保你的环境满足以下条件：

- Python >= 3.10
- Git（源码安装需要）
- uv（推荐的包安装器）

如果你还没有 `uv`，请先安装：

```bash
# 使用 curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip
pip install uv
```

以下 `uv pip` 和 CLI 命令均在已激活的虚拟环境中运行。使用 Linux/macOS shell 创建并激活环境：

```bash
uv venv
source .venv/bin/activate
```

## 2. 基础安装

Data-Juicer 已上架 PyPI：

```bash
uv pip install py-data-juicer
```

这将安装核心数据处理能力，包括：

- 数据加载和操作
- 文件系统操作
- 并行处理
- 基础 I/O 和工具

> 💡 **我该装哪个？**
> 基础安装可覆盖那些无需额外模型的算子所构成的文本类菜谱。如果某个算子报缺包，就从下一节添加对应的 extra——或者让 Data-Juicer 按需自动安装（见[第 5 节](#5-特定算子的安装)）。

## 3. 场景化安装

Extra 让你只安装所需的能力。每个 extra 背后的确切包列表，请参考 [pyproject.toml](../../pyproject.toml)。

**核心 ML & DL** —— 包括 PyTorch、Transformers、VLLM 等。

```bash
# 通用 ML/DL 能力
uv pip install "py-data-juicer[generic]"
```

**领域特定功能**

```bash
# 计算机视觉
uv pip install "py-data-juicer[vision]"

# 自然语言处理
uv pip install "py-data-juicer[nlp]"

# 音频处理
uv pip install "py-data-juicer[audio]"
```

**附加组件**

```bash
# 分布式计算
uv pip install "py-data-juicer[distributed]"

# AI 服务和 API
uv pip install "py-data-juicer[ai_services]"
```

**开发工具**

```bash
# 开发和测试
uv pip install "py-data-juicer[dev]"
```

### 常见组合

Extra 可以组合，典型配置只需一行：

```bash
# 文本处理
uv pip install "py-data-juicer[generic,nlp]"

# 视觉处理
uv pip install "py-data-juicer[generic,vision]"

# 完整处理流程
uv pip install "py-data-juicer[generic,nlp,vision,distributed]"

# 除沙盒外的全部功能
uv pip install "py-data-juicer[all]"
```

## 4. 从源码安装

想使用最新的特性与更新时，请从源码安装：

```bash
# 克隆仓库
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
uv venv
source .venv/bin/activate
uv pip install -e .

# 同样也可以安装指定的额外领域依赖
uv pip install -e ".[vision]"
```

这将以可编辑模式安装 Data-Juicer，因此你的本地修改无需重新安装即可生效。同时你还会获得[快速上手](QuickStart_ZH.md)中提到的 `tools/`、`demos/` 和 `app.py` 等入口。

> 💡 **注意**：从源码安装时请使用 `-e`。部分工具和示例仅在源码检出环境下可用。

## 5. 特定算子的安装

随着算子数量的增长，安装全部依赖会变得非常庞大。除了 `uv pip install -e ".[all]"`，你也可以只安装某份菜谱所需的依赖：

- **自动最小依赖安装** —— Data-Juicer 在运行时按算子需要自动安装缺失依赖。这可以立即执行，但可能导致依赖冲突。

- **手动最小依赖安装** —— 提前为某一份具体菜谱解析依赖：

  ```shell
  # 仅适用于从源码安装
  python tools/dj_install.py --config path_to_your_data-juicer_config_file

  # 使用命令行工具
  dj-install --config path_to_your_data-juicer_config_file
  ```

  工具扫描配置中算子源码里的直接导入和 LazyLoader 声明，不会递归收集辅助模块中的全部依赖。运行时仍可能安装额外包或下载模型权重；命令成功不代表菜谱可以离线运行。

## 6. 使用 Docker 安装

你可以拉取预构建镜像，也可以自行构建。

**从 DockerHub 拉取：**

```shell
docker pull datajuicer/data-juicer:<version_tag>
```

如果无法连接 DockerHub，请使用镜像源：

```shell
docker pull <other_registry_mirror>/datajuicer/data-juicer:<version_tag>
```

**或者构建镜像**，使用提供的 [Dockerfile](../../Dockerfile) 构建包含最新 `data-juicer` 的镜像：

```shell
docker build -t datajuicer/data-juicer:<version_tag> .
```

> 💡 **注意**：`<version_tag>` 与发布版本标签一致，例如 `v0.2.0`。

容器启动后，可使用与本地安装相同的 CLI 命令（`dj-process`、`dj-analyze` 等）。

## 7. 验证安装

```python
import data_juicer as dj
print(dj.__version__)
```

如果这里打印出版本号，你就可以运行第一个任务了——继续阅读[快速上手](QuickStart_ZH.md)。

## 常见问题

### Q1：我该安装哪些 extra？

**A**：只安装你需要的，并按需组合（例如 `[generic,nlp]`）。想要除沙盒外的全部功能时使用 `all`。如果不确定，先从最小安装开始，再用 `dj-install` 为某份具体菜谱解析依赖。

### Q2：视频相关算子运行失败

**A**：视频算子需要安装 **FFmpeg** 并确保可通过 `$PATH` 访问：

- 使用包管理器安装（例如 Debian/Ubuntu 上 `sudo apt install ffmpeg`，macOS 上 `brew install ffmpeg`），或访问 [FFmpeg 官网](https://ffmpeg.org/download.html)。
- 在终端运行 `ffmpeg` 命令来验证。

### Q3：如何安装沙盒？

**A**：沙盒独立于 `all` extra 单独安装，因为它面向实验性的数据-模型协同开发流程。请参见[沙盒文档](https://datajuicer.github.io/data-juicer-sandbox/zh_CN/main/index_ZH.html)。

### Q4：在哪里获取帮助？

**A**：请先查看文档和已有 [issues](https://github.com/datajuicer/data-juicer/issues)，必要时再创建 GitHub issue。你也可以加入 [README](../../README_ZH.md) 中链接的社区频道参与讨论。
