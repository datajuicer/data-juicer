# 安装

## 前置条件

- Python >= 3.10 & <= 3.12
- Git (用于源码安装)
- uv (推荐的包安装器)
  - uv 可以通过以下方式安装：

```bash
# 使用 curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip
pip install uv
```

## 基础安装

Data-Juicer 已上架 PyPI：

```bash
uv pip install py-data-juicer
```

最小安装包含核心数据处理能力，这将提供：
- 数据加载和操作
- 文件系统操作
- 并行处理
- 基础 I/O 和工具

## 场景化安装
组件详情请参考 [pyproject.toml](../../pyproject.toml)。

**核心 ML & DL**
```bash
# 通用 ML/DL 能力
uv pip install "py-data-juicer[generic]"
```
包括：PyTorch、Transformers、VLLM 等。

**领域特定功能**

```bash
# 计算机视觉
uv pip install "py-data-juicer[vision]"

# 自然语言处理
uv pip install "py-data-juicer[nlp]"

# 音频处理
uv pip install "py-data-juicer[audio]"

**附加组件**

```bash
# 分布式计算
uv pip install "py-data-juicer[distributed]"

# AI 服务和 API
uv pip install "py-data-juicer[ai_services]"

**开发工具**
```bash
# 开发和测试
uv pip install "py-data-juicer[dev]"
```

## 常见安装模式

**1. 文本处理设置**
```bash
uv pip install "py-data-juicer[generic,nlp]"
```

**2. 视觉处理设置**
```bash
uv pip install "py-data-juicer[generic,vision]"
```

**3. 完整处理流程**
```bash
uv pip install "py-data-juicer[generic,nlp,vision,distributed]"
```

**4. 完整安装**
```bash
# 安装所有功能（除沙盒外）
uv pip install "py-data-juicer[all]"
```

## 从源码安装

如果想要使用最新的特性与更新，也可以从源码安装 Data-Juicer：
```bash
# 克隆仓库
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
uv pip install -e .

# 同样也可以安装指定的额外领域的依赖
uv pip install -e ".[vision]"
```

**注意：**从源码安装时，建议使用 `-e` 模式打开开发模式。

## 特定算子安装
除了基于场景的安装外，我们还提供基于算子和基于菜谱的安装方式。

- 安装特定算子的依赖

随着算子数量的增长，所有算子的依赖变得非常庞大。除了使用 `pip install -v -e .[all]` 安装所有依赖外，
我们提供了两个更轻量级的替代方案：

  - 自动最小依赖安装：在 Data-Juicer 执行过程中，将自动安装最小依赖。这允许立即执行，但可能会导致依赖冲突。

  - 手动最小依赖安装：要手动安装针对特定执行配置的最小依赖，请运行以下命令：
    ```shell
    # 仅适用于从源码安装
    python tools/dj_install.py --config path_to_your_data-juicer_config_file

    # 使用命令行工具
    dj-install --config path_to_your_data-juicer_config_file
    ```

## 使用 Docker 安装

- 您可以
  - 从 DockerHub 拉取预构建镜像：
    ```shell
    docker pull datajuicer/data-juicer:<version_tag>
    ```
  
    - 如果无法连接到 DockerHub，请使用其他镜像源（您可以在互联网上找到一些）：
    ```shell
    docker pull <other_registry_mirror>/datajuicer/data-juicer:<version_tag>
    ```

  - 或运行以下命令构建包含最新 `data-juicer` 的 docker 镜像，使用提供的 [Dockerfile](../../Dockerfile)：

    ```shell
    docker build -t datajuicer/data-juicer:<version_tag> .
    ```

  - `<version_tag>` 的格式类似于 `v0.2.0`，与发布版本标签相同。

## 注意事项和故障排除

0. **安装检查**
   
```python
import data_juicer as dj
print(dj.__version__)
```

1. **模块化安装**
   - 只安装您需要的组件
   - 根据需要组合组件
   - 使用 `all` 进行完整安装

2. **沙盒环境**
   - 实验性功能的单独安装
   - 未来将作为微服务提供

3. **视频相关算子**
   - 在使用视频相关算子之前，需要安装 **FFmpeg** 并确保可以通过 $PATH 环境变量访问。
   - 您可以使用包管理器安装 FFmpeg（例如在 Debian/Ubuntu 上使用 sudo apt install ffmpeg，在 OS X 上使用 brew install ffmpeg）或访问 [官方 ffmpeg 链接](https://ffmpeg.org/download.html)。
   - 通过从终端运行 ffmpeg 命令来检查您的环境路径是否正确设置。

4. **获取帮助**  
   - 请先查看文档/问题
   - 必要时创建 GitHub issues
   - 加入社区频道进行讨论
