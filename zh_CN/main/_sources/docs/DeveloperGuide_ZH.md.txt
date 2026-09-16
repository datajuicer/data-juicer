# 开发者指南

- [开发者指南](#开发者指南)
  - [1. 快速构建你自己的算子](#1-快速构建你自己的算子)
  - [2. 构建你自己的数据菜谱和配置项](#2-构建你自己的数据菜谱和配置项)
    - [2.1 丰富的配置源和类型提示](#21-丰富的配置源和类型提示)
    - [2.2 层次化的配置和帮助](#22-层次化的配置和帮助)
  - [3. 依赖管理](#3-依赖管理)
    - [3.1 安装 uv](#31-安装-uv)
    - [3.2 虚拟环境管理](#32-虚拟环境管理)
    - [3.3 添加新依赖](#33-添加新依赖)
    - [3.4 开发环境设置](#34-开发环境设置)
    - [3.5 延迟加载](#35-延迟加载)
  - [4. 为开源社区贡献](#4-为开源社区贡献)
    - [4.1 编码规范](#41-编码规范)
    - [4.2 文档规范](#42-文档规范)
    - [4.3 将你的新算子贡献到开源社区](#43-将你的新算子贡献到开源社区)
      - [4.3.1 提供算子基本功能（alpha版本）](#431-提供算子基本功能alpha版本)
      - [4.3.2 使算子更可用（beta版本）](#432-使算子更可用beta版本)
      - [4.3.3 使算子更快更完备（stable版本）](#433-使算子更快更完备stable版本)
    - [4.4 贡献您的新配方](#44-贡献您的新配方)

## 1. 快速构建你自己的算子

- Data-Juicer 支持每个人灵活、便捷定义自己的算子。
- 在实现新的算子之前，请参考已有 [算子池](Operators.md) 以避免不必要的重复。

> 以下示例的开发过程以直接在源码对应模块中添加算子为例。如果外部添加算子，可以通过传参`--custom-operator-paths` 或 yaml配置文件中配置`custom_operator_paths`参数注册新算子，例如：`custom_operator_paths: ['/path/to/new/op.py', '/path/to/new/ops/directory/]`。

下面以 "TextLengthFilter" 的算子（过滤仅包含预期文本长度的样本语料）为例，展示相应开发构建过程。

1. (可选) 如果该算子定义了某个统计变量，那么请在 `data_juicer/utils/constant.py` 文件中添加一个新的`StatsKeys`属性来统一保存管理。

```python
class StatsKeysConstant(object):
    ...              # other keys
    text_len = 'text_len'
```

2. 在 `data_juicer/ops/filter/` 目录下创建一个新的算子文件 `text_length_filter.py`，内容如下：
    - 因为它是一个 Filter 算子，所以需要继承 `base_op.py` 中的 `Filter` 基类，并用 `@OPERATORS.register_module(xx_op)` 装饰器标记，以实现自动注册。
    - 为了方便实现，我们可以按单样本处理的方式实现两个核心方法 `compute_stats_single` 和 `process_single`，它们的输入输出均为单个样本的字典结构。
    - 【进阶】如果你比较熟悉 Data-Juicer 中的batch化处理，你也可以通过覆写 `compute_stats_batched` 和 `process_batched` 方法直接实现它们的batch化版本，它的处理会比单样本版本稍快一些。它们的输入和输出则是按列存储的字典结构，其中包括多个样本 （详见下方 2.1.3 小节）。

    ```python
    import sys

    from jsonargparse.typing import PositiveInt

    from data_juicer.utils.constant import Fields, StatsKeys

    from data_juicer.ops.base_op import OPERATORS, Filter


    @OPERATORS.register_module('text_length_filter')
    class TextLengthFilter(Filter):
        """Filter to keep samples with total text length within a specific
        range."""

        def __init__(self,
                    min_len: PositiveInt = 10,
                    max_len: PositiveInt = sys.maxsize,
                    *args,
                    **kwargs):
            """
            Initialization method.

            :param min_len: The min text length in the filtering. samples
                will be filtered if their text length is below this
                parameter.
            :param max_len: The max text length in the filtering. samples
                will be filtered if their text length exceeds this
                parameter.
            :param args: extra args
            :param kwargs: extra args
            """
            super().__init__(*args, **kwargs)
            self.min_len = min_len
            self.max_len = max_len

        def compute_stats_single(self, sample):
            # check if it's computed already
            if StatsKeys.text_len in sample[Fields.stats]:
                return sample

            # compute text length and store it in the corresponding stats field
            sample[Fields.stats][StatsKeys.text_len] = len(sample[self.text_key])
            return sample

        def process_single(self, sample):
            if self.min_len <= sample[Fields.stats][StatsKeys.text_len] <= self.max_len:
                return True
            else:
                return False
    ```


3. 实现后，将其添加到 `data_juicer/ops/filter` 目录下 `__init__.py` 文件中的算子字典中：

```python
# other OPs
from .text_length_filter import TextLengthFilter  # import this new OP class
__all__ = [
    # other Ops
    "TextLengthFilter",  # add this new Op to __all__
]
```

4. （可选）当算子有第三方包依赖时，使用 `LazyLoader` 在运行时按需加载，而非在模块顶层 import。这与所有内置算子的惯例一致（参见 [`data_juicer/utils/lazy_loader.py`](../data_juicer/utils/lazy_loader.py) 以及[算子插件文档](OperatorPlugins_ZH.md)）。无需手工维护 requirements 文件——Data-Juicer 在首次使用时会自动安装缺失的包。

5. 全部完成！现在您可以在自己的配置文件中使用新添加的算子：

```yaml
# other configs
...

# process configs
process:
  - text_length_filter:  # add this op to your process list and set the parameters
      min_len: 10
      max_len: 1000
```

6. 社区贡献者可在alpha状态后就提相应算子PR。此后该贡献者可以与Data-Juicer团队一起在后续PR中，将其渐进完善到beta和stable版本。更多细节请参考下方第4节。我们非常欢迎共建，并会[高亮致谢](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)！

## 2. 构建你自己的数据菜谱和配置项

- 我们提供基于 [jsonargparse](https://github.com/omni-us/jsonargparse/) 的简单配置以降低样板代码的成本。
- 我们提供大量的示例性菜谱以供参阅复用和扩展，[数据菜谱Gallery](https://datajuicer.github.io/data-juicer-hub/zh_CN/main/docs/RecipeGallery_ZH.html)。
- 📣📣📣 社区贡献者可提PR在*数据菜谱Gallery*中添加自定义的数据菜谱，促进传播、复用和相关技术演进。更多细节请参考下方第4节。我们非常欢迎共建，并会[高亮致谢](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)！

### 2.1 丰富的配置源和类型提示

- 全局配置对象可以通过以下方式初始化

```python
# core.executor.py
self.cfg = init_configs()
```

- 其中可以指定和混合来自不同来源的函数参数，包括
    1. *硬编码默认值* 将配置注册到解析器中或在类的 `__init__` 函数中指定
    2. json 格式的默认*配置文件*（yaml 或 jsonnet 超集）
    3. *环境变量*
    4. *POSIX-style 命令行参数*， 例如 `--project_name my_data_demo` 或 `--project_name=my_data_demo`，包含配置文件

- 最终解析的值是来自这些来源的混合。 并且覆盖顺序与上面的数字相同。

此外，还支持许多参数类型和相应的验证。
包含 Python内置类型、来自 [Lib/typing](https://docs.python.org/3/library/typing.html) 的类型，以及来自 jsonargparse 的 [扩展类型](https://jsonargparse.readthedocs.io/en/stable/#type-hints)，例如具有自定义限制的 `restricted types` 和 `Paths`。

### 2.2 层次化的配置和帮助

- 您可以在参数名称中自由使用点符号来定义层次结构， 例如 `maximum_line_length_filter.min`.
更重要的是，默认情况下，我们自动注册已实现的运算符的 docstring。 也就是说，所有的结构配置始终与代码同步。
- 您可以通过运行脚本来获取层次化的帮助信息，例如：

```
$ python tools/process_data.py --help

usage: process_data.py [-h] [--config CONFIG] [--print_config[=flags]] [--project_name PROJECT_NAME] [--dataset_path DATASET_PATH] [--dataset_dir DATASET_DIR] [--export_path EXPORT_PATH] [--process PROCESS]
                            [--np NP] [--text_kes TEXT_KEYS] [--document_deduplicator CONFIG] [--document_deduplicator.hash_method HASH_METHOD] [--document_deduplicator.lowercase LOWERCASE]
                            [--document_deduplicator.ignore_non_character IGNORE_NON_CHARACTER] [--language_id_score_filter CONFIG] [--language_id_score_filter.lang LANG] [--words_num_filter CONFIG] [--words_num_filter.min MIN] [--words_num_filter.max MAX]
                            [--alphanumeric_filter CONFIG] [--alphanumeric_filter.min MIN] [--alphanumeric_filter.max MAX] [--average_line_length_filter CONFIG] [--average_line_length_filter.min MIN] [--average_line_length_filter.max MAX]
                            [--maximum_line_length_filter CONFIG] [--maximum_line_length_filter.min MIN] [--maximum_line_length_filter.max MAX] [--text_length_filter CONFIG] [--text_length_filter.min MIN] [--text_length_filter.max MAX]
                            [--remove_comments_mapper CONFIG] [--remove_comments_mapper.type TYPE] [--remove_comments_mapper.inline INLINE] [--remove_comments_mapper.multiline MULTILINE] [--remove_header_mapper CONFIG]
                            [--remove_header_mapper.before_section BEFORE_SECTION]

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.
  --project_name PROJECT_NAME
                        name of your data process project. (type: str, default: null)
  --dataset_path DATASET_PATH
                        path to your dataset file, relative with respect to the config file's location (type: Path_fr, default: null)
  --dataset_dir DATASET_DIR
                        path to your dataset(s) within a directory, relative with respect to the config file's location (type: Path_drw, default: null)
  --export_path EXPORT_PATH
                        path to the output processed dataset, relative with respect to the config file's location (type: Path_fc, default: null)
  --process PROCESS, --process+ PROCESS
                        a list of several process operators with their arguments (type: List[Dict], default: null)
  --np NP               number of subprocess to process your dataset. (type: PositiveInt, default: null)

<class 'data_juicer.ops.filter.alphanumeric_filter.AlphanumericFilter'>:
  --alphanumeric_filter CONFIG
                        Path to a configuration file.
  --alphanumeric_filter.min MIN
                        the min filter rate in alphanumeric op. (type: ClosedUnitInterval, default: 0.0)
  --alphanumeric_filter.max MAX
                        the max filter rate in alphanumeric op. (type: ClosedUnitInterval, default: 0.25)

<class 'data_juicer.ops.filter.text_length_filter.TextLengthFilter'>:
  --text_length_filter CONFIG
                        Path to a configuration file.
  --text_length_filter.min MIN
                        min text length in the filtering (type: int, default: 10)
  --text_length_filter.max MAX
                        max text length in the filtering (type: int, default: 10000)

......

```

## 3. 依赖管理

Data-Juicer 使用基于 `uv` 和 `pyproject.toml` 的现代依赖管理系统。依赖通过标准的 Python 打包格式 (PEP 621) 进行管理，并使用延迟加载系统按需安装。

### 3.1 安装 uv

`uv` 是一个快速的 Python 包安装器和解析器，用于替代 pip。您可以通过以下方式安装：

```bash
# 使用 curl 安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip 安装
pip install uv
```

安装完成后，您可以使用 `uv --version` 验证安装是否成功。

### 3.2 虚拟环境管理

`uv` 提供了虚拟环境管理功能，可以替代 `venv` 和 `virtualenv`。以下是常用命令：

```bash
# 创建新的虚拟环境
uv venv

# 创建指定 Python 版本的虚拟环境
uv venv --python 3.10

# 激活虚拟环境
# 在 Unix/macOS 上
source .venv/bin/activate
# 在 Windows 上
.venv\Scripts\activate

# 在虚拟环境中安装最小依赖
uv pip install -e .
```

### 3.3 添加新依赖

添加新依赖的方法：

1. 将依赖添加到 `pyproject.toml` 的相应部分：
   - 核心依赖放在 `[project.dependencies]` 中
   - 可选依赖放在 `[project.optional-dependencies]` 的相应组中（ generic、dev、audio、video，etc.）

2. 延迟加载系统会在首次使用时自动处理依赖安装。

示例：
```toml
[project.dependencies]
# 核心依赖
numpy = ">=1.26.4,<2.0.0"

[project.optional-dependencies]
generic = [
    "torch==2.6.0",
    "transformers>=4.47.0",
    ...
]
```

### 3.4 开发环境设置

1. 安装所有依赖：
```bash
uv pip install -e ".[all]"
```

2. 或安装特定组：
```bash
uv pip install -e ".[generic]"      # 通用依赖
uv pip install -e ".[dev]"          # 开发工具
uv pip install -e ".[ai_services]"  # 服务依赖
```

### 3.5 延迟加载

延迟加载系统在首次使用时自动安装依赖。这意味着：
- 初始安装更快
- 只安装必需的依赖
- 依赖按需安装
- 优先使用 `uv` 进行快速安装

## 4. 为开源社区贡献
### 4.1 编码规范

我们将编码规范定义在 `.pre-commit-config.yaml` 中。在向仓库贡献代码之前，请使用 `pre-commit` 工具对代码进行自动规范化。

```shell
# ===========install pre-commit tool===========
uv pip install pre-commit

cd <path_to_data_juicer>
# install pre-commit script for data_juicer
pre-commit install


# ===========check all files===========
git add .
pre-commit run --all-files

# commit after all checking are passed
git commit -m "<your_commit_message>"
```

**注意**：我们在github workflow配置了pre-commit的检查。如果您的PR中该检查没通过，请在本地①确保pre-commit 的相关依赖与项目配置一致（可通过`pre-commit clean`和`pre-commit install`完成）；②push前执行了`pre-commit run --all-files`.


### 4.2 文档规范

我们使用 Sphinx 进行文档管理。为保证开发文档顺利集成到 Sphinx 文档系统中，请在编写时注意以下规范：

1. 标题层级

    - 一级标题（`#`）：每个文档**必须且只能**包含一个一级标题，作为文档的整体标题。
    - 确保标题层级结构正确，不要跳级使用标题。例如，一级标题下应该是二级标题，而不是直接跳到三级标题。

2. 文件命名规范

    - 中文文档：中文 Markdown 文件的命名必须以 `_ZH` 结尾。例如：`README_ZH.md`


### 4.3 将你的新算子贡献到开源社区

- 根据实现完整性，算子会被分类为3类：
  - ![alpha](https://img.shields.io/badge/alpha-red?style=plastic) 版本：仅实现了最基本的算子能力
  - ![beta](https://img.shields.io/badge/beta-yellow?style=plastic) 版本：在 alpha 版本基础上为算子添加了单元测试，补充基础文档描述
  - ![stable](https://img.shields.io/badge/stable-green?style=plastic) 版本：在 beta 版本基础上进行了各项算子优化（如模型管理、批处理、算子融合等）

- 社区贡献者可以在 alpha 版本提交相应的算子 PR。之后，贡献者可以与 Data-Juicer 团队合作，在后续 PR 中逐步将其改进到 beta 版本和稳定版本。我们欢迎共同创作，并将重点标注[致谢](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)！

- 欢迎添加新算子的相应参考文献（例如，受现有想法或代码启发的新实现，或现有论文中提出的高级算法）。


#### 4.3.1 提供算子基本功能（alpha版本）
在前面[章节](#1-快速构建你自己的算子)中，我们实现的算子已经完整实现了基本功能，因此它已经满足![alpha](https://img.shields.io/badge/alpha-red?style=plastic)版本的要求。接下来，我们将介绍如何将这个算子进行扩展，使其更可用、更规范。

#### 4.3.2 使算子更可用（beta版本）

- （![beta](https://img.shields.io/badge/beta-yellow?style=plastic) 强烈推荐）为了增强代码鲁棒性、验证正确性和直观展示如何使用其功能，最好为新添加的算子进行单元测试。对于上面的 `TextLengthFilter` 算子，在 `tests/ops/filter/` 中实现如 `test_text_length_filter.py` 的测试文件：

```python
import unittest
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextLengthFilterTest(DataJuicerTestCaseBase):

    def test_func1(self):
        pass

    def test_func2(self):
        pass

    def test_func3(self):
        pass
        
if __name__ == '__main__':
    unittest.main()
```

- （![beta](https://img.shields.io/badge/beta-yellow?style=plastic) 强烈推荐）为了方便其他用户理解和使用，最好将新增的算子信息更新到相应的文档中，具体包括如下两个基本动作：
   1. 请在算子基类的doc string中补充基础信息，确保其完整可读（包括算子基本功能描述、入参、出参等）。无需用户麻烦地多处撰写，我们的`pre-commit`和sphinx构建脚本会自动抽取doc string形成算子池文档和API文档。
   2. `data_juicer/config/config_all.yaml`：该全集配置文件保存了所有算子及参数的一个列表，作为一些自动化特性的信息来源以及用户参考可用算子的一个重要文档之一。因此，在新增算子后，请将其也添加到该文档process列表里（按算子类型分组并按字母序排序）：
   
   ```yaml
   ...
   - stopwords_filter:                                       # filter text with stopword ratio smaller than a specific min value
       lang: en                                                # consider stopwords in what language
       tokenization: false                                     # whether to use model to tokenize documents
       min_ratio: 0.3                                          # the min ratio to filter text
       stopwords_dir: ./assets                                 # directory to store stopwords dictionaries
       use_words_aug: false                                    # whether to augment words, especially for Chinese and Vietnamese
       words_aug_group_sizes: [2]                              # the group size of words to augment
       words_aug_join_char: ""                                 # the join char between words to augment
   - text_length_filter:                                     # filter text with length out of specific range
       min_len: 10                                             # the min length of filter range
       max_len: 10000                                          # the max length of filter range
   - token_num_filter:                                       # filter text with total token number out of specific range
       hf_tokenizer: EleutherAI/pythia-6.9b-deduped            # name of used Hugging Face tokenizer
       min_num: 10                                             # the min number of filter range
       max_num: 10000                                          # the max number of filter range
   ...
   ```


#### 4.3.3 使算子更快更完备（stable版本）

- (![stable](https://img.shields.io/badge/stable-green?style=plastic)) 如果在算子中使用了 Hugging Face 模型，您可能希望利用 GPU 加速。为了实现这一点，请在算子的构造函数中声明 `_accelerator = 'cuda'`，并确保 `compute_stats_single/batched` 和 `process_single/batched` 方法接受一个额外的位置参数 `rank`。

    ```python
    # ... (same as above)

    @OPERATORS.register_module('text_length_filter')
    class TextLengthFilter(Filter):
   
        _accelerator = 'cuda'
   
        def __init__(self,
                    min_len: PositiveInt = 10,
                    max_len: PositiveInt = sys.maxsize,
                    *args,
                    **kwargs):
            # ... (same as above)

        def compute_stats_single(self, sample, rank=None):
            # ... (same as above)

        def process_single(self, sample, rank=None):
            # ... (same as above)
    ```

- (![stable](https://img.shields.io/badge/stable-green?style=plastic)) 如果算子批量处理数据，输入不是一个样本而是一个batch，或者你想在单样本实现上直接激活batch化处理，需要声明`_batched_op = True`。
      - 对于单样本实现中原来的 `compute_stats_single` 和 `process_single` 方法，你可以保持它们不变，Data-Juicer 会调用默认的batch化处理版本，它们会自动拆分单个样本以调用单样本版本的两个方法来支持batch化处理。你也可以自行实现更高效的batch化的版本。
    ```python
    # ... (import some other libraries)
    OP_NAME = 'image_diffusion_mapper'
    @OPERATORS.register_module(OP_NAME)
    @LOADED_IMAGES.register_module(OP_NAME)
    class ImageDiffusionMapper(Mapper):
        _batched_op = True

        def __init__(self,
                 # ... (OP parameters)
                 *args,
                 **kwargs):
            super().__init__(*args, **kwargs)

        def process_batched(self, samples):
            # ... (some codes)
    ```

- (![stable](https://img.shields.io/badge/stable-green?style=plastic)) 在mapper算子中，我们提供了产生额外数据的存储路径生成接口，避免出现进程冲突和数据覆盖的情况。生成的存储路径格式默认为`{ORIGINAL_DATAPATH}/__dj__produced_data__/{OP_NAME}/{ORIGINAL_FILENAME}__dj_hash_#{HASH_VALUE}#.{EXT}`，其中`HASH_VALUE`是算子初始化参数、每个样本中相关参数、进程ID和时间戳的哈希值。也可以指定保存路径（例如`save_dir`）或者通过环境变量（`DJ_PRODUCED_DATA_DIR`）设置额外数据的存储路径。为了方便，可以在OP类初始化开头调用`self.remove_extra_parameters(locals())`获取算子初始化参数，同时可以调用`self.add_parameters`添加每个样本与生成额外数据相关的参数。例如，利用diffusion模型对图像进行增强的算子：
    ```python
    # ... (import some library)
    OP_NAME = 'image_diffusion_mapper'
    @OPERATORS.register_module(OP_NAME)
    @LOADED_IMAGES.register_module(OP_NAME)
    class ImageDiffusionMapper(Mapper):
        def __init__(self,
                 # ... (OP parameters)
                 save_dir: str = None,
                 *args,
                 **kwargs):
            super().__init__(*args, **kwargs)
            self._init_parameters = self.remove_extra_parameters(locals())
            self.save_dir = save_dir

        def process_single(self, sample):
            # ... (some codes)
            # captions[index] is the prompt for diffusion model
            related_parameters = self.add_parameters(
                    self._init_parameters, caption=captions[index])
            new_image_path = transfer_filename(
                    origin_image_path, OP_NAME, self.save_dir, **related_parameters)
            # ... (some codes)
    ```
    针对一个数据源衍生出多个额外数据的情况，我们允许在生成的存储路径后面再加后缀。比如，根据关键帧将视频拆分成多个视频：
    ```python
    # ... (import some library)
    OP_NAME = 'video_split_by_key_frame_mapper'
    @OPERATORS.register_module(OP_NAME)
    @LOADED_VIDEOS.register_module(OP_NAME)
    class VideoSplitByKeyFrameMapper(Mapper):
        def __init__(self,
                 # ... (OP parameters)
                 save_dir: str = None,
                 *args,
                 **kwargs):
            super().__init__(*args, **kwargs)
            self._init_parameters = self.remove_extra_parameters(locals())
            self.save_dir = save_dir

        def process_single(self, sample):
            # ... (some codes)
            split_video_path = transfer_filename(
                        original_video_path, OP_NAME, self.save_dir, **self._init_parameters)
            split_video_path = add_suffix_to_filename(split_video_path, f'_{count}')
            # ... (some codes)
    ```


（![stable](https://img.shields.io/badge/stable-green?style=plastic) 可选）**使新算子可以进行算子融合**

- 如果我们的新算子中的部分中间变量的计算过程与已有的算子重复，那么可以将其添加到可融合算子中，以在数据处理时利用算子融合进行加速。（如`words_num_filter`与`word_repetition_filter`都需要对输入文本进行分词）
- 当算子融合（OP Fusion）功能开启时，这些重复的计算过程和中间变量是可以在算子之间的`context`中共享的，从而可以减少重复计算。
- 可通过如下步骤使包含共有中间变量的算子可进行算子融合（以`words_num_filter`算子为例）。

1. （可选）如果新算子中产生了新的中间变量，需要在`utils/constant.py`中的`InterVars`类中添加新的中间变量名称。通常需要在名称前加上`DEFAULT_PREFIX`前缀。

```python
class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'  # 在这里添加新的中间变量
    ...
```

2. （可选）第1步中添加的新的中间变量还需在`ops/op_fusion.py`中为其定义一个注册组，并添加到保存了所有注册组的列表中，方便算子融合模块追踪涉及到这些中间变量的算子。

```python
...
# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)  # 为新的中间变量定义注册组

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# all
ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_IMAGES]  # 并添加到注册组列表中
...
```

3. 在涉及到该中间变量的算子前，将该算子注册到中间变量对应的注册组中，表示该算子中可能对该中间变量进行了计算与使用。

```python
...
@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)  # 将该算子注册到注册组中
class WordsNumFilter(Filter):
...
```

4. 在算子计算该中间变量的过程中，可将计算逻辑修改为：
   1. 如果`context`参数为True，则表示已开启了算子融合，优先从`context`中获取前序算子已经计算过的该中间变量的值
   2. 如果中间变量在`context`中不存在，则表示在该算子中首次对该中间变量进行计算，在计算完成后，定义一个唯一的key并将其存放到`context`中，以供后续算子使用
   3. 如果`context`参数为False，则按照正常计算流程进行

```python
# 修改计算逻辑前
...
tokenizer = get_model(self.model_key)
words = get_words_from_document(
    sample[self.text_key],
    token_func=tokenizer.encode_as_pieces if tokenizer else None)
...        

# 修改计算逻辑后
...
words_key = f'{InterVars.words}-{self.model_key}'
if context and words_key in sample[Fields.context]:
    # 直接使用context中已有的中间变量值
    words = sample[Fields.context][words_key]
else:
    # 正常计算流程
    tokenizer = get_model(self.model_key)
    words = get_words_from_document(
        sample[self.text_key],
        token_func=tokenizer.encode_as_pieces if tokenizer else None)
    if context:
        # 第一次计算该中间变量后，放入context供后续算子使用
        sample[Fields.context][words_key] = words
...
```

5. 随着算子数量的增加，Data-Juicer的依赖也不断增多。为了防止Data-Juicer的依赖越来越重，我们为算子额外增加的依赖提供了一套延迟加载加上使用时安装依赖的策略。
    - OP的 `_requirements` 属性可用于指定算子所需的额外依赖。它可以是一个包列表或指向 requirements.txt 文件的字符串路径。在 Ray 模式下，此属性有助于 Data-Juicer 在 Ray 集群的计算节点上自动安装这些额外依赖。我们建议开发者为新算子显式设置此属性。
    - `LazyLoader`会检查加载的module对应的package有没有都安装，没有的话会动态自动安装。

```python
# ... (import some library)
from data_juicer.utils.lazy_loader import LazyLoader

# 懒导入
torch = LazyLoader('torch')
transformers = LazyLoader('transformers')
nltk = LazyLoader('nltk')

class PerplexityFilter(Filter):
    # 显式为 OP 设置额外的依赖
    _requirements = ['kenlm', 'sentencepiece', 'fasttext-predict']

    def __init__(self,
                # ... (OP parameters)
                *args,
                **kwargs):
        # 在初始化前自动安装所需依赖库
        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(['fasttext-predict'])
        # ... (some codes)

    def process_single(self, sample):
        # ... (some codes)
```

- 至此，该算子已经能够在算子融合功能开启后，自动地与其他算子进行融合并共享共有的中间变量，减少重复计算，加快整体的数据处理速度


### 4.4 贡献您的新配方
- 社区贡献者可以在[数据配方库](https://github.com/datajuicer/data-juicer-hub)中提交 PR，添加自定义数据配方，以促进传播、复用和相关技术演进。

- 欢迎添加您新配方的相应参考文献，或提出一些新需求、以及改进现有配方的想法。

- 我们非常欢迎共建，并将重点[注明致谢](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)！