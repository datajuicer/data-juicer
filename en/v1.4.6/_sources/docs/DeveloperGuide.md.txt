# How-to Guide for Developers

- [How-to Guide for Developers](#how-to-guide-for-developers)
  - [1. Build Your Own OPs Quickly](#1-build-your-own-ops-quickly)
  - [2. Build Your Own Data Recipes and Configs](#2-build-your-own-data-recipes-and-configs)
    - [2.1 Fruitful Config Sources \& Type Hints](#21-fruitful-config-sources--type-hints)
    - [2.2 Hierarchical Configs and Helps](#22-hierarchical-configs-and-helps)
  - [3. Dependency Management](#3-dependency-management)
    - [3.1 Installing uv](#31-installing-uv)
    - [3.2 Virtual Environment Management](#32-virtual-environment-management)
    - [3.3 Adding New Dependencies](#33-adding-new-dependencies)
    - [3.4 Development Setup](#34-development-setup)
    - [3.5 Lazy Loading](#35-lazy-loading)
  - [4. Contribution to the Open-source Community](#4-contribution-to-the-open-source-community)
    - [4.1 Coding Style](#41-coding-style)
    - [4.2 Documentation Style](#42-documentation-style)
    - [4.3 Contribute Your New OPs to the Open-Source Community](#43-contribute-your-new-ops-to-the-open-source-community)
      - [4.3.1 Providing Basic OP Functions (alpha version)](#431-providing-basic-op-functions-alpha-version)
      - [4.3.2 Making the OP More Usable (beta version)](#432-making-the-op-more-usable-beta-version)
      - [4.3.3 Making OP Faster \& More complete (stable version)](#433-making-op-faster--more-complete-stable-version)
    - [4.4 Contribute Your New Recipes](#44-contribute-your-new-recipes)

## 1. Build Your Own OPs Quickly

- Data-Juicer allows everybody to easily build their own OPs.
- Before implementing a new OP, please refer to existing [OperatorsZoo](Operators.md) to avoid unnecessary duplication.

> The development process of the following example takes directly adding operators in the corresponding module of the source code as an example. If an operator is added externally, the new operator can be registered by passing the parameter `--custom-operator-paths` or configuring the `custom_operator_paths` parameter in the yaml file, for example: `custom_operator_paths: ['/path/to/new/op.py', '/path/to/new/ops/directory/]`.

Assuming we want to add a new Filter operator called "TextLengthFilter" to get corpus of expected text length, we can follow the following steps to build it.

1. (Optional) If the new OP defines  some statistical variables, please add the corresponding new `StatsKeysConstant` attribute in `data_juicer/utils/constant.py` for unified management.

```python
class StatsKeysConstant(object):
    ...              # other keys
    text_len = 'text_len'
```

2. Create a new OP file `text_length_filter.py` in the corresponding `data_juicer/ops/filter/` directory as follows.
   - It's a Filter OP, so the new OP needs to inherit from the basic `Filter` class in the `base_op.py`, and be decorated with `@OPERATORS.register_module(xx_op)` to register itself automatically.
   - For convenience, we can implement the core functions `compute_stats_single` and `process_single` in a single-sample way, whose input and output are a single sample dictionary. 
   - [Advanced] If you are familiar with batched processing in Data-Juicer, you can also implement the batched version directly by overwriting the `compute_stats_batched` and `process_batched` functions, which will be slightly faster than single-sample version. Their input and output are a column-wise dict with multiple samples (detailed in the following Section 4.3.3).

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

3. After implementation, add it to the OP dictionary in the `__init__.py` file in `data_juicer/ops/filter/` directory.

```python
# other OPs
from .text_length_filter import TextLengthFilter  # import this new OP class
__all__ = [
    # other Ops
    "TextLengthFilter",  # add this new Op to __all__
]
```

4. When an operator has package dependencies listed in `environments/science_requires.txt`, you need to add the corresponding dependency packages to the `OPS_TO_PKG` dictionary in `data_juicer/utils/auto_install_mapping.py` to support dependency installation at the operator level.

5. Now you can use this new OP with custom arguments in your own config files!

```yaml
# other configs
...

# process configs
process:
  - text_length_filter:  # add this OP to your process list and set the parameters
      min_len: 10
      max_len: 1000
```

6. Community contributors can submit corresponding operator PRs and work with the Data-Juicer team to gradually improve it in subsequent PRs. Please see more details [below](#4-contribution-to-the-open-source-community). We greatly welcome co-construction and will [highlight acknowledgements](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)!


## 2. Build Your Own Data Recipes and Configs
- We provide easy configuration based on [jsonargparse](https://github.com/omni-us/jsonargparse/) to reduce cost for boilerplate codes.
- We provide fruitful examples in [Data Recipe Gallery](https://datajuicer.github.io/data-juicer-hub/en/main/docs/RecipeGallery.html) for reference reuse and extension.
- 📣📣📣 Community contributors can submit PRs in the *Data Recipe Gallery* to add customized data recipes to promote dissemination, reuse and related technical evolution. We greatly welcome co-construction and will highlight [acknowledgements](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)!

### 2.1 Fruitful Config Sources & Type Hints
- A global config object can be initialized via
```
# core.executor.py
self.cfg = init_configs()
```
- in which function arguments from diverse sources can be specified and mixed
up, including
1. *hard-coded default values* when registering the config into parser or specified in the classes' `__init__` functions
2. default *config files* in json (yaml or jsonnet supersets)
3. *environment variables*
4. *POSIX-style command line arguments*, such as ``--project_name
   my_data_demo`` or ``--project_name=my_data_demo`` , including config files

- The final parsed values are mixed from these sources. And the override order is the same as the numbers above.

Besides, many argument types and respective validation are supported.
Including python built-in types, types from [Lib/typing](https://docs.python.org/3/library/typing.html) module, and
extended [types](https://jsonargparse.readthedocs.io/en/stable/#type-hints)
from jsonargparse, such as `restricted types` and `Paths` with customized
limitations.

### 2.2 Hierarchical Configs and Helps
- You can use dot notation in the argument names freely to define the
hierarchy, e.g., `maximum_line_length_filter.min`.
More importantly, by default, we automatically register the configs from
the docstrings of implemented operators. That is, the structure of all
configs are always in sync with codes.

- You can get the hierarchical help information by running a script that calls
our executor such as
```
$ python tools/process_data.py --help

usage: process_data.py [-h] [--config CONFIG] [--print_config[=flags]] [--project_name PROJECT_NAME] [--dataset_path DATASET_PATH] [--dataset_dir DATASET_DIR] [--export_path EXPORT_PATH] [--process PROCESS]
                            [--np NP] [--text_keys TEXT_KEYS] [--document_deduplicator CONFIG] [--document_deduplicator.hash_method HASH_METHOD] [--document_deduplicator.lowercase LOWERCASE]
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

## 3. Dependency Management

Data-Juicer uses a modern dependency management system based on `uv` and `pyproject.toml`. Dependencies are managed through the standard Python packaging format (PEP 621) and installed on-demand using a lazy loading system.

### 3.1 Installing uv

`uv` is a fast Python package installer and resolver, as a better drop-in replacement for pip. You can install it using:

```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

After installation, verify it's working by running `uv --version`.

### 3.2 Virtual Environment Management

`uv` provides virtual environment management capabilities that replaces `venv` and `virtualenv`. Here are the common commands:

```bash
# Create a new virtual environment
uv venv

# Or create a virtual environment with a specific Python version
uv venv --python 3.10

# Activate the virtual environment
# On Unix/macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

# Install minimal dependencies in the virtual environment
uv pip install -e .

```

### 3.3 Adding New Dependencies

To add new dependencies:

1. Add them to the appropriate section in `pyproject.toml`:
   - Core dependencies go in `[project.dependencies]`
   - Optional dependencies go in `[project.optional-dependencies]` under the appropriate group (generic, dev, audio, video, etc.)

2. The lazy loading system will automatically handle installation when the dependencies are first used.

Example:
```toml
[project.dependencies]
# Core dependencies
numpy = ">=1.26.4,<2.0.0"

[project.optional-dependencies]
generic = [
    "torch==2.6.0",
    "transformers>=4.47.0",
    ...
]
```

### 3.4 Development Setup

1. Install the package with all dependencies:
```bash
uv pip install -e ".[all]"
```

2. Or install with specific groups:
```bash
uv pip install -e ".[generic]"      # Generic dependencies
uv pip install -e ".[dev]"          # Development tools
uv pip install -e ".[ai_services]"  # Services dependencies
```

### 3.5 Lazy Loading

The lazy loading system automatically installs dependencies when they are first used. This means:
- Initial installation is faster
- Only required dependencies are installed
- Dependencies are installed on-demand
- Uses `uv` for fast installation when available

## 4. Contribution to the Open-source Community


### 4.1 Coding Style

We define our styles in `.pre-commit-config.yaml`. Before committing,
please install `pre-commit` tool to automatically check and modify accordingly:

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
git commit -m "xxxx"
```

**Note**: We have configured pre-commit checks in github workflow. If this 
check in your PR fails, please locally ① ensure that the relevant 
dependencies of pre-commit are consistent with the project configuration 
(which can be completed through `pre-commit clean` and `pre-commit install`); 
and ② execute `pre-commit run --all-files` before push.

### 4.2 Documentation Style

We use Sphinx for document management. To ensure the smooth integration of development documents into the Sphinx documentation system, please pay attention to the following guidelines when writing:

1.  Heading Hierarchy

    - Level 1 Heading (`#`): Each document **must and can only** contain one Level 1 heading, which serves as the overall title of the document.
    - Ensure the heading hierarchy is correct and avoid skipping heading levels. For example, a Level 1 heading should be followed by a Level 2 heading, not a Level 3 heading.

2.  File Naming Conventions

    - Chinese Documents: Chinese Markdown files must be named with the suffix `_ZH`. For example: `README_ZH.md`


### 4.3 Contribute Your New OPs to the Open-Source Community

- To faciliate community contribution, according to the implementation progress, OP will be categorized into 3 types of versions:
  - ![alpha](https://img.shields.io/badge/alpha-red?style=plastic) version: Only the basic OP implementations are finished.
  - ![beta](https://img.shields.io/badge/beta-yellow?style=plastic) version: Based on the alpha version, unittests for this OP and basic docstring are added as well.
  - ![stable](https://img.shields.io/badge/stable-green?style=plastic) version: Based on the beta version, OP optimizations (e.g. model management, batched processing, OP fusion, ...)

- Community contributors can submit corresponding operator PRs in the alpha state. After that, the contributor can work with the Data-Juicer team to gradually improve it to beta and stable versions in subsequent PRs. We welcome co-construction and will highlight [acknowledgements](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)!

- Welcome to add the corresponding references of your new recipe (e.g., a new implementation inspired by some existing ideas or codes, or an advanced algorithm proposed in an existing paper).

#### 4.3.1 Providing Basic OP Functions (alpha version)

In the previous [section](#1-build-your-own-ops-quickly), the operator we implemented has already fulfilled the basic functionalities, thus meeting the requirements for the ![alpha](https://img.shields.io/badge/alpha-red?style=plastic) version. Next, we will introduce how to extend this operator to make it more usable and standardized.


#### 4.3.2 Making the OP More Usable (beta version)

- (![beta](https://img.shields.io/badge/beta-yellow?style=plastic) strongly recommended) In order to enhance the robustness of the code, verify the correctness and intuitively show how to use its functions, it is best to unit test the newly added operators. For the `TextLengthFilter` operator above, implement a test file such as `test_text_length_filter.py` in `tests/ops/filter/`:

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

- (![beta](https://img.shields.io/badge/beta-yellow?style=plastic) strongly recommend) In order to facilitate other users to understand and use, it is best to update the newly added operator information to the corresponding documents, including the following two basic actions:
   1. Please add basic information to the doc string of the operator class to ensure that it is complete and readable (including basic function description of the operator, input parameters, output parameters, etc.). There is no need for users to write in multiple places. Our `pre-commit` and sphinx build scripts will automatically extract doc strings to form operator pool documents and API documents.
   2. `data_juicer/config/config_all.yaml`: This complete configuration file saves a list of all operators and parameters, as a source of information for some automated features and one of the important documents for users to refer to available operators. Therefore, after adding a new operator, please also add it to the document process list (grouped by operator type and sorted alphabetically):
   
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


#### 4.3.3 Making OP Faster & More complete (stable version)


- (![stable](https://img.shields.io/badge/stable-green?style=plastic)) If Hugging Face models are used within an operator, you might want to leverage GPU acceleration. To achieve this, declare `_accelerator = 'cuda'` in the OP's constructor, and ensure that `compute_stats_single/batched` and `process_single/batched` methods accept an additional positional argument `rank`.

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

- (![stable](https://img.shields.io/badge/stable-green?style=plastic)) If the operator processes data in batches rather than a single sample, or you want to enable batched processing, it is necessary to declare `_batched_op = True`.
      - For the original `compute_stats_single` and `process_single` functions, you can keep it still and Data-Juicer will call the default batched version to call the single version to support batched processing. Or you can implement your batched version in a more efficient way.
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

- (![stable](https://img.shields.io/badge/stable-green?style=plastic)) In a mapper operator, to avoid process conflicts and data coverage, we offer an interface to make a saving path for produced extra data. The format of the saving path is `{ORIGINAL_DATAPATH}/__dj__produced_data__/{OP_NAME}/{ORIGINAL_FILENAME}__dj_hash_#{HASH_VALUE}#.{EXT}`, where the `HASH_VALUE` is hashed from the init parameters of the operator, the related parameters in each sample, the process ID, and the timestamp. You can also specify the save path (for example, save_dir) or set the storage path through the environment variable `DJ_PRODUCED_DATA_DIR`. For convenience, we can call `self.remove_extra_parameters(locals())` at the beginning of the initiation to get the init parameters. At the same time, we can call `self.add_parameters` to add related parameters with the produced extra data from each sample. Take the operator which enhances the images with diffusion models as example:
    ```python
    from data_juicer.utils.file_utils import transfer_filename
    # ... (import some other libraries)
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
    For the mapper to produce multi extra data base on one origin data, we can add suffix at the saving path. Take the operator which splits videos according to their key frames as example:
    ```python
    from data_juicer.utils.file_utils import add_suffix_to_filename, transfer_filename
    # ... (import some other libraries)
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
            split_video_path = add_suffix_to_filename(split_video_path,  f'_{count}')
            # ... (some codes)
    ```

(![stable](https://img.shields.io/badge/stable-green?style=plastic) Optional) **Make your OP fusible**

- If the calculation process of some intermediate variables in the new OP is reused in other existing OPs, this new OP can be
added to the fusible OPs to accelerate the whole data processing with OP fusion technology. (e.g. both the `words_num_filter`
and `word_repetition_filter` need to split the input text into words)
- When opening OP fusion, these reused calculation processes and intermediate variables can be shared in the `context` between
OPs, thus reducing repeated calculations.
- OPs that contain common intermediate variables can be fused in OP fusion through the following steps:

1. (Optional) If a new intermediate variable is generated in the new OP, we need to add this new intermediate variable name to 
the `InterVars` class in `utils/constant.py`. In general, we need to add a prefix `DEFAULT_PREFIX` before the name.

```python
class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'  # add the new intermediate variable here
    ...
```

2. (Optional) We need to define a registry group in `ops/op_fusion.py` for the new intermediate variable in the 1st step, and add
this registry group to the registry group list that stores all groups of intermediate variables. This facilitates the OP Fusion module
to track OPs involving these intermediate variables.

```python
...
# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)  # define registry group for the new intermediate variable

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# all
ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_IMAGES]  # and add it to the registry group list
...
```

3. Before the OP class definition that involves the intermediate variable, register this OP in the registry group corresponding
to this intermediate variable, indicating that the intermediate variable may be calculated and used in this OP.

```python
...
@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)  # register this new OP into the registry group
class WordsNumFilter(Filter):
...
```

4. In the calculation process of this intermediate variable of the new OP, we can modify the calculation logic to:
   1. If the argument `context` is True, it means the OP fusion is opening, so we get the value of this intermediate variable 
   from `context` first, which has been calculated by the previous OPs.
   2. If this intermediate variable doesn't exist in the `context`, it means it's the first time to calculate this variable in this
   OP, so we need to define a unique key and use it to store the intermediate variable in the `context` for subsequent OPs after
   it's calculated by this new OP.
   3. If the argument `context` is False, just follow the normal calculation process.

```python
# before modification
...
tokenizer = get_model(self.model_key)
words = get_words_from_document(
    sample[self.text_key],
    token_func=tokenizer.encode_as_pieces if tokenizer else None)
...        

# after modification
...
words_key = f'{InterVars.words}-{self.model_key}'
if context and words_key in sample[Fields.context]:
    # get the value of intermediate variable from context directly
    words = sample[Fields.context][words_key]
else:
    # normal calculation process
    tokenizer = get_model(self.model_key)
    words = get_words_from_document(
        sample[self.text_key],
        token_func=tokenizer.encode_as_pieces if tokenizer else None)
    if context:
        # After calculating the intermediate variable for the first time,
        # store it in the context for subsequent OPs.
        sample[Fields.context][words_key] = words
...
```

5. As the number of OPs increases, Data-Juicer's dependencies also multiply. To prevent Data-Juicer from becoming excessively burdened with dependencies, we've implemented a strategy that incorporates lazy importing and on-demand installation of additional dependencies required by OPs. `LazyLoader` will check if the packages corresponding to the module being loaded are installed, and if not, it will dynamically install them automatically. Below is an example illustrating this approach:

```python
# ... (import some library)
from data_juicer.utils.lazy_loader import LazyLoader

# lazy import
kenlm = LazyLoader('kenlm')
sp = LazyLoader('sentencepiece')

class PerplexityFilter(Filter):
    def __init__(self,
                # ... (OP parameters)
                *args,
                **kwargs):
        # auto install before init
        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(['fasttext-wheel'])
        # ... (some codes)

    def process_single(self, sample):
        # ... (some codes)
```


### 4.4 Contribute Your New Recipes 
- Community contributors can submit PRs in the [Data Recipe Gallery](https://github.com/datajuicer/data-juicer-hub) to add customized data recipes to promote dissemination, reuse and related technical evolution. 

- Feel free to add the corresponding references of your new recipe, or just propose some requirements/ideas to improve the existing recipes.

- We greatly welcome co-construction and will highlight [acknowledgements](https://github.com/datajuicer/data-juicer?tab=readme-ov-file#contribution-and-acknowledgements)!
