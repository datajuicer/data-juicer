# 预处理工具

此文件夹包含一些预处理脚本，用于在使用 Data-Juicer 之前对数据集进行处理。

## 用法

### 按语言将数据集拆分为子数据集

该工具将根据语言信息将原始数据集拆分为不同的子数据集。

```shell
python tools/preprocess/dataset_split_by_language.py        \
    --src_dir             <src_dir>          \
    --target_dir          <target_dir>       \
    --suffixes            <suffixes>         \
    --text_key            <text_key>         \
    --num_proc            <num_proc>

# get help
python tools/preprocess/dataset_split_by_language.py --help
```

- `src_dir`: 将此参数设置为存储数据集的路径即可。
- `target_dir`: 用于存储转换后的 jsonl 文件的结果目录。
- `text_key`: 存储示例文本的字段的 key，默认为 text。
- `suffixes`: 待读取文件的后缀名，默认为 None。
- `num_proc` (可选): worker 进程数量，默认为 1。

### 将原始 arXiv 数据转换为 jsonl

该工具用于将从 S3 下载的原始 arXiv 数据转换为对 Data-Juicer 友好的 jsonl 格式。

```shell
python tools/preprocess/raw_arxiv_to_jsonl.py           \
    --arxiv_src_dir       <arxiv_src_dir>    \
    --target_dir          <target_dir>       \
    --temp_dir            <temp_dir>         \
    --num_proc            <num_proc>

# get help
python tools/preprocess/raw_arxiv_to_jsonl.py  --help
```

- `arxiv_src_dir`: 如果你像 Redpajama 一样下载原始 arXiv 数据，你将得到一个目录 src，其中包含数千个 tar 文件，其文件名类似于 `arXiv_src_yymm_xxx.tar`。 您只需将此参数设置为该目录的路径即可。
- `target_dir`: 用于存储转换后的 jsonl 文件的结果目录。
- `temp_dir`: 用于存储临时文件的目录，该目录将在转化结束时自动被删除，默认为 `./tmp`。
- `num_proc` (可选): worker 进程数量，默认为 1。

**注意事项：**

* 下载过程请参考[这里](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/arxiv)。

* 在下载、转换或处理之前，您需要确保您的硬盘空间足够大，可以存储原始数据（超过 3TB）、转换后的数据（超过 3TB）、最小处理后的数据（大约 500-600GB），以及处理期间的缓存数据。

### 将原始 stack_exchange 数据转换为 jsonl

使用 `raw_stackexchange_to_jsonl.py` 来转化原始 stack_exchange 数据.

该工具用于将从 [Archive](https://archive.org/download/stackexchange) 下载的原始 Stack Exchange 数据转化为多个 jsonl 文件.

```shell
python tools/preprocess/raw_arxiv_stackexchange_to_jsonl.py           \
    --src_dir       <src_dir>      \
    --target_dir    <target_dir>   \
    --topk          <topk>         \
    --num_proc      <num_proc>     \

# get help
python tools/preprocess/raw_stackexchange_to_jsonl.py  --help
```

- `src_dir`: 如果像 Redpajama 一样下载原始 Stack Exchange 数据，你将得到一个目录 src，其中包含数百个 7z 文件，其文件名类似于 `*.*.com.7z`。 您需要解压这些文件并将 POSTs.xml 重命名为相应的压缩包名称并将其放在该目录中。更多详情请参考[这里](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/stack_exchange)。
- `target_dir`: 用于存储转换后的 jsonl 文件的结果目录。
- `topk` (可选): 选择内容最多的 k 个站点，默认为 28.
- `num_proc` (可选): worker 进程数量，默认为 1。

**注意事项：** 在下载、转换或处理之前，您需要确保您的硬盘空间足够大，可以存储原始数据（超过 100GB）、转换后的数据（超过 100GB）

### 将原始 Alpaca-CoT 数据转换为 jsonl

使用 `raw_alpaca_cot_merge_add_meta.py` 来转化原始 Alpaca-CoT 数据.

该工具用于将从 [HuggingFace]( https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) 下载的原始 Alpaca-Cot 数据转化为 jsonl 文件.

```shell
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py           \
    --src_dir       <src_dir>      \
    --target_dir    <target_dir>   \
    --num_proc      <num_proc>     \

# get help
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py  --help
```

- `src_dir`: 将此参数设置为存储Alpaca-CoT数据集的路径。
- `target_dir`: 用于存储转换后的 jsonl 文件的结果目录。
- `num_proc` (可选): worker 进程数量，默认为 1。

### 重新格式化 csv 或者 tsv 文件

此工具用于将某些字段中可能具有 NaN 值的 csv 或 tsv 文件格式化为多个 jsonl 文件。

```shell
python tools/preprocess/reformat_csv_nan_value.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --suffixes          <suffixes>        \
    --is_tsv            <is_tsv>          \
    --keep_default_na   <keep_default_na> \
    --num_proc          <num_proc>

# get help
python tools/preprocess/reformat_csv_nan_value.py --help
```

- `src_dir`: 将此参数设置为存储数据集的路径，例如 `*.csv` 或 `*.tsv` 即可。
- `target_dir`: 用于存储转换后的 jsonl 文件的结果目录。
- `suffixes`: 待读取文件的后缀名，可指定多个，例如 `--suffixes '.tsv', '.csv'`
- `is_tsv`: 如果为 true，则分隔符将设置为 `\t`，否则默认设置为 `,`。
- `keep_default_na`: 如果为 False，字符串将被解析为 NaN，否则仅使用默认的 NaN 值进行解析。
- `num_proc` (可选): worker 进程数量，默认为 1。

### 重新格式化 jsonl 文件

该工具用于重新格式化某些字段中可能包含 Nan 值的 jsonl 文件。

```shell
python tools/preprocess/reformat_jsonl_nan_value.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --num_proc          <num_proc>

# get help
python tools/preprocess/reformat_jsonl_nan_value.py --help
```

- `src_dir`: 将此参数设置为存储数据集的路径，例如 `*.jsonl`。
- `target_dir`: 用于存储转换后的 jsonl 文件的结果目录。
- `num_proc` (可选): worker 进程数量，默认为 1。

### 序列化 jsonl 文件中的元字段

在一些jsonl文件中，不同的样本可能有不同的meta字段，甚至同一个meta字段中的数据类型也可能不同，这会导致使用 [HuggingFace Dataset](https://huggingface.co/docs/datasets/index) 读取数据集失败。 该工具用于将这些jsonl文件中除 `text_key` 之外的所有元字段序列化为字符串，以方便后续的Data-juicer处理。 数据集处理后，通常需要配合使用 [deserialize_meta.py](../postprocess/deserialize_meta.py) 对其进行反序列化。


```shell
python tools/preprocess/serialize_meta.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --text_key          <text_key>        \
    --serialized_key    <serialized_key>  \
    --num_proc          <num_proc>

# get help
python tools/preprocess/serialize_meta.py --help
```
- `src_dir`: 存储原始 jsonl 文件的路径。
- `target_dir`: 保存转换后的 jsonl 文件的路径。
- `text_key`: 不会被序列化的字段对应的 key, 默认为 “text”。
- `serialized_key`: 序列化后的信息保存的字段对应的 key, 默认为 “source_info”。
- `num_proc` (可选): worker 进程数量，默认为 1
