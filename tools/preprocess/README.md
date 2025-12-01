# Preprocess Tools

This folder contains some preprocess scripts for additional processing of your dataset before using Data-Juicer.

## Usage

### Split datasets to sub-datasets by language

This tool will split raw dataset to different sub-datasets by language information.


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
- `src_dir`: you just need to set this argument to the path which stores your datasets.
- `target_dir`: result directory to store the converted jsonl files.
- `text_key`: key name of field that stores sample text. Default: text
- `suffixes`: the suffix of files that will be read. Default: None
- `num_proc` (optional): number of process workers. Default it's 1.

### Convert raw arXiv data to jsonl

This tool is used to convert the raw arXiv data downloaded from S3 into the jsonl format which is friendly to Data-Juicer.



```shell
python tools/preprocess/raw_arxiv_to_jsonl.py           \
    --arxiv_src_dir       <arxiv_src_dir>    \
    --target_dir          <target_dir>       \
    --temp_dir            <temp_dir>         \
    --num_proc            <num_proc>

# get help
python tools/preprocess/raw_arxiv_to_jsonl.py  --help
```
- `arxiv_src_dir`: if you download raw arXiv data as Redpajama did, you will get a directory src which includes thousands of tar files whose filenames are like `arXiv_src_yymm_xxx.tar`. You just need to set this argument to the path of this dir.
- `target_dir`: result directory to store the converted jsonl files.
- `temp_dir`: directory to store intermediate files, and they will be removed once the conversion ends. Default it's `./tmp`
- `num_proc` (optional): number of process workers. Default it's 1.

**Note:**

* For downloading process, please refer to [here](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/arxiv).

* Before you downloading, converting or processing, you might make sure that your drive space is large enough to store the raw data (over 3TB), converted data (over 3TB), at least processed data (about 500-600GB), and even more cache data during processing.

### Convert raw stack_exchange data to jsonl

Use `raw_stackexchange_to_jsonl.py` to convert raw stack_exchange data.

This tool is used for converting the raw Stack Exchange data downloaded from from [Archive](https://archive.org/download/stackexchange) to several jsonl files.



```shell
python tools/preprocess/raw_arxiv_stackexchange_to_jsonl.py           \
    --src_dir       <src_dir>      \
    --target_dir    <target_dir>   \
    --topk          <topk>         \
    --num_proc      <num_proc>     \

# get help
python tools/preprocess/raw_stackexchange_to_jsonl.py  --help
```
- `src_dir`: if you download raw Stack Exchange data as Redpajama did, you will get a directory src which includes hundreds of 7z files whose filenames are like `*.*.com.7z `. You need to unzip these files and rename the POSTs.xml to the corresponding compressed package name and place it in that dir. For more details, please refer to [here](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/stack_exchange).
- `target_dir`: result directory to store the converted jsonl files.
- `topk` (optional): select the topk sites with the most content. Default it's 28.
- `num_proc` (optional): number of process workers. Default it's 1.

**Note:** Before you downloading, converting or processing, you might make sure that your drive space is large enough to store the raw data (over 100GB), converted data (over 100GB)

### Convert raw Alpaca-CoT data to jsonl

Use `raw_alpaca_cot_merge_add_meta.py` to convert raw Alpaca-CoT data.

This tool is used for converting the raw Alpaca-Cot data downloaded from [HuggingFace](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) to jsonl files.



```shell
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py       \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --num_proc          <num_proc>

# get help
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py --help
```
- `src_dir`: you just need to set this argument to the path which stores Alpaca_CoT data.
- `target_dir`: result directory to store the converted jsonl files.
- `num_proc` (optional): number of process workers. Default it's 1.

### Reformat csv or tsv file

This tool is used to reformat csv or tsv files which may have Nan values in some field to several jsonl files.



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
- `src_dir`: you just need to set this argument to the path which stores filenames are like `*.csv` or `*.tsv`.
- `target_dir`: result directory to store the converted jsonl files.
- `suffixes`: what kind of suffixes you want to process, multi-suffixes args like  `--suffixes '.tsv', '.csv'`.
- `is_tsv`: if true, sep will be set to `'\t'`, otherwise `','` as default.
- `keep_default_na`: if False, strings will be parsed as NaN, otherwise only the default NaN values are used for parsing.
- `num_proc` (optional): number of process workers. Default it's 1.

### Reformat jsonl file

This tool is used to reformat jsonl files which may have Nan values in some field.



```shell
python tools/preprocess/reformat_jsonl_nan_value.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --num_proc          <num_proc>

# get help
python tools/preprocess/reformat_jsonl_nan_value.py --help
```
- `src_dir`: you just need to set this argument to the path which stores filenames are like `*.jsonl`.
- `target_dir`: result directory to store the converted jsonl files.
- `num_proc` (optional): number of process workers. Default it's 1.


### Serialize meta fields in jsonl file

In some jsonl files, different samples may have different meta fields, and even the data types in the same meta field may be different, which will cause failure to read the dataset using [HuggingFace Dataset](https://huggingface.co/docs/datasets/index). This tool is used to serialize all meta fields except `text_key` in these jsonl files into strings to facilitate subsequent Data-juicer processing. After the dataset is processed, it usually needs to be deserialized using [deserialize_meta.py](../postprocess/deserialize_meta.py).



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
- `src_dir`: path to store jsonl files.
- `target_dir`: path to save the converted jsonl files.
- `text_key`: the key corresponding to the field that will not be serialized. Default it's 'text'.
- `serialized_key`: the key corresponding to the field that the serialized info saved. Default it's 'source_info'.
- `num_proc` (optional): number of process workers. Default it's 1.
