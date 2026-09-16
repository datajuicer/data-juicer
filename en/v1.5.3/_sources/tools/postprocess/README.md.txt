# Postprocess tools

This folder contains some postprocess scripts for additional processing of your processed dataset using Data-Juicer.

## Usage

### Count tokens for datasets

Use [count_token.py](count_token.py) to count tokens for datasets.

```shell
python tools/postprocess/count_token.py        \
    --data_path            <data_path>         \
    --text_keys            <text_keys>         \
    --tokenizer_method     <tokenizer_method>  \
    --num_proc             <num_proc>

# get help
python tools/postprocess/count_token.py --help
```

- `data_path`: path to the input dataset. Only support `jsonl` now.
- `text_keys`: field keys that will be considered into token counts.
- `tokenizer_method`: name of the Hugging Face tokenizer.
- `num_proc`: number of processes to count tokens.

### Mix multiple datasets with optional weights

Use [data_mixture.py](data_mixture.py) to mix multiple datasets.

This script will randomly select samples from every dataset and mix these samples and export to a new_dataset.


```shell
python tools/postprocess/data_mixture.py        \
    --data_path             <data_path>         \
    --export_path           <export_path>       \
    --export_shard_size     <export_shard_size> \
    --num_proc              <num_proc>

# get help
python tools/postprocess/data_mixture.py  --help
```

- `data_path`: a dataset file or a list of dataset files or a list of both them, optional weights, if not set, 1.0 as default.
- `export_path`: a dataset file name for exporting mixed dataset, support `json` / `jsonl` / `parquet`.
- `export_shard_size`:  dataset file size in Byte. If not set, mixed dataset will be exported into only one file.
- `num_proc`:  process num to load and export datasets.

- e.g., `python tools/postprocess/data_mixture.py  --data_path  <w1> ds.jsonl <w2> ds_dir <w3> ds_file.json`

**Note:** All datasets must have the same meta field, so we can use [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) to align their features.

### Deserialize meta fields in jsonl file

This tool is usually used with [serialize_meta.py](../preprocess/serialize_meta.py) to deserialize the specified field into the original format.


```shell
python tools/postprocess/deserialize_meta.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --serialized_key    <serialized_key>  \
    --num_proc          <num_proc>

# get help
python tools/postprocess/deserialize_meta.py --help
```
- `src_dir`: path to store jsonl files.
- `target_dir`: path to save the converted jsonl files.
- `serialized_key`: the key corresponding to the field that will be deserialized. Default it's 'source_info'.
- `num_proc` (optional): number of process workers. Default it's 1.

**Note:** After deserialization, all serialized fields in the original file will be placed in `'serialized_key'`, this is to ensure that the fields generated after data-juicer processing will not conflict with the original meta fields.
