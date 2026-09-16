# Postprocess tools

此文件夹包含一些后处理脚本，用于对 Data-Juicer 处理后的数据集进行进一步处理。

## 用法

### 为数据集计算token数目

使用 [count_token.py](count_token.py) 计算数据集包含的 token 数目。

```shell
python tools/postprocess/count_token.py        \
    --data_path            <data_path>         \
    --text_keys            <text_keys>         \
    --tokenizer_method     <tokenizer_method>  \
    --num_proc             <num_proc>

# get help
python tools/postprocess/count_token.py --help
```

- `data_path`: 输入数据集的路径。目前只支持 `jsonl` 格式。
- `text_keys`: 单个样本中会被算入 token 数目的字段名称。
- `tokenizer_method`: 使用的 Hugging Face tokenizer 的名称。
- `num_proc`: 计算 token 数目时所用的进程数。

### 将多个数据集以可选的权重混合

使用 [data_mixture.py](data_mixture.py) 将多个数据集混合。

该脚本将从每个数据集中随机选择样本并混合这些样本并导出到新的数据集。

```shell
python tools/postprocess/data_mixture.py        \
    --data_path             <data_path>         \
    --export_path           <export_path>       \
    --export_shard_size     <export_shard_size> \
    --num_proc              <num_proc>

# get help
python tools/postprocess/data_mixture.py  --help
```

- `data_path`: 数据集文件或数据集文件列表或两者的列表。可附加可选权重，权重未设置时默认值为 1.0。
- `export_path`: 用于导出混合数据集的数据集文件名，支持 `json` / `jsonl` / `parquet` 格式。
- `export_shard_size`: 数据集文件大小（以字节为单位）。 如果未设置，混合数据集将仅导出到一个文件中。
- `num_proc`:  加载以及导出数据集使用的进程数量

- 例，`python tools/postprocess/data_mixture.py  --data_path  <w1> ds.jsonl <w2> ds_dir <w3> ds_file.json`

**注意事项:** 所有数据集必须具有相同的元字段，从而可以使用 [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) 来对齐它们的特征。

### 反序列化 jsonl 文件中的元字段

该工具通常和 [serialize_meta.py](../preprocess/serialize_meta.py) 配合使用，将指定字段反序列化为原始格式。

```shell
python tools/postprocess/deserialize_meta.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --serialized_key    <serialized_key>  \
    --num_proc          <num_proc>

# get help
python tools/postprocess/deserialize_meta.py --help
```
- `src_dir`: 存储原始 jsonl 文件的路径。
- `target_dir`: 保存转换后的 jsonl 文件的路径。
- `serialized_key`: 将被反序列化的字段对应的 key, 默认为“source_info”。.
- `num_proc` (optional): worker 进程数量，默认为 1

**注意事项:** 经过反序列化后原始文件中所有被序列化的字段都会放在`‘serialized_key’`中，这样做是为了保证 data-juicer 处理后生成的字段不会和原有的元字段冲突。
