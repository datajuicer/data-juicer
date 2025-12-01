# Redpajama 配置文件

Data-Juicer-Hub 中的[文件夹](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama)包含的配置文件用于轻松复现 [Redpajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep) 的处理流程。

在开始前，请将 [data-juicer-hub](https://github.com/datajuicer/data-juicer-hub) 仓库克隆到本地，其已经包含了大量的数据处理菜谱。
```shell
git clone https://github.com/datajuicer/data-juicer-hub.git
```

## arXiv

原始数据文件从 [Redpajama/arXiv](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/arxiv) 中相同的 AWS 链接下载。

下载完成后，使用 [raw_arxiv_to_jsonl.py](../../tools/preprocess/raw_arxiv_to_jsonl.py) 将原始格式转换为 Data-Juicer 易于处理的格式：

```shell
python tools/preprocess/raw_arxiv_to_jsonl.py           \
    --arxiv_src_dir       <arxiv_src_dir>    \
    --target_dir          <target_dir>       \
    --temp_dir            <temp_dir>         \
    --num_proc            <num_proc>
```

预处理完成后，修改 [redpajama-arxiv.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-arxiv.yaml) 中的数据路径，执行以下命令复现 RedPajama 的处理流程：

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-arxiv.yaml
```

### 指标对比

| | 样本数 | 令牌数 | 峰值内存 | 运行时间 |
| --- | :---: | :---: | :---: | --- |
| redpajama | 1,724,497 | 30,667,506,934 | 35GB |`total: 11h52min` |
| Data-Juicer | 2,675,426| 30,338,153,178 | 21GB | preprocess: 5h21min<br>read+unify: 25min<br>remove_header_mapper: 5min<br>remove_comments_mapper: 3min<br> remove_bibliography_mapper: 4min<br>expand_macro_mapper: 5min19s<br>text_length_filter: 4min<br>export: 43min<br>`total: 6h53min` |

## Books

原始数据文件从 [Redpajama/Books](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/book) 中相同的 HuggingFace 链接下载。

下载完成后，修改 [redpajama-books.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-books.yaml) 中的数据路径，执行以下命令复现 RedPajama 的处理流程：

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-books.yaml
```

### 指标对比

| | 样本数 | 令牌数 | 峰值内存 | 运行时间 |
| --- | :---: | :---: | :---: | --- |
| redpajama | 205,183 | 25,962,395,123 | 450GB | split_for_dedup: 5min<br>dedup: 117min<br> `total: 122min` |
| Data-Juicer | 207,902 | 26,108,635,683 | 96GB | read+unify: 20min<br>compute_hash: 78min<br>dedup: 3min<br>export: 3min<br>`total: 114min` |

## Code

原始数据文件从 [Redpajama/Code](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/github) 中相同的 Google BigQuery 获取。

下载完成后，解压缩并删除扩展名不在以下白名单中的其他文件：

```text
.asm, .bat, .cmd, .c, .h, .cs, .cpp, .hpp, .c++, .h++, .cc, .hh, .C, .H, .cmake, .css, .dockerfile, .f90, .f, .f03, .f08, .f77, .f95, .for, .fpp, .go, .hs, .html, .java, .js, .jl, .lua, .md, .markdown, .php, .php3, .php4, .php5, .phps, .phpt, .pl, .pm, .pod, .perl,  ps1, .psd1, .psm1, .py, .rb, .rs, .sql, .scala, .sh, .bash, .command, .zsh, .ts, .tsx, .tex, .vb, Dockerfile, Makefile, .xml, .rst, .m, .smali
```

修改 [redpajama-code.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-code.yaml) 中的数据路径，执行以下命令复现 redpajama 的处理流程：

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-code.yaml
```

### 指标对比

| | 样本数 | 令牌数 | 峰值内存 | 运行时间 |
| --- | :---: | :---: | :---: | --- |
| redpajama | 73,208,524 | 150,390,270,060| 212GB | local-dedup: 37h<br>global-dedup: 1h<br>merge-dedup: 6h<br>filter: 17h<br>`total: 61h` |
| Data-Juicer | 73,169,889| 150,310,903,230| 370GB | preprocess: 5h21min<br>read+unify: 12h<br>document_deduplicator: 20h<br>clean_copyright_mappe:  3h<br>maximum_line_length_filter: 2.5h<br>average_line_length_filter: 2h<br>alphanumeric_filter: 13h<br>export: 2.5h<br>`total: 59h` |

## StackExchange

原始数据文件从 [Redpajama/Stack_exchange](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/stack_exchange) 中相同的 Archive 链接获取。

下载完成后，使用 [raw_stackexchange_to_jsonl.py](../../tools/preprocess/raw_stackexchange_to_jsonl.py) 将原始格式转换为 Data-Juicer 易于处理的格式：

```shell
python tools/preprocess/raw_arxiv_stackexchange_to_jsonl.py           \
    --src_dir       <src_dir>      \
    --target_dir    <target_dir>   \
    --topk          <topk>         \
    --num_proc      <num_proc>     \
```

预处理完成后，修改 [redpajama-stackexchange.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-stackexchange.yaml) 中的数据路径，执行以下命令复现 redpajama 的处理流程：

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-stackexchange.yaml
```

### 指标对比

| | 样本数 | 令牌数 | 峰值内存 | 运行时间 |
| --- | :---: | :---: | :---: | --- |
| redpajama | 29,825,086 | 20,502,757,123 | >500GB | filter: 170min<br>postprocess: 90min<br>`total: 260min` |
| Data-Juicer | 29,825,086 | 20,628,082,262 | 100GB | preprocess: 210min<br>read+unify: 86min<br>clean_html: 15min<br>language_id_score_filter: 18min<br>`total: 391min` |
