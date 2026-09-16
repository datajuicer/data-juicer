# Redpajama Config Files

The [folder](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama) in Data-Juicer-Hub contains example configuration files to easily and quickly reproduce the processing flow of [Redpajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep).

Before starting, please clone the [data-juicer-hub](https://github.com/datajuicer/data-juicer-hub) to local, which contains lots of data processing recipes.
```shell
git clone https://github.com/datajuicer/data-juicer-hub.git
```

## arXiv
The raw data files can be downloaded from the same AWS link as in [Redpajama/arXiv](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/arxiv).

Once downloaded, use [raw_arxiv_to_jsonl.py](../../tools/preprocess/raw_arxiv_to_jsonl.py) to convert from the original format to `jsonl` that Data-Juicer can handle easily:

```shell
python tools/preprocess/raw_arxiv_to_jsonl.py           \
    --arxiv_src_dir       <arxiv_src_dir>    \
    --target_dir          <target_dir>       \
    --temp_dir            <temp_dir>         \
    --num_proc            <num_proc>
```

After conversion, modify the path configurations in [redpajama-arxiv.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-arxiv.yaml) and execute the following command to reproduce the processing flow of RedPajama:

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-arxiv.yaml
```

### Comparison

| | num_samples | num_tokens | peak_memory | wall_time |
| --- | :---: | :---: | :---: | --- |
| redpajama | 1,724,497 | 30,667,506,934 | 35GB |`total: 11h52min` |
| Data-Juicer | 2,675,426| 30,338,153,178 | 21GB | preprocess: 5h21min<br>read+unify: 25min<br>remove_header_mapper: 5min<br>remove_comments_mapper: 3min<br> remove_bibliography_mapper: 4min<br>expand_macro_mapper: 5min19s<br>text_length_filter: 4min<br>export: 43min<br>`total: 6h53min` |

## Books

The raw data files can be downloaded from the same HuggingFace datasets as in [Redpajama/Books](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/book).

Once downloaded, modify the path configurations in [redpajama-books.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-books.yaml) and execute the following command to reproduce the processing flow of RedPajama.

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-books.yaml
```

### Comparison

| | num_samples | num_tokens | peak_memory | wall_time |
| --- | :---: | :---: | :---: | --- |
| redpajama | 205,183 | 25,962,395,123 | 450GB | split_for_dedup: 5min<br>dedup: 117min<br> `total: 122min` |
| Data-Juicer | 207,902 | 26,108,635,683 | 96GB | read+unify: 20min<br>compute_hash: 78min<br>dedup: 3min<br>export: 3min<br>`total: 114min` |

## Code

The raw data files can be downloaded from Google BigQuery as in [Redpajama/Code](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/github).

Once downloaded, unzip and delete files whose extensions are not in the following whitelist:

```text
.asm, .bat, .cmd, .c, .h, .cs, .cpp, .hpp, .c++, .h++, .cc, .hh, .C, .H, .cmake, .css, .dockerfile, .f90, .f, .f03, .f08, .f77, .f95, .for, .fpp, .go, .hs, .html, .java, .js, .jl, .lua, .md, .markdown, .php, .php3, .php4, .php5, .phps, .phpt, .pl, .pm, .pod, .perl,  ps1, .psd1, .psm1, .py, .rb, .rs, .sql, .scala, .sh, .bash, .command, .zsh, .ts, .tsx, .tex, .vb, Dockerfile, Makefile, .xml, .rst, .m, .smali
```

After preparation, modify the path configurations in [redpajama-code.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-code.yaml) and execute the following command to reproduce the processing flow of redpajama:

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-code.yaml
```

### Comparison

| | num_samples | num_tokens | peak_memory | wall_time |
| --- | :---: | :---: | :---: | --- |
| redpajama | 73,208,524 | 150,390,270,060| 212GB | local-dedup: 37h<br>global-dedup: 1h<br>merge-dedup: 6h<br>filter: 17h<br>`total: 61h` |
| Data-Juicer | 73,169,889| 150,310,903,230| 370GB | preprocess: 5h21min<br>read+unify: 12h<br>document_deduplicator: 20h<br>clean_copyright_mappe:  3h<br>maximum_line_length_filter: 2.5h<br>average_line_length_filter: 2h<br>alphanumeric_filter: 13h<br>export: 2.5h<br>`total: 59h` |

## StackExchange

The raw data files can be downloaded from the same Archive link as in [Redpajama/Stack_exchange](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/stack_exchange).

Once downloaded, use [raw_stackexchange_to_jsonl.py](../../tools/preprocess/raw_stackexchange_to_jsonl.py) to convert from the original format to `jsonl` that Data-Juicer can handle easily:

```shell
python tools/preprocess/raw_arxiv_stackexchange_to_jsonl.py           \
    --src_dir       <src_dir>      \
    --target_dir    <target_dir>   \
    --topk          <topk>         \
    --num_proc      <num_proc>     \
```

After conversion, modify the path configurations in [redpajama-stackexchange.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/reproduced_redpajama/redpajama-stackexchange.yaml) and execute the following command to reproduce the processing flow of redpajama:

```shell
python tools/process_data.py --config <path-to-data-juicer-hub>/reproduced_redpajama/redpajama-stackexchange.yaml
```

### Comparison

| | num_samples | num_tokens | peak_memory | wall_time |
| --- | :---: | :---: | :---: | --- |
| redpajama | 29,825,086 | 20,502,757,123 | >500GB | filter: 170min<br>postprocess: 90min<br>`total: 260min` |
| Data-Juicer | 29,825,086 | 20,628,082,262 | 100GB | preprocess: 210min<br>read+unify: 86min<br>clean_html: 15min<br>language_id_score_filter: 18min<br>`total: 391min` |
