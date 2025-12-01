import os
from pathlib import Path

import jsonlines
import pandas as pd
import streamlit as st
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import Analyzer, DefaultExecutor
from data_juicer.utils.constant import HashKeys

demo_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(demo_path))

stack_exchange_recipe_desc = '''
# Alpaca-CoT -- ZH (refined by Data-Juicer)

A refined Chinese version of Alpaca-CoT dataset by [Data-Juicer](https://github.com/datajuicer/data-juicer). Removing some "bad" samples from the original dataset to make it higher-quality.

This dataset is usually used to fine-tune a Large Language Model.

The whole dataset is available [here](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/CFT/alpaca-cot-zh-refine_result.jsonl) (About 18.7GB).

## Dataset Information

- Number of samples: 9,873,214 (Keep ~46.58% from the original dataset)

## Refining Recipe
'''

data_juicer_recipe_desc = '''
# Refined open source dataset by Data-Juicer

We found that there are still some "bad" samples in existing processed datasets (e.g. RedPajama, The Pile.). So we use our Data-Juicer to refine them and try to feed them to LLMs for better performance.

We use simple 3-σ rule to set the hyperparameters for ops in each recipe.

## Before and after refining for Pretraining Dataset

| subset               |       #samples before       | #samples after | keep ratio |data link                                                                                                                                                                                                                                                                                  | source                  |
|----------------------|:---------------------------:|:--------------:|:----------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| arXiv                |          1,724,497          |   1,655,259    |   95.99%   | [redpajama-arxiv-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-arxiv-refine.yaml)                                                                                                                                                                         | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-arxiv-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-arxiv-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-arxiv-refined-by-data-juicer)                                        | Redpajama               |
| Books                |           205,182           |    195,983     |   95.51%   | [redpajama-book-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-book-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-book-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-book-refined-by-data-juicer/summary)   <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-book-refined-by-data-juicer)                                        | Redpajama               |
| Wikipedia            |         29,834,171          |   26,990,659   |   90.47%   | [redpajama-wiki-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-wiki-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-wiki-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-wiki-refined-by-data-juicer/summary)   <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-wiki-refined-by-data-juicer)                                        | Redpajama               |
| C4                   |         364,868,892         |  344,491,171   |   94.42%   | [redpajama-c4-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-c4-refine.yaml)                                                                                                                                                                               | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-c4-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-c4-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-c4-refined-by-data-juicer)                                             | Redpajama               |
| Common Crawl 2019-30 |         81,085,420          |   36,557,283   |   45.08%   | [redpajama-cc-2019-30-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-cc-2019-30-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2019-30-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2019-30-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-cc-2019-30-refined-by-data-juicer)  | Redpajama               |
| Common Crawl 2020-05 |         90,850,492          |   42,612,596   |   46.90%   | [redpajama-cc-2020-05-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-cc-2020-05-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2020-05-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2020-05-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-cc-2020-05-refined-by-data-juicer)  | Redpajama               |
| Common Crawl 2021-04 |         98,878,523          |   44,724,752   |   45.23%   | [redpajama-cc-2021-04-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-cc-2021-04-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2021-04-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2021-04-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-cc-2021-04-refined-by-data-juicer)  | Redpajama               |
| Common Crawl 2022-05 |         94,058,868          |   42,648,496   |   45.34%   | [redpajama-cc-2022-05-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-cc-2022-05-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2022-05-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2022-05-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-cc-2022-05-refined-by-data-juicer)  | Redpajama               |
| Common Crawl 2023-06 |         111,402,716         |   50,643,699   |   45.46%   | [redpajama-cc-2023-06-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-cc-2023-06-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-cc-refine-results/redpajama-cc-2023-06-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-cc-2023-06-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-cc-2023-06-refined-by-data-juicer) | Redpajama               |
| Github Code          | 73,208,524 <br>+ 21,387,703 |   49,279,344   |   52.09%   | [redpajama-code-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/github_code/redpajama-code-refine.yaml)<br>[stack-code-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/github_code/stack-code-refine.yaml)<br>[redpajama-stack-code-deduplicate.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/github_code/redpajama-stack-code-deduplicate.yaml) | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-stack-code-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-stack-code-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-stack-code-refined-by-data-juicer)                             | Redpajama<br>The Stack  |
| StackExchange        |         45,447,328          |   26,309,203   |   57.89%   | [redpajama-pile-stackexchange-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/redpajama-pile-stackexchange-refine.yaml)                                                                                                                                               | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-pile-stackexchange-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/redpajama-pile-stackexchange-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/redpajama-pile-stackexchange-refined-by-data-juicer)             | Redpajama<br>The Pile   |
| EuroParl             |           69,814            |     61,601     |   88.23%   | [pile-europarl-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-europarl-refine.yaml)                                                                                                                                                                             | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-europarl-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-europarl-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-europarl-refined-by-data-juicer)                                   | The Pile                |
| FreeLaw              |          3,562,015          |   2,942,612    |   82.61%   | [pile-freelaw-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-freelaw-refine.yaml)                                                                                                                                                                               | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-freelaw-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-freelaw-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-freelaw-refined-by-data-juicer)                                     | The Pile                |
| HackerNews           |           373,027           |    371,331     |   99.55%   | [pile-hackernews-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-hackernews-refine.yaml)                                                                                                                                                                         | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-hackernews-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-hackernews-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-hackernews-refined-by-data-juicer)                               | The Pile                |
| NIH ExPorter         |           939,661           |    858,492     |   91.36%   | [pile-nih-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-nih-refine.yaml)                                                                                                                                                                                       | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-hin-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-nih-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-nih-refined-by-data-juicer)                                             | The Pile                |
| PhilPapers           |           32,782            |     29,117     |   88.82%   | [pile-philpaper-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-philpaper-refine.yaml)                                                                                                                                                                           | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-philpaper-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-philpaper-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-philpaper-refined-by-data-juicer)                                 | The Pile                |
| PubMed Abstracts     |         15,518,009          |   15,009,325   |   96.72%   | [pile-pubmed-abstract-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-pubmed-abstract-refine.yaml)                                                                                                                                                               | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-pubmed-abstract-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-pubmed-abstracts-refined-by-data-juicer/summary)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-pubmed-abstracts-refined-by-data-juicer)                    | The Pile                |
| PubMed Central       |          3,098,930          |   2,694,860    |   86.96%   | [pile-pubmed-central-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-pubmed-central-refine.yaml)                                                                                                                                                                 | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-pubmed-central-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-pubmed-central-refined-by-data-juicer/summary) <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-pubmed-central-refined-by-data-juicer)                        | The Pile                |
| USPTO                |          5,883,024          |   4,516,283    |   76.77%   | [pile-uspto-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/pretrain/pile-uspto-refine.yaml)                                                                                                                                                                                   | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-uspto-refine-result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/the-pile-uspto-refined-by-data-juicer/summary) <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/the-pile-uspto-refined-by-data-juicer) | The Pile                |


## Before and after refining for Alpaca-CoT Dataset

| subset | #samples before     |             #samples after             | keep ratio | config link                                                                                                                                                                                                                        | data link                                                                                                                                                         | source                 |
|------------------|:-------------------------:|:--------------------------------------:|:----------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| Alpaca-Cot EN | 136,219,879               | 72,855,345 |   54.48%   | [alpaca-cot-en-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot/alpaca-cot-en-refine.yaml)                                                                                                                                                                         | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/CFT/alpaca-cot-en-refine_result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/alpaca-cot-en-refined-by-data-juicer/summary) <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/alpaca-cot-en-refined-by-data-juicer)                          | [39 Subsets of Alpaca-CoT](alpaca_cot/AlpacaCOT.md#refined-alpaca-cot-dataset-meta-info)              |
| Alpaca-Cot ZH | 21,197,246               |               9,873,214                |   46.58%   | [alpaca-cot-zh-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml)                                                                                                                                                                         | [Aliyun](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/CFT/alpaca-cot-zh-refine_result.jsonl) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/alpaca-cot-zh-refined-by-data-juicer/summary) <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/alpaca-cot-zh-refined-by-data-juicer)                          | [28 Subsets of Alpaca-CoT](alpaca_cot/AlpacaCOT.md#refined-alpaca-cot-dataset-meta-info)              |
'''


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


def process_and_show_res():

    config_file = os.path.join(
        project_path,
        'data-juicer-hub/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml')
    dataset_path = os.path.join(demo_path, 'data/alpaca-cot.jsonl')
    export_path = os.path.join(demo_path, 'outputs/processed_alpaca-cot.jsonl')
    cfg_cmd = f'--config {config_file} --dataset_path {dataset_path} --export_path {export_path}'
    args_in_cmd = cfg_cmd.split()
    cfg = init_configs(args=args_in_cmd)
    cfg.open_tracer = True
    cfg.np = 1
    cfg.process.pop(0)
    logger.info('=========Stage 1: analyze original data=========')
    analyzer = Analyzer(cfg)
    analyzed_dataset = analyzer.run()

    logger.info('=========Stage 2: process original data=========')
    executor = DefaultExecutor(cfg)
    processed_dataset = executor.run()
    st.session_state.analyzed_dataset = analyzed_dataset
    st.session_state.processed_dataset = processed_dataset

    trace_dir = executor.tracer.work_dir
    trace_files = list(Path(trace_dir).glob('*jsonl'))
    st.session_state.trace_files = trace_files


class Visualize:

    @staticmethod
    def setup():
        # get data-juicer-hub
        if not os.path.exists(os.path.join(project_path, 'data-juicer-hub')):
            import subprocess
            subprocess.run(
                ['git', 'clone',
                 'https://github.com/datajuicer/data-juicer-hub.git'],
                cwd=project_path,
                check=True)

        st.set_page_config(
            page_title='Data-Juicer',
            page_icon=':smile',
            layout='wide',
            # initial_sidebar_state="expanded",
        )

        readme_link = 'https://github.com/alibaba/data-juicer'
        st.markdown(
            '<div align = "center"> <font size = "70"> Data-Juicer \
            </font> </div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div align = "center"> A One-Stop Data Processing System for \
                Large Language Models, \
                see more details in our <a href={readme_link}>Github</a></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def show_recipe():

        def show_yaml(config_file):
            with open(config_file, 'r') as f:
                st.code(f.read(), language='yaml', line_numbers=False)

        with st.expander('Data-Juicer-Alpaca-CoT Chinese Recipe',
                         expanded=False):
            st.markdown(stack_exchange_recipe_desc)
            config_file = os.path.join(
                project_path,
                'data-juicer-hub/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml'
            )
            show_yaml(config_file)

    @staticmethod
    def analyze_process():

        start_btn_process = st.button('Start to process data',
                                      use_container_width=True)

        with st.expander('Data Processing Results', expanded=True):

            if start_btn_process:
                with st.spinner('Wait for process...'):
                    process_and_show_res()

            col1, col2 = st.columns(2)
            with col1:
                st.header('Original Data')
                analyzed_dataset = st.session_state.get(
                    'analyzed_dataset', None)
                st.dataframe(analyzed_dataset, use_container_width=True)
                st.download_button('Download Original data as JSONL',
                                   data=convert_to_jsonl(
                                       pd.DataFrame(analyzed_dataset)),
                                   file_name='original_dataset.jsonl')

            with col2:
                st.header('Processed Data')
                processed_dataset = st.session_state.get(
                    'processed_dataset', None)
                st.dataframe(processed_dataset, use_container_width=True)
                st.download_button('Download Processed data as JSONL',
                                   data=convert_to_jsonl(
                                       pd.DataFrame(processed_dataset)),
                                   file_name='processed_dataset.jsonl')

            trace_files = st.session_state.get('trace_files', [])

            def display_tracer_result(op_type, prefix, files):
                st.subheader(op_type)
                for file in files:

                    filename = file.stem
                    filepath = str(file)

                    if filename.startswith(prefix):
                        st.markdown(f'- {filename.split(prefix)[1]}')
                        with jsonlines.open(filepath, 'r') as reader:
                            objs = [obj for obj in reader]
                        for obj in objs:
                            # simhash value may exceed the range of
                            # integer type of streamlit
                            if 'simhash_deduplicator' in filename:
                                obj['dup1'].pop(HashKeys.simhash)
                                obj['dup2'].pop(HashKeys.simhash)

                        st.dataframe(objs)

            if len(trace_files) > 0:

                st.header('Tracer Results')
                for op_type, prefix in zip(
                    ['Mapper', 'Filter', 'Deduplicator'],
                    ['mapper-', 'filter-', 'duplicate-']):
                    display_tracer_result(op_type, prefix, trace_files)
        with st.expander('Refined Datasets by Data-Juicer', expanded=False):
            st.markdown(data_juicer_recipe_desc)

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.show_recipe()
        Visualize.analyze_process()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
