import inspect
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from data_juicer.config import init_configs
from data_juicer.core import Analyzer
from data_juicer.format.formatter import FORMATTERS
from data_juicer.ops.base_op import OPERATORS

demo_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(demo_path))

project_desc = '''
**Data-Juicer** is a one-stop data processing system to make data higher-quality, juicier, and more digestible for LLMs.
This project is being actively updated and maintained, and we will periodically enhance and add more features and data recipes. We welcome you to join us in promoting LLM data development and research!
'''

features_desc = '''
- **Broad Range of Operators**: Equipped with 50+ core :blue[operators (OPs)], including `Formatters`, `Mappers`, `Filters`, `Deduplicators`, and beyond.

- **Specialized Toolkits**: Feature-rich specialized toolkits such as `Text Quality Classifier`, `Dataset Splitter`, `Analyzers`, `Evaluators`, and more that elevate your dataset handling capabilities.

- **Systematic & Reusable**: Empowering users with a systematic library of reusable `config recipes` and `OPs`, designed to function independently of specific datasets, models, or tasks.

- **Data-in-the-loop**: Allowing detailed data analyzes with an automated report generation feature for a deeper understanding of your dataset. Coupled with timely multi-dimension automatic evaluation capabilities, it supports a feedback loop at multiple stages in the LLM development process.

- **Comprehensive Processing Recipes**: Offering tens of `pre-built data processing recipes` for pre-training, CFT, en, zh, and more scenarios.

- **User-Friendly Experience**: Designed for simplicity, with `comprehensive documentation`, `easy start guides` and `demo configs`, and intuitive configuration with simple adding/removing OPs from existing configs.

- **Flexible & Extensible**: Accommodating most types of data formats (e.g., jsonl, parquet, csv, ...) and allowing flexible combinations of OPs. Feel free to `implement your own OPs` for customizable data processing.

- **Enhanced Efficiency**: Providing a speedy data processing pipeline requiring less memory, optimized for maximum productivity.
'''

quick_start_desc = '''
### Data Processing

- Run `process_data.py` tool with your config as the argument to process
  your dataset.

```shell
python tools/process_data.py --config demos/process_simple/process.yaml
```

### Data Analysis
- Run `analyze_data.py` tool with your config as the argument to analyze your dataset.

```shell
python tools/analyze_data.py --config demos/analyze_simple/analyzer.yaml
```

- **Note:** Analyzer only compute stats of Filter ops. So extra Mapper or Deduplicator ops will be ignored in the analysis process.

### Data Visualization

- Run `app.py` tool to visualize your dataset in your browser.

```shell
streamlit run app.py
```
'''

config_desc = '''

Data-Juicer provides some configuration files to allow users to easily understand the configuration methods of various functions and quickly reproduce the processing flow of different datasets.

### Usage

```shell
# To process your dataset.
python tools/process_data.py --config xxx.yaml
# To analyze your dataset.
python tools/analyze_data.py --config xxx.yaml
```
'''

config_all_desc = '''
`config_all.yaml` which includes **all** ops and default arguments. You just need to **remove** ops that you won't use and refine some arguments of ops.
'''

with open(os.path.join(project_path, 'docs/Operators.md'), 'r') as f:
    op_text = f.read()

def extract_op_desp(markdown_text,  header):
    start_index = markdown_text.find(header)
    end_index = markdown_text.find("\n##", start_index + len(header)) 
    return markdown_text[start_index+ len(header):end_index].strip()

op_desc = extract_op_desp(op_text, '## Overview')
op_list_desc = {
    'formatter': extract_op_desp(op_text, '## Formatter <a name="formatter"/>'),
    'mapper':extract_op_desp(op_text, '## Mapper <a name="mapper"/>'),
    'filter':extract_op_desp(op_text, '## Filter <a name="filter"/>'),
    'deduplicator':extract_op_desp(op_text, '## Deduplicator <a name="deduplicator"/>'),
    'selector':extract_op_desp(op_text, '## Selector <a name="selector"/>'),
}

demo_desc = '''
- Introduction to Data-Juicer [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
- Data Visualization:
  - Basic Statistics [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  - Lexical Diversity [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  - Operator Effect [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
- Data Processing:
  - Scientific Literature (e.g. [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  - Programming Code (e.g. [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  - Chinese Instruction Data (e.g. [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
- Tool Pool:
  - Dataset Splitting by Language [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  - Quality Classifier for CommonCrawl [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  - Auto Evaluation on [HELM](https://github.com/stanford-crfm/helm) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  - Data Sampling and Mixture [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
- Data Processing Loop [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]
- Data Processing HPO [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_hpo/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_hpo)]
'''


def run_demo():

    config_file = os.path.join(project_path, 'demos/analyze_simple/analyzer.yaml')
    data_path = os.path.join(demo_path, 'data/demo-dataset.jsonl')
    st.markdown(f'dataset: `{data_path}`')
    start_btn = st.button(' Start to analyze', use_container_width=True)

    cfg_cmd = f'--config {config_file} --dataset_path {data_path}'
    args_in_cmd = cfg_cmd.split()
    cfg = init_configs(args=args_in_cmd)

    images_ori = []
    cfg['save_stats_in_one_file'] = True

    if start_btn:
        analyzer = Analyzer(cfg)

        with st.spinner('Wait for analyze...'):
            analyzer.run()

        overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
        analysis_res_ori = pd.DataFrame()
        if os.path.exists(overall_file):
            analysis_res_ori = pd.read_csv(overall_file)

        if os.path.exists(analyzer.analysis_path):
            for f_path in os.listdir(analyzer.analysis_path):
                if '.png' in f_path and 'all-stats' in f_path:
                    images_ori.append(
                        os.path.join(analyzer.analysis_path, f_path))

        st.subheader('Statistics')
        st.dataframe(analysis_res_ori, use_container_width=True)
        if len(images_ori) > 0:
            st.subheader('Histograms')
            for img in images_ori:
                st.image(img, output_format='png', use_column_width=True)


class Visualize:

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Data-Juicer',
            page_icon=':smile',
            # layout='wide',
            # initial_sidebar_state="expanded",
        )

        readme_link = 'https://github.com/alibaba/data-juicer'

        st.markdown(
            '<div align = "center"> <font size = "70"> Data-Juicer  \
            </font> </div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div align = "center"> A One-Stop Data Processing System for \
                Large Language Models, \
                see more details in <a href={readme_link}>GitHub</a></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def overrall():

        st.image(os.path.join(project_path, 'docs/imgs/data-juicer.jpg'),
                 output_format='jpg',
                 use_column_width=True)
        st.markdown(project_desc)
        with st.expander('Features', expanded=False):
            st.markdown(features_desc)
        with st.expander('Quick Start', expanded=False):
            st.markdown(quick_start_desc)
        with st.expander('Configuration', expanded=False):
            st.markdown(config_desc)
            st.markdown('### Recipes')
            tab1, tab2, tab3, tab4 = st.tabs([
                ':blue[ Reproduced Redpajama]', ':blue[ Reproduced BLOOM]',
                ':blue[ Data-Juicer Recipes]', ':blue[ Config-all]'
            ])

            def show_yaml(config_file):
                with open(config_file, 'r') as f:
                    st.code(f.read(), language='yaml', line_numbers=False)

            # get data-juicer-hub
            if not os.path.exists(os.path.join(project_path, 'data-juicer-hub')):
                import subprocess
                subprocess.run(
                    ['git', 'clone',
                     'https://github.com/datajuicer/data-juicer-hub.git'],
                    cwd=project_path,
                    check=True)

            with tab1:
                label = 'Data-Juicer have reproduced the processing \
                         flow of some RedPajama datasets.'

                config_files = Path(
                    os.path.join(
                        project_path,
                        'data-juicer-hub/reproduced_redpajama')).glob('*.yaml')
                config_dict = {
                    config.stem: str(config)
                    for config in config_files
                }

                selected = st.selectbox(
                    label=label,
                    options=sorted(list(config_dict.keys())),
                )
                show_yaml(config_dict[selected])

            with tab2:
                label = 'Data-Juicer have reproduced the processing flow \
                    of some BLOOM datasets.'

                config_files = Path(
                    os.path.join(project_path,
                                 'data-juicer-hub/reproduced_bloom')).glob('*.yaml')
                config_dict = {
                    config.stem: str(config)
                    for config in config_files
                }

                selected = st.selectbox(
                    label=label,
                    options=sorted(list(config_dict.keys())),
                )
                show_yaml(config_dict[selected])
            with tab3:
                label = 'Data-Juicer have refined some open source datasets \
                    (including CFT datasets) by using Data-Juicer and have \
                        provided configuration files for the refine flow.'

                config_files = Path(
                    os.path.join(
                        project_path,
                        'data-juicer-hub/refined_recipes')).rglob('*.yaml')
                config_dict = {
                    config.stem: str(config)
                    for config in config_files
                }

                selected = st.selectbox(
                    label=label,
                    options=sorted(list(config_dict.keys())),
                )
                show_yaml(config_dict[selected])
            with tab4:
                st.markdown(config_all_desc)
                show_yaml(os.path.join(project_path,
                                       'data_juicer/config/config_all.yaml'))

        with st.expander('Operators', expanded=False):

            st.markdown(op_desc)
            st.markdown('### Operator List')
            tabs = st.tabs([
                ':blue[ Formatters]',
                ':blue[ Mappers]',
                ':blue[ Filters]',
                ':blue[ Deduplicators]',
                ':blue[ Selecttors]',
            ])

            for op_type, tab in zip(
                ['formatter', 'mapper', 'filter', 'deduplicator', 'selector'],
                    tabs):
                with tab:
                    show_list = st.checkbox(f'Show {op_type} list')
                    if show_list:
                        st.markdown(op_list_desc[op_type])
                    if op_type == 'formatter':
                        repo = FORMATTERS
                        options = list(repo.modules.keys())

                    else:
                        repo = OPERATORS
                        all_ops = list(repo.modules.keys())
                        options = [
                            name for name in all_ops if name.endswith(op_type)
                        ]

                    label = f'Select a {op_type} to show details'
                    op_name = st.selectbox(label=label, options=options)

                    op_class = repo.modules[op_name]

                    st.markdown('#### Source Code')
                    text = inspect.getsourcelines(op_class)
                    st.code(''.join(text[0]),
                            language='python',
                            line_numbers=False)
        with st.expander('Demos', expanded=False):
            st.markdown('### Total Demos')
            st.markdown(demo_desc)
            st.markdown('### First Demo: Basic Statistics')
            run_demo()

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.overrall()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
