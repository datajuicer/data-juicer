import copy
import os

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import Analyzer, DefaultExecutor
from data_juicer.ops.base_op import OPERATORS


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


def pretty_out(d):
    res = ''
    process = ''
    op_names = set(OPERATORS.modules.keys())
    for key, value in d.items():
        if key == 'process':
            process = yaml.dump(value,
                                allow_unicode=True,
                                default_flow_style=False)
        elif key == 'config' or key.split('.')[0] in op_names:
            continue
        else:
            res += f'{key}:\n \t {value}\n'
    res += 'process:\n' + \
           '\n'.join(['\t' + line for line in process.splitlines()])

    return res


def parse_cfg():

    cfg_file = st.session_state.input_cfg_file
    cfg_cmd = st.session_state.input_cfg_cmd

    cfg_f_name = 'null'
    del_cfg_file = False
    if cfg_file is not None:
        cfg_f_name = cfg_file.name
        file_contents = cfg_file.getvalue()
        with open(cfg_f_name, 'wb') as f:
            f.write(file_contents)
        cfg_cmd = f'--config {cfg_f_name}'
        del_cfg_file = True

    args_in_cmd = cfg_cmd.split()

    if len(args_in_cmd) >= 2 and args_in_cmd[0] == '--config':
        cfg_f_name = args_in_cmd[1]
    else:
        st.warning('Please specify a config command or upload a config file.')
        st.stop()

    if not os.path.exists(cfg_f_name):
        st.warning('do not parse'
                   f'config file does not exist with cfg_f_name={cfg_f_name}')
        st.stop()

    with open(cfg_f_name, 'r') as cfg_f:
        specified_cfg = yaml.safe_load(cfg_f)

    try:
        parsed_cfg = init_configs(args=args_in_cmd)
        st.session_state.cfg = parsed_cfg
        if del_cfg_file:
            os.remove(cfg_f_name)
        return pretty_out(parsed_cfg), pretty_out(specified_cfg), parsed_cfg
    except Exception as e:
        return str(e), pretty_out(specified_cfg), None


def analyze_and_show_res():
    images_ori = []
    cfg = st.session_state.get('cfg', parse_cfg()[2])
    if cfg is None:
        raise ValueError('you have not specify valid cfg')
    # force generating separate figures
    cfg['save_stats_in_one_file'] = True

    logger.info('=========Stage 1: analyze original data=========')
    analyzer = Analyzer(cfg)
    analyzed_dataset = analyzer.run()

    overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
    analysis_res_ori = pd.DataFrame()
    if os.path.exists(overall_file):
        analysis_res_ori = pd.read_csv(overall_file)

    if os.path.exists(analyzer.analysis_path):
        for f_path in os.listdir(analyzer.analysis_path):
            if '.png' in f_path and 'all-stats' in f_path:
                images_ori.append(os.path.join(analyzer.analysis_path, f_path))

    st.session_state.analyzed_dataset = analyzed_dataset
    st.session_state.original_overall = analysis_res_ori
    st.session_state.original_imgs = images_ori


def process_and_show_res():
    images_processed = []
    cfg = st.session_state.get('cfg', parse_cfg()[2])
    if cfg is None:
        raise ValueError('you have not specify valid cfg')
    # force generating separate figures
    cfg['save_stats_in_one_file'] = True
    logger.info('=========Stage 2: process original data=========')
    executor = DefaultExecutor(cfg)
    processed_dataset = executor.run()

    logger.info('=========Stage 3: analyze the processed data==========')
    analysis_res_processed = pd.DataFrame()
    try:
        if len(processed_dataset) > 0:
            cfg_for_processed_data = copy.deepcopy(cfg)
            cfg_for_processed_data.dataset_path = cfg.export_path

            cfg_for_processed_data.export_path = os.path.dirname(
                cfg.export_path) + '_processed/data.jsonl'

            analyzer = Analyzer(cfg_for_processed_data)
            analyzer.analysis_path = os.path.dirname(
                cfg_for_processed_data.export_path) + '/analysis'
            analyzer.run()

            overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
            if os.path.exists(overall_file):
                analysis_res_processed = pd.read_csv(overall_file)

            if os.path.exists(analyzer.analysis_path):
                for f_path in os.listdir(analyzer.analysis_path):
                    if '.png' in f_path and 'all-stats' in f_path:
                        images_processed.append(
                            os.path.join(analyzer.analysis_path, f_path))
        else:
            st.warning('No sample left after processing. Please change \
                anther dataset or op parameters then rerun')
    except Exception as e:
        st.warning(f'Something error with {str(e)}')

    logger.info('=========Stage 4: Render the analysis results==========')
    st.session_state.processed_dataset = processed_dataset
    st.session_state.processed_overall = analysis_res_processed
    st.session_state.processed_imgs = images_processed


class Visualize:

    @staticmethod
    def setup():
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
    def parser():
        with st.expander('Configuration', expanded=True):
            st.markdown('Please specify the cfg via '
                        '(i) specifying the cfg file path with commands or '
                        '(ii) uploading the cfg file.')

            col1, col2 = st.columns(2)
            with col1:
                example_cfg_f = os.path.abspath(
                    os.path.join(os.path.dirname(__file__),
                                 './configs/demo.yaml'))
                st.text_area(label='(i) Input Cfg Commands',
                             key='input_cfg_cmd',
                             value=f'--config {example_cfg_f}')
                example_my_cmd = '--dataset_path ./data/demo-dataset.jsonl ' \
                                 '--export_path '\
                                 './outputs/demo/demo-processed.jsonl'

                st.text_area(
                    label='cmd example. (the cmd-args will override '
                    'yaml-file-args)',
                    disabled=True,
                    value=f'--config {example_cfg_f} {example_my_cmd}')

            with col2:
                st.file_uploader(label='(ii) Input Cfg File',
                                 key='input_cfg_file',
                                 type=['yaml'])

            btn_show_cfg = st.button('1. Parse Cfg', use_container_width=True)
            if btn_show_cfg:
                text1, text2, cfg = parse_cfg()
                st.session_state.cfg_text1 = text1
                st.session_state.cfg_text2 = text2

            else:
                text1 = st.session_state.get('cfg_text1', '')
                text2 = st.session_state.get('cfg_text2', '')

            col3, col4 = st.columns(2)
            with col3:
                st.text_area(label='Parsed Cfg (in memory)', value=text1)
            with col4:
                st.text_area(label='Specified Cfg (in yaml file)', value=text2)

    @staticmethod
    def analyze_process():
        start_btn = st.button(
            '2. Start to analyze original data (per filter op)',
            use_container_width=True)
        start_btn_process = st.button('3. Start to process data',
                                      use_container_width=True)

        with st.expander('Data Analysis Results', expanded=True):

            if start_btn:
                with st.spinner('Wait for analyze...'):
                    analyze_and_show_res()

            if start_btn_process:
                with st.spinner('Wait for process...'):
                    process_and_show_res()

            original_overall = st.session_state.get('original_overall', None)
            original_imgs = st.session_state.get('original_imgs', [])
            processed_overall = st.session_state.get('processed_overall', None)
            processed_imgs = st.session_state.get('processed_imgs', [])

            display_dataset_details = st.checkbox('Display dataset details')

            col1, col2 = st.columns(2)
            with col1:
                st.header('Original Data')
                if display_dataset_details:
                    st.subheader('Details')
                    analyzed_dataset = st.session_state.get(
                        'analyzed_dataset', None)
                    st.dataframe(analyzed_dataset, use_container_width=True)
                    st.download_button('Download Original data as JSONL',
                                       data=convert_to_jsonl(
                                           pd.DataFrame(analyzed_dataset)),
                                       file_name='original_dataset.jsonl')

            with col2:
                st.header('Processed Data')
                if display_dataset_details:
                    st.subheader('Details')
                    processed_dataset = st.session_state.get(
                        'processed_dataset', None)
                    st.dataframe(processed_dataset, use_container_width=True)
                    st.download_button('Download Processed data as JSONL',
                                       data=convert_to_jsonl(
                                           pd.DataFrame(processed_dataset)),
                                       file_name='processed_dataset.jsonl')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Statistics')
                st.dataframe(original_overall, use_container_width=True)
                for img in original_imgs:
                    st.image(img, output_format='png')

            with col2:
                st.subheader('Statistics')
                st.dataframe(processed_overall, use_container_width=True)
                for img in processed_imgs:
                    st.image(img, output_format='png')

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.parser()
        Visualize.analyze_process()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
