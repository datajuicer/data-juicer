import os

import pandas as pd
import streamlit as st
import yaml
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import Analyzer
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

    cfg_cmd = '--config configs/demo.yaml'

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

        return pretty_out(parsed_cfg), pretty_out(specified_cfg), parsed_cfg
    except Exception as e:
        return str(e), pretty_out(specified_cfg), None


def analyze_and_show_res(dataset_file):

    images_ori = []
    cfg = st.session_state.get('cfg', parse_cfg()[2])
    if cfg is None:
        raise ValueError('you have not specify valid cfg')
    # force generating separate figures
    cfg['save_stats_in_one_file'] = True

    del_file = False
    logger.info('=========Stage: analyze original data=========')
    if dataset_file is not None:

        file_contents = dataset_file.getvalue()
        with open(dataset_file.name, 'wb') as f:
            f.write(file_contents)
        cfg.dataset_path = dataset_file.name
        del_file = True

    analyzer = Analyzer(cfg)
    dataset = analyzer.run()

    overall_file = os.path.join(analyzer.analysis_path, 'overall.csv')
    analysis_res_ori = pd.DataFrame()
    if os.path.exists(overall_file):
        analysis_res_ori = pd.read_csv(overall_file)

    if os.path.exists(analyzer.analysis_path):
        for f_path in os.listdir(analyzer.analysis_path):
            if '.png' in f_path and 'all-stats' in f_path:
                images_ori.append(os.path.join(analyzer.analysis_path, f_path))

    st.session_state.dataset = dataset
    st.session_state.original_overall = analysis_res_ori
    st.session_state.original_imgs = images_ori
    if del_file:
        os.remove(dataset_file.name)


class Visualize:

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Data-Juicer',
            page_icon=':smile',
            layout='wide',
            # initial_sidebar_state="expanded",
        )

        readme_link = 'https://github.com/datajuicer/data-juicer'
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
    def analyze_process():
        col1, col2 = st.columns(2)
        with col1:
            dataset_file = st.file_uploader(
                label='Upload your custom dataset csv or jsonl',
                type=['csv', 'json', 'jsonl'])
        with col2:
            st.text_area(label='Default Demo dataset',
                         disabled=True,
                         value='demo/demo-dataset.jsonl')

        start_btn = st.button(
            '2. Start to analyze original data (per filter op)',
            use_container_width=True)

        with st.expander('Data Analysis Results', expanded=True):

            if start_btn:
                with st.spinner('Wait for analyze...'):
                    analyze_and_show_res(dataset_file)

            original_overall = st.session_state.get('original_overall', None)
            original_imgs = st.session_state.get('original_imgs', [])

            st.header('Statistics')
            st.dataframe(original_overall, use_container_width=True)
            if len(original_imgs) > 0:
                st.header('Histograms')
                for img in original_imgs:
                    st.image(img, output_format='png', use_column_width=True)

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.analyze_process()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
