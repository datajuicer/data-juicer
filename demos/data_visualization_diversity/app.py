import os

import plotly.express as px
import streamlit as st
import yaml
from loguru import logger

from data_juicer.analysis.diversity_analysis import (DiversityAnalysis,
                                                     get_diversity)
from data_juicer.config import init_configs
from data_juicer.core import Analyzer
from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.model_utils import prepare_model, get_model


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


@st.cache_data
def get_diversity_model(lang):
    model_key = prepare_model('spacy', lang=lang)
    diversity_model = get_model(model_key)
    return diversity_model


@st.cache_data
def postproc_diversity(dataframe, **kwargs):
    df = get_diversity(dataframe, **kwargs)
    return df


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


def load_dataset(dataset_file):

    cfg = st.session_state.get('cfg', parse_cfg()[2])
    if cfg is None:
        raise ValueError('you have not specify valid cfg')
    # force generating separate figures
    cfg['save_stats_in_one_file'] = True

    del_file = False
    if dataset_file is not None:

        file_contents = dataset_file.getvalue()
        with open(dataset_file.name, 'wb') as f:
            f.write(file_contents)
        cfg.dataset_path = dataset_file.name
        del_file = True

    logger.info('=========Stage: analyze original data=========')
    analyzer = Analyzer(cfg)

    dataset = analyzer.formatter.load_dataset()
    if del_file:
        os.remove(dataset_file.name)
    return dataset


class Visualize:

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Juicer',
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
    def draw_sunburst(df, path, values):

        fig = px.sunburst(df, path=path, values=values)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          font_family='Times New Roman',
                          font=dict(size=40))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def diversity():
        col1, col2 = st.columns(2)
        with col1:
            dataset_file = st.file_uploader(
                label='Upload your custom dataset(jsonl/csv)',
                type=['json', 'jsonl', 'csv'])

        with col2:
            st.text_area(label='Default Demo dataset',
                         disabled=True,
                         value='data/demo-dataset.jsonl')

        with st.expander('Set diversity params', expanded=True):

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                label = 'Which language of your dataset'
                options = ['en', 'zh']
                lang_select = st.selectbox(
                    label=label,
                    options=options,
                )
            with col2:
                top_k_verbs = st.number_input('Set the top_k of verbs',
                                              value=20)
            with col3:
                top_k_nouns = st.number_input('Set the top_k of nouns',
                                              value=4)
            with col4:
                threshold = st.slider('Count threshold',
                                      min_value=0,
                                      value=0,
                                      max_value=100,
                                      step=1)
        diversity_btn = st.button('Start to analyze Verb-Noun diversity',
                                  use_container_width=True)

        with st.expander('Diversity Results ', expanded=True):

            cfg = st.session_state.get('cfg', parse_cfg()[2])
            output_path = os.path.join(os.path.dirname(cfg.export_path),
                                       'analysis')
            raw_df = None
            if diversity_btn:
                try:
                    with st.spinner('Wait for analyze diversity...'):
                        dataset = load_dataset(dataset_file)

                        diversity_analysis = DiversityAnalysis(
                            dataset, output_path)

                        raw_df = diversity_analysis.compute(
                            lang_or_model=get_diversity_model(lang_select))

                    st.session_state[f'diversity{lang_select}'] = raw_df

                except Exception as e:
                    st.warning(f'Error {str(e)} in {lang_select}')
            else:
                raw_df = st.session_state.get(f'diversity{lang_select}', None)

            if raw_df is not None:
                df = postproc_diversity(raw_df,
                                        top_k_verbs=top_k_verbs,
                                        top_k_nouns=top_k_nouns)
                df = df[df['count'] >= threshold]
                Visualize.draw_sunburst(df,
                                        path=['verb', 'noun'],
                                        values='count')

                st.download_button(
                    label='Download diversity data as CSV',
                    data=convert_to_csv(df),
                    file_name='diversity.csv',
                    mime='text/csv',
                )

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.diversity()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
