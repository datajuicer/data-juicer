import os

import streamlit as st
from loguru import logger

from quality_classifier.qc_utils import (init_spark, load_dataset, predict,
                                         prepare_model)


@st.cache_data
def install_jdk():

    os.system('apt update')
    os.system('apt install -y default-jre')
    os.system('apt install -y default-jdk')
    os.system('export JAVA_HOME=/usr/lib/jvm/default-java')


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


@st.cache_resource
def st_init_spark():
    return init_spark()


@st.cache_resource
def st_prepare_model(model_name):
    return prepare_model(model_name)


def st_load_dataset(spark, ds_path, text_key='text', only_text=False):
    return load_dataset(spark=spark,
                        ds_path=ds_path,
                        text_key=text_key,
                        only_text=only_text)


def st_predict(model, ds, tokenizer=None, keep_method='label'):
    return predict(model=model,
                   ds=ds,
                   tokenizer=tokenizer,
                   keep_method=keep_method)


def quality_classifier(dataset_file, model):

    del_file = False

    logger.info('=========Stage: analyze original data=========')
    if dataset_file is not None:
        file_contents = dataset_file.getvalue()
        with open(dataset_file.name, 'wb') as f:
            f.write(file_contents)
        dataset_path = dataset_file.name
        del_file = True
    else:
        dataset_path = st.session_state.get('default_demo_dataset')

    if model == 'chinese':
        tokenizer = 'zh.sp.model'
        keep_method = 'label'
    if model == 'code':
        tokenizer = 'code.sp.model'
        keep_method = 'label'
    if model == 'gpt3':
        tokenizer = None
        keep_method = 'gpt3'

    spark = st_init_spark()
    model = st_prepare_model(model_name=model)
    ds = st_load_dataset(spark, dataset_path)

    pred = st_predict(model, ds, tokenizer=tokenizer, keep_method=keep_method)
    overall = pred.select('doc_score').toPandas().describe(include='all')

    st.session_state.dataset = pred
    st.session_state.original_overall = overall
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

        install_jdk()

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
    def quality():
        col1, col2 = st.columns(2)
        with col1:
            dataset_file = st.file_uploader(
                label='Upload your custom dataset(jsonl/parquet)',
                type=['json', 'jsonl', 'parquet'])

            st.text_input(label='Default Demo dataset',
                          disabled=True,
                          key='default_demo_dataset',
                          value='data/demo-dataset.jsonl')
        with col2:
            label = 'Select a quality classifier'
            quality_model_map = {
                'Chinese quality classifier': 'chinese',
                'Code quality classifier': 'code',
                'GPT3 quality classifier': 'gpt3'
            }

            selected_model = st.selectbox(label=label,
                                          options=list(
                                              quality_model_map.keys()))
        model_name = quality_model_map[selected_model]

        start_btn = st.button(
            f'2. Start to analyze dataset with {selected_model}',
            use_container_width=True)

        with st.expander(f'{selected_model} Results', expanded=True):

            if start_btn:
                with st.spinner('Wait for analyze...'):
                    quality_classifier(dataset_file, model_name)

            col1, col2 = st.columns(2)
            with col1:
                original_overall = st.session_state.get(
                    'original_overall', None)
                st.header('Statistics')
                st.dataframe(original_overall, use_container_width=True)
            with col2:
                pred = st.session_state.get('dataset', None)
                st.header('Details')
                st.dataframe(pred, use_container_width=True)

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.quality()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
