import os
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_splitting_by_language import main as split_dataset_by_language


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


def split_dataset(dataset_file, target_dir):

    del_file = False

    if dataset_file is not None:
        file_contents = dataset_file.getvalue()
        with open(dataset_file.name, 'wb') as f:
            f.write(file_contents)
        dataset_path = dataset_file.name
        del_file = True
    else:
        dataset_path = st.session_state.get('default_demo_dataset')
    split_dataset_by_language(dataset_path, target_dir)

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
    def split():
        col1, col2 = st.columns(2)
        with col1:
            dataset_file = st.file_uploader(
                label='Upload your custom dataset(jsonl/parquet)',
                type=['json', 'jsonl', 'parquet'])

        with col2:
            st.text_input(label='Default Demo dataset',
                          disabled=True,
                          key='default_demo_dataset',
                          value='data/demo-dataset.jsonl')

        start_btn = st.button('Start to split dataset by language ',
                              use_container_width=True)

        with st.expander('Splitted Results', expanded=True):

            target_dir = 'outputs'
            if start_btn:
                with st.spinner('Wait for splitting...'):
                    shutil.rmtree(target_dir, ignore_errors=True)
                    split_dataset(dataset_file, target_dir)

            splitted_datasets = Path(target_dir).glob('*.jsonl')
            datasets_dict = {
                ds_file.stem: ds_file
                for ds_file in splitted_datasets
            }
            count = len(datasets_dict)
            if count > 1:
                st.header(f'There are :red[{len(datasets_dict)}] languages:\
                         {[ lang for lang in datasets_dict.keys()]}')
            elif count == 1:
                st.header(f'There is :red[{len(datasets_dict)}] language: \
                        {[ lang for lang in datasets_dict.keys()]}')

            for k, v in datasets_dict.items():
                df = pd.read_json(str(v), lines=True)
                count = len(df.index)
                be = 'is' if count <= 1 else 'are'
                samples = 'sample ' if count <= 1 else 'samples'
                st.subheader(
                    f'There {be} :red[ {count} ] {samples} in :red[{k}]')
                st.dataframe(df, use_container_width=True)
                st.download_button(f'Download {k} as JSONL',
                                   data=convert_to_jsonl(df),
                                   file_name=v.name)

            shutil.rmtree(target_dir, ignore_errors=True)

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.split()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
