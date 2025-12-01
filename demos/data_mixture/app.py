from pathlib import Path

import pandas as pd
import streamlit as st
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.config import get_default_cfg

if st.__version__ >= '1.23.0':
    data_editor = st.data_editor
else:
    data_editor = st.data_editor.experimental_data_editor


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf_8_sig')


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient='records', lines=True,
                      force_ascii=False).encode('utf_8_sig')


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
    def mix_dataset():

        data_files = list(Path('./data').glob('*jsonl'))

        data_files_dict = {file.stem: str(file) for file in data_files}
        col1, col2 = st.columns(2)
        all_selected = []
        with col1:
            col3, col4 = st.columns(2)
            with col3:
                st.subheader('Select datasets')
                options = sorted(list(data_files_dict.keys()))
                selected_ds = st.multiselect(label='datasets',
                                             options=options,
                                             label_visibility='hidden')
                for ds in selected_ds:
                    all_selected.append({'dataset': ds, 'weight': 1.0})
            with col4:
                st.subheader('Select sampling method')
                options = ['Random']
                st.selectbox(label='method',
                             options=options,
                             label_visibility='hidden')

            st.subheader('Set weight (0.0-1.0)')
            datasets = data_editor(all_selected, use_container_width=True)
            ds_names = [ds['dataset'] for ds in datasets]
            ds_files = [data_files_dict[ds['dataset']] for ds in datasets]
            weights = [ds['weight'] for ds in datasets]
        with col2:
            st.subheader('Show selected dataset details')
            display_select = st.checkbox('Display')
            if display_select:
                if len(datasets) > 0:
                    tabs = st.tabs(ds_names)
                    for tab, ds_file in zip(tabs, ds_files):
                        with tab:
                            st.write(pd.read_json(ds_file, lines=True))

        start_btn = st.button('Start to mix datasets',
                              use_container_width=True)
        if start_btn:
            if len(datasets) > 0:
                data_path = ' '.join([
                    ' '.join([str(weight), ds_file])
                    for ds_file, weight in zip(ds_files, weights)
                ])
                cfg = get_default_cfg()
                cfg.dataset_path = data_path
                dataset_builder = DatasetBuilder(cfg)
                df = pd.DataFrame(dataset_builder.load_dataset())

                st.session_state.dataset = df
            else:
                st.warning('Please select one dataset at least')

        dataset = st.session_state.get('dataset', pd.DataFrame())
        st.subheader('Mixed dataset')
        st.dataframe(dataset, use_container_width=True)
        st.download_button(label='Download mixed dataset as JSONL',
                           data=convert_to_jsonl(dataset),
                           file_name='mixed_dataset.jsonl')

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.mix_dataset()


def main():
    Visualize.visualize()


if __name__ == '__main__':
    main()
