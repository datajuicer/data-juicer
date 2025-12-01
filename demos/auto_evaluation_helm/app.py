import os
import re

import streamlit as st


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
            '# <div align="center"> Data-Juicer </div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div align = "center"> A One-Stop Data Processing System for \
                Large Language Models, \
                see more details in our <a href={readme_link}>Github</a></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def visualize():
        Visualize.setup()


def main():

    def make_image(line):
        pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.findall(pattern, line)
        st.image(matches[0][1], output_format='png', use_column_width=True)

    Visualize.visualize()
    buffer = []
    with open('README_ZH.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'imgs/' in line:
                st.markdown('\n'.join(buffer))
                make_image(line)
                buffer.clear()
            else:
                buffer.append(line)
    st.markdown('\n'.join(buffer))
    # hello()


if __name__ == '__main__':
    main()
