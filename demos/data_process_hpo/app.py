import streamlit as st


class Visualize:

    @staticmethod
    def setup():
        st.set_page_config(
            page_title='Data-Juicer',
            page_icon=':smile',
            #layout='wide',
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
    def visualize():
        Visualize.setup()


def main():

    def hello():

        st.image('imgs/data-juicer.png',
                 output_format='png',
                 use_column_width=True)
        demo = 'The demo is coming soon😊'
        st.markdown(
            f'<div align = "center"> <font size = "50"> {demo} \
            </font> </div>',
            unsafe_allow_html=True,
        )

    Visualize.visualize()
    hello()


if __name__ == '__main__':
    main()
