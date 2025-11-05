# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

import copy
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from loguru import logger

from data_juicer.analysis.diversity_analysis import DiversityAnalysis, get_diversity
from data_juicer.config import init_configs
from data_juicer.core import Analyzer, DefaultExecutor
from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.logger_utils import get_log_file_path
from data_juicer.utils.model_utils import MODEL_ZOO, prepare_model


@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf_8_sig")


@st.cache_data
def convert_to_jsonl(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_json(orient="records", lines=True, force_ascii=False).encode("utf_8_sig")


@st.cache_data
def get_diversity_model(lang):
    model_key = prepare_model("spacy", lang=lang)
    diversity_model = MODEL_ZOO.get(model_key)
    return diversity_model


@st.cache_data
def postproc_diversity(dataframe, **kwargs):
    df = get_diversity(dataframe, **kwargs)
    return df


def read_log_file():
    log_f_path = get_log_file_path()
    if log_f_path is None or not os.path.exists(log_f_path):
        return ""
    sys.stdout.flush()
    with open(log_f_path, "r") as f:
        return f.read()


def pretty_out(d):
    res = ""
    process = ""
    op_names = set(OPERATORS.modules.keys())
    for key, value in d.items():
        if key == "process":
            process = yaml.dump(value, allow_unicode=True, default_flow_style=False)
        elif key == "config" or key.split(".")[0] in op_names:
            continue
        else:
            res += f"{key}:\n \t {value}\n"
    res += "process:\n" + "\n".join(["\t" + line for line in process.splitlines()])

    return res


def parse_cfg():
    cfg_file = st.session_state.input_cfg_file
    cfg_cmd = st.session_state.input_cfg_cmd

    cfg_f_name = "null"
    del_cfg_file = False
    if cfg_file is not None:
        cfg_f_name = cfg_file.name
        file_contents = cfg_file.getvalue()
        with open(cfg_f_name, "wb") as f:
            f.write(file_contents)
        cfg_cmd = f"--config {cfg_f_name}"
        del_cfg_file = True

    args_in_cmd = cfg_cmd.split()

    if len(args_in_cmd) >= 2 and args_in_cmd[0] == "--config":
        cfg_f_name = args_in_cmd[1]
    else:
        st.warning("Please specify a config command or upload a config file.")
        st.stop()

    if not os.path.exists(cfg_f_name):
        st.warning("do not parse" f"config file does not exist with cfg_f_name={cfg_f_name}")
        st.stop()

    with open(cfg_f_name, "r") as cfg_f:
        specified_cfg = yaml.safe_load(cfg_f)

    try:
        parsed_cfg = init_configs(args=args_in_cmd)
        st.session_state.cfg = parsed_cfg
        if isinstance(parsed_cfg.text_keys, list):
            text_key = parsed_cfg.text_keys[0]
        else:
            text_key = parsed_cfg.text_keys
        st.session_state.text_key = text_key
        if del_cfg_file:
            os.remove(cfg_f_name)
        return pretty_out(parsed_cfg), pretty_out(specified_cfg), parsed_cfg
    except Exception as e:
        return str(e), pretty_out(specified_cfg), None


def analyze_and_show_res():
    images_ori = []
    cfg = st.session_state.get("cfg", parse_cfg()[2])
    if cfg is None:
        raise ValueError("you have not specify valid cfg")
    # force generating separate figures
    cfg["save_stats_in_one_file"] = True

    logger.info("=========Stage 1: analyze original data=========")
    analyzer = Analyzer(cfg)
    dataset = analyzer.run()

    overall_file = os.path.join(analyzer.analysis_path, "overall.csv")
    analysis_res_ori = pd.DataFrame()
    if os.path.exists(overall_file):
        analysis_res_ori = pd.read_csv(overall_file)

    if os.path.exists(analyzer.analysis_path):
        for f_path in os.listdir(analyzer.analysis_path):
            if ".png" in f_path and "all-stats" in f_path:
                images_ori.append(os.path.join(analyzer.analysis_path, f_path))

    st.session_state.dataset = dataset
    st.session_state.original_overall = analysis_res_ori
    st.session_state.original_imgs = images_ori


def process_and_show_res():
    images_processed = []
    cfg = st.session_state.get("cfg", parse_cfg()[2])
    if cfg is None:
        raise ValueError("you have not specify valid cfg")
    # force generating separate figures
    cfg["save_stats_in_one_file"] = True
    logger.info("=========Stage 2: process original data=========")
    executor = DefaultExecutor(cfg)
    dataset = executor.run()

    logger.info("=========Stage 3: analyze the processed data==========")
    analysis_res_processed = pd.DataFrame()
    try:
        cfg_for_processed_data = copy.deepcopy(cfg)
        cfg_for_processed_data.dataset_path = cfg.export_path

        cfg_for_processed_data.export_path = os.path.dirname(cfg.export_path) + "_processed/data.jsonl"
        analyzer = Analyzer(cfg_for_processed_data)
        analyzer.analysis_path = os.path.dirname(cfg_for_processed_data.export_path) + "/analysis"
        analyzer.run()

        overall_file = os.path.join(analyzer.analysis_path, "overall.csv")
        if os.path.exists(overall_file):
            analysis_res_processed = pd.read_csv(overall_file)

        if os.path.exists(analyzer.analysis_path):
            for f_path in os.listdir(analyzer.analysis_path):
                if ".png" in f_path and "all-stats" in f_path:
                    images_processed.append(os.path.join(analyzer.analysis_path, f_path))
    except Exception as e:
        st.warning(f"Something error with {str(e)}")

    logger.info("=========Stage 4: Render the analysis results==========")
    st.session_state.dataset = dataset
    st.session_state.processed_overall = analysis_res_processed
    st.session_state.processed_imgs = images_processed


def get_min_max_step(data):
    max_value = np.max(data)
    if max_value > 2.0:
        min_value = 0
        max_value = int(max_value + 1)
        step = 1
    else:
        min_value = 0.0
        max_value = max(1.0, max_value)
        step = 0.01
    return min_value, max_value, step


op_stats_dict = {
    "alphanumeric_filter": [StatsKeys.alpha_token_ratio, StatsKeys.alnum_ratio],
    "average_line_length_filter": [StatsKeys.avg_line_length],
    "character_repetition_filter": [StatsKeys.char_rep_ratio],
    "flagged_words_filter": [StatsKeys.flagged_words_ratio],
    "language_id_score_filter": [StatsKeys.lang, StatsKeys.lang_score],
    "maximum_line_length_filter": [StatsKeys.max_line_length],
    "perplexity_filter": [StatsKeys.perplexity],
    "special_characters_filter": [StatsKeys.special_char_ratio],
    "stopwords_filter": [StatsKeys.stopwords_ratio],
    "text_length_filter": [StatsKeys.text_len],
    "token_num_filter": [StatsKeys.num_token],
    "words_num_filter": [StatsKeys.num_words],
    "word_repetition_filter": [StatsKeys.word_rep_ratio],
}


class Visualize:
    @staticmethod
    def filter_dataset(dataset):
        if Fields.stats not in dataset.features:
            return
        text_key = st.session_state.get("text_key", "text")
        text = dataset[text_key]
        stats = pd.DataFrame(dataset[Fields.stats])
        stats[text_key] = text

        non_num_list = [StatsKeys.lang]
        min_cutoff_list = [
            StatsKeys.lang_score,
            StatsKeys.stopwords_ratio,
        ]
        max_cutoff_list = [
            StatsKeys.flagged_words_ratio,
            StatsKeys.perplexity,
        ]
        mask_list = [text_key]

        cfg = st.session_state.get("cfg", None)
        if cfg is None:
            return

        def set_sliders(total_stats, ordered):
            stats = copy.deepcopy(total_stats)
            conds = list()
            index = 1
            for op_cfg in cfg.process:
                op_name = list(op_cfg.keys())[0]
                op_stats = op_stats_dict.get(op_name, [])

                cutoff_ratio = None

                with st.sidebar.expander(f"{index} {op_name}"):
                    for column_name in op_stats:
                        if column_name not in stats:
                            continue
                        data = stats[column_name]

                        if column_name in non_num_list:
                            options = ["all"] + list(set(data))
                            label = f"Which {column_name} would \
                                     you like to keep?"

                            selected = st.selectbox(
                                label=label,
                                options=options,
                            )
                            if selected == "all":
                                cond = [True] * len(data)
                            else:
                                cond = data == selected
                            Visualize.display_discarded_ratio(cond, column_name)

                        elif column_name in min_cutoff_list:
                            label = f"If the {column_name} of a document  \
                                    is lower than this number,  \
                                    the document is removed."

                            low, high, step = get_min_max_step(data)

                            cutoff_ratio = st.slider(label, low, high, low, step=step)
                            cond = data >= cutoff_ratio
                            Visualize.display_discarded_ratio(cond, column_name)

                        elif column_name in max_cutoff_list:
                            label = f"If the {column_name} of a document  \
                                    is higher than this number,  \
                                    the document is removed."

                            low, high, step = get_min_max_step(data)
                            cutoff_ratio = st.slider(label, low, high, high, step=step)
                            cond = data <= cutoff_ratio

                            Visualize.display_discarded_ratio(cond, column_name)
                        elif column_name not in mask_list:
                            # lower
                            label = f"If the {column_name} of a document  \
                                    is lower than this number,  \
                                    the document is removed."

                            low, high, step = get_min_max_step(data)

                            cutoff_ratio_l = st.slider(label, low, high, low, step=step)
                            cond_l = data >= cutoff_ratio_l

                            Visualize.display_discarded_ratio(cond_l, column_name)

                            # higher
                            label = f"If the {column_name} of a document  \
                                    is higher than this number,  \
                                    the document is removed."

                            cutoff_ratio_h = st.slider(label, low, high, high, step=step)

                            cond_h = data <= cutoff_ratio_h
                            Visualize.display_discarded_ratio(cond_h, column_name)
                            cond = [low & high for low, high in zip(cond_l, cond_h)]

                            cutoff_ratio = (cutoff_ratio_l, cutoff_ratio_h)

                        if column_name not in mask_list:
                            Visualize.draw_hist(data, cutoff_ratio)
                            conds.append({(" ".join([str(index), op_name]), column_name): cond})

                        if ordered:
                            stats = stats.loc[cond]
                    index += 1
            return conds, stats

        st.sidebar.subheader("Parameters of filter ops")
        ordered = st.sidebar.checkbox("Process by op order")
        conds, filtered_stats = set_sliders(stats, ordered)

        st.subheader("How many samples do you want to show?")
        show_num = st.number_input(label="How many samples do you want to show?", value=5, label_visibility="hidden")
        if ordered:
            all_conds = [True if i in filtered_stats.index else False for i in range(len(stats))]
        else:
            all_conds = np.all([list(cond.values())[0] for cond in conds], axis=0)
        ds = pd.DataFrame(dataset)
        Visualize.display_dataset(ds, all_conds, show_num, "Retained samples", "docs")
        st.download_button(
            "Download Retained data as JSONL", data=convert_to_jsonl(ds.loc[all_conds]), file_name="retained.jsonl"
        )
        Visualize.display_dataset(ds, np.invert(all_conds), show_num, "Discarded samples", "docs")
        st.download_button(
            "Download Discarded data as JSONL",
            data=convert_to_jsonl(ds.loc[np.invert(all_conds)]),
            file_name="discarded.jsonl",
        )
        display_discarded_details = st.checkbox("Display discarded documents by filter details")

        show_stats = copy.deepcopy(stats)
        bar_labels = []
        bar_sizes = []
        for item in conds:
            for op_key, cond in item.items():
                op_name, column_name = op_key
                if column_name not in mask_list:
                    sub_stats = show_stats[[column_name, text_key]]
                    if display_discarded_details:
                        Visualize.display_dataset(
                            sub_stats,
                            np.invert(cond) if len(cond) > 0 else [],
                            show_num,
                            # f'Discarded documents for the filter on \
                            f"{op_name} {column_name} filtered ",
                            "docs",
                        )
                    before_filtered_num = len(show_stats.index)
                    if ordered:
                        show_stats = show_stats.loc[cond]
                        retained = np.sum(1 * cond)
                        filtered = before_filtered_num - len(show_stats.index)
                    else:
                        retained = np.sum(1 * cond)
                        filtered = before_filtered_num - retained

                    bar_sizes.append(retained)
                    bar_sizes.append(filtered)
                    bar_labels.append(f"{op_name}\n{column_name}")

        bar_title = "Effect of Filter OPs"
        Visualize.draw_stack_bar(bar_sizes, bar_labels, len(stats.index), bar_title)

    @staticmethod
    def diversity():
        with st.expander("Diversity for CFT dataset", expanded=False):
            dataset = st.session_state.get("dataset", None)
            cfg = st.session_state.get("cfg", parse_cfg()[2])
            text_key = st.session_state.get("text_key", "text")
            if dataset:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    label = "Which language of your dataset"
                    options = ["en", "zh"]
                    lang_select = st.selectbox(
                        label=label,
                        options=options,
                    )
                with col2:
                    top_k_verbs = st.number_input("Set the top_k nums of verbs", value=20)
                with col3:
                    top_k_nouns = st.number_input("Set the top_k nums of nouns", value=4)
                with col4:
                    threshold = st.slider("Count threshold", min_value=0, value=0, max_value=100, step=1)

                diversity_btn = st.button("Analyze_diversity", use_container_width=True)
                output_path = os.path.join(os.path.dirname(cfg.export_path), "analysis")
                raw_df = None
                if diversity_btn:
                    try:
                        diversity_analysis = DiversityAnalysis(dataset, output_path)
                        with st.spinner("Wait for analyze diversity..."):
                            raw_df = diversity_analysis.compute(
                                lang_or_model=get_diversity_model(lang_select), column_name=text_key
                            )

                        st.session_state[f"diversity{lang_select}"] = raw_df

                    except Exception as e:
                        st.warning(f"Error {str(e)} in {lang_select}")
                else:
                    raw_df = st.session_state.get(f"diversity{lang_select}", None)

                if raw_df is not None:
                    df = postproc_diversity(raw_df, top_k_verbs=top_k_verbs, top_k_nouns=top_k_nouns)
                    df = df[df["count"] >= threshold]
                    Visualize.draw_sunburst(df, path=["verb", "noun"], values="count")

                    st.download_button(
                        label="Download diversity data as CSV",
                        data=convert_to_csv(df),
                        file_name="diversity.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("Please analyze original data first")

    @staticmethod
    def draw_sunburst(df, path, values):
        fig = px.sunburst(df, path=path, values=values)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), font_family="Times New Roman", font=dict(size=40))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def draw_stack_bar(bar_sizes, bar_labels, total_num, title=""):
        filtered_size = [k / total_num * 100 for i, k in enumerate(bar_sizes[::-1]) if i % 2 == 0]
        retain_size = [k / total_num * 100 for i, k in enumerate(bar_sizes[::-1]) if i % 2 != 0]
        plt.clf()
        plt.title(title)
        bar_labels = bar_labels[::-1]
        # retained
        r_bars = plt.barh(bar_labels, retain_size, label="Retained", height=0.5, color="limegreen")

        # filtered
        f_bars = plt.barh(bar_labels, filtered_size, label="Filtered", left=retain_size, height=0.5, color="orangered")

        for idx, bar in enumerate(r_bars):
            width = bar.get_width()
            plt.text(
                bar.get_x() + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{retain_size[idx]:.2f}%",
                ha="center",
                va="center",
            )

        for idx, bar in enumerate(f_bars):
            width = bar.get_width()
            plt.text(
                bar.get_x() + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{filtered_size[idx]:.2f}%",
                ha="center",
                va="center",
            )

        plt.legend()
        plt.gcf()
        st.pyplot(plt, use_container_width=True)

    @staticmethod
    def draw_pie(bar_labels, big_sizes, small_labels, bar_sizes):
        plt.clf()

        # filter op circle
        plt.pie(big_sizes, labels=bar_labels, startangle=90, frame=True)
        # retained and filtered circle
        plt.pie(bar_sizes, labels=small_labels, radius=0.7, rotatelabels=True, startangle=90, labeldistance=0.7)
        centre_circle = plt.Circle((0, 0), 0.4, color="white", linewidth=0)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        plt.axis("equal")
        plt.tight_layout()
        st.pyplot(plt, use_container_width=True)

    @staticmethod
    def display_discarded_ratio(cond, key):
        if len(cond) > 0:
            st.caption(
                f":red[{(len(cond) - np.sum(1*cond)) / len(cond) * 100:.2f}%] \
                of the total (:red[{len(cond)}]) is discarded with {key}."
            )
        else:
            st.caption(
                f":red[{0:.2f}%] \
                of the total (:red[0]) is discarded with {key}."
            )

    @staticmethod
    def display_dataset(dataframe, cond, show_num, desp, type, all=True):
        examples = dataframe.loc[cond]
        if all or len(examples) > 0:
            st.subheader(
                f"{desp}: :red[{len(examples)}] of "
                f"{len(dataframe.index)} {type} "
                f"(:red[{len(examples)/len(dataframe.index) * 100:.2f}%])"
            )

            # st.markdown('Click on a column to sort by it, \
            #    place the cursor on the text to display it.')
            st.dataframe(examples[:show_num], use_container_width=True)

    @staticmethod
    def draw_hist(data, cutoff=None):
        fig, ax = plt.subplots()
        data_num = len(data)
        if data_num >= 100:
            rec_bins = int(math.sqrt(len(data)))
        else:
            rec_bins = 50

        if data_num > 0:
            ax.hist(data, bins=rec_bins, density=True)
        if hasattr(data, "name"):
            ax.set_title(data.name)

        if isinstance(cutoff, (float, int)):
            ax.axvline(x=cutoff, color="r", linestyle="dashed")
        elif isinstance(cutoff, tuple) and len(cutoff) == 2:
            ax.axvline(x=cutoff[0], color="r", linestyle="dashed")
            ax.axvline(x=cutoff[1], color="r", linestyle="dashed")
        st.pyplot(fig)

    @staticmethod
    def setup():
        st.set_page_config(
            page_title="Data-Juicer",
            page_icon=":smile",
            layout="wide",
            # initial_sidebar_state="expanded",
        )

        readme_link = "https://github.com/datajuicer/data-juicer"
        st.markdown(
            '<div align = "center"> <font size = "70"> Data-Juicer \
            </font> </div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div align = "center"> A One-Stop Data Processing System for \
                Large Language Models, \
                see more details in our <a href={readme_link}>page</a></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def parser():
        with st.expander("Configuration", expanded=True):
            st.markdown(
                "Please specify the cfg via "
                "(i) specifying the cfg file path with commands or "
                "(ii) uploading the cfg file."
            )

            col1, col2 = st.columns(2)
            with col1:
                example_cfg_f = os.path.abspath(os.path.join(os.path.dirname(__file__), "./configs/demo/process.yaml"))
                st.text_area(label="(i) Input Cfg Commands", key="input_cfg_cmd", value=f"--config {example_cfg_f}")
                example_my_cmd = (
                    "--dataset_path "
                    "./demos/data/demo-dataset.jsonl "
                    "--export_path "
                    "./outputs/demo/demo-processed.jsonl"
                )

                st.text_area(
                    label="cmd example. (the cmd-args will override " "yaml-file-args)",
                    disabled=True,
                    value=f"--config {example_cfg_f} {example_my_cmd}",
                )

            with col2:
                st.file_uploader(label="(ii) Input Cfg File", key="input_cfg_file", type=["yaml"])

            btn_show_cfg = st.button("1. Parse Cfg", use_container_width=True)
            if btn_show_cfg:
                text1, text2, cfg = parse_cfg()
                st.session_state.cfg_text1 = text1
                st.session_state.cfg_text2 = text2

            else:
                text1 = st.session_state.get("cfg_text1", "")
                text2 = st.session_state.get("cfg_text2", "")

            col3, col4 = st.columns(2)
            with col3:
                st.text_area(label="Parsed Cfg (in memory)", value=text1)
            with col4:
                st.text_area(label="Specified Cfg (in yaml file)", value=text2)

    @staticmethod
    def analyze_process():
        start_btn = st.button("2. Start to analyze original data (per filter op)", use_container_width=True)
        start_btn_process = st.button("3. Start to process data", use_container_width=True)

        # with st.expander('Log', expanded=False):
        #    logs = st.Textbox(show_label=False)
        #    demo.load(read_log_file, inputs=None, outputs=logs, every=1)

        with st.expander("Data Analysis Results", expanded=False):
            if start_btn:
                with st.spinner("Wait for analyze..."):
                    analyze_and_show_res()

            if start_btn_process:
                with st.spinner("Wait for process..."):
                    process_and_show_res()

            original_overall = st.session_state.get("original_overall", None)
            original_imgs = st.session_state.get("original_imgs", [])
            processed_overall = st.session_state.get("processed_overall", None)
            processed_imgs = st.session_state.get("processed_imgs", [])

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Original Data")
                st.dataframe(original_overall, use_container_width=True)
                for img in original_imgs:
                    st.image(img, output_format="png")

            with col2:
                st.caption("Processed Data")
                st.dataframe(processed_overall, use_container_width=True)
                for img in processed_imgs:
                    st.image(img, output_format="png")

    @staticmethod
    def filter():
        with st.expander("Effect of Filter OPs", expanded=False):
            dataset = st.session_state.get("dataset", None)
            if dataset:
                Visualize.filter_dataset(dataset)
            else:
                st.warning("Please analyze original data first")

    @staticmethod
    def auxiliary():
        st.markdown("[WIP] Auxiliary Models on Processed Data")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Quality Scorer", expanded=False):
                wiki_socre_btn = st.button("Run Wiki-score classifier", use_container_width=True)

                if wiki_socre_btn:
                    st.warning("No support for now")

                wikibook_score_btn = st.button("Run WikiBook-score classifier", use_container_width=True)
                if wikibook_score_btn:
                    st.warning("No support for now")

        with col2:
            with st.expander("[WIP] Proxy LM Models Training", expanded=False):
                st.file_uploader(label="LM Training Cfg File", type=["yaml"])
                st.button("Train proxy model")
                st.markdown(
                    "[Training Monitoring](http://" "8.130.26.137:8083/dail/" "llama-re-2nd?workspace=user-dail)"
                )

    @staticmethod
    def visualize():
        Visualize.setup()
        Visualize.parser()
        Visualize.analyze_process()
        Visualize.filter()
        Visualize.diversity()
        Visualize.auxiliary()


def main():
    Visualize.visualize()


if __name__ == "__main__":
    main()
