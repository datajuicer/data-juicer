import os

import pandas as pd
import spacy
from loguru import logger

from data_juicer.utils.model_utils import get_model, prepare_model


# Modify from self_instruct, please refer to
# https://github.com/yizhongw/self-instruct/blob/main/self_instruct/instruction_visualize.ipynb
def find_root_verb_and_its_dobj(tree_root):
    """
    Find the verb and its object closest to the root.

    :param tree_root: the root of lexical tree
    :return: valid verb and its object.
    """
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return (
                    tree_root.lemma_ if len(tree_root.lemma_) else tree_root.text,
                    child.lemma_ if len(child.lemma_) else child.text,
                )
        return tree_root.lemma_ if len(tree_root.lemma_) else tree_root.text, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None


# Modify from self_instruct, please refer to
# https://github.com/yizhongw/self-instruct/blob/main/self_instruct/instruction_visualize.ipynb
def find_root_verb_and_its_dobj_in_string(nlp, s, first_sent=True):
    """
    Find the verb and its object closest to the root of lexical tree of input
    string.

    :param nlp: the diversity model to analyze the diversity strings
    :param s: the string to be analyzed
    :param first_sent: whether to analyze the first sentence in the
        input string only. If it's true, return the analysis result of
        the first sentence no matter it's valid or not. If it's false,
        return the first valid result over all sentences
    :return: valid verb and its object of this string
    """
    doc = nlp(s)
    for sent in doc.sents:
        verb, noun = find_root_verb_and_its_dobj(sent.root)
        if first_sent or (verb is not None and noun is not None):
            return verb, noun
    return None, None


def get_diversity(dataset, top_k_verbs=20, top_k_nouns=4, **kwargs):
    """
    Given the lexical tree analysis result, return the diversity results.

    :param dataset: lexical tree analysis result
    :param top_k_verbs: only keep the top_k_verbs largest verb groups
    :param top_k_nouns: only keep the top_k_nouns largest noun groups
        for each verb group
    :param kwargs: extra args
    :return: the diversity results
    """
    phrases = pd.DataFrame(dataset).dropna()
    logger.info(
        f"find valid verb-noun structure \
                {phrases.shape[0]} of {dataset.shape[0]}"
    )
    top_verbs = phrases.groupby(["verb"]).size().nlargest(top_k_verbs).reset_index()

    df = phrases[phrases["verb"].isin(top_verbs["verb"].tolist())]
    df = (
        df.groupby(["verb", "noun"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
    )

    df = (
        df.groupby("verb")
        .apply(lambda x: x.sort_values("count", ascending=False).head(top_k_nouns))
        .reset_index(drop=True)
    )
    return df


class DiversityAnalysis:
    """Apply diversity analysis for each sample and get an overall analysis
    result."""

    def __init__(self, dataset, output_path, lang_or_model="en"):
        """Initialization method :param dataset: the dataset to be analyzed
        :param output_path: path to store the analysis results :param
        lang_or_model: the diversity model or a specific language used to load
        the diversity model."""

        self.dataset = dataset
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.lang_or_model = lang_or_model

    def compute(self, lang_or_model=None, column_name="text"):
        """
        Apply lexical tree analysis on each sample.

        :param lang_or_model: the diversity model or a specific language
            used to load the diversity model
        :param column_name: the name of column to be analyzed
        :return: the analysis result.
        """
        # load diversity model
        lang_or_model = lang_or_model if lang_or_model else self.lang_or_model
        if isinstance(lang_or_model, str):
            model_key = prepare_model("spacy", lang=lang_or_model)
            diversity_model = get_model(model_key)
        else:
            diversity_model = lang_or_model

        assert isinstance(diversity_model, spacy.Language)

        def find_verb_noun(sample):
            try:
                verb, noun = find_root_verb_and_its_dobj_in_string(diversity_model, sample[column_name])
            except Exception as e:
                print(str(e))
                verb, noun = None, None
            return {"verb": verb, "noun": noun}

        dataset = self.dataset.map(find_verb_noun)
        return pd.DataFrame(dataset)

    def analyze(self, lang_or_model=None, column_name="text", postproc_func=get_diversity, **postproc_kwarg):
        """
        Apply diversity analysis on the whole dataset.

        :param lang_or_model: the diversity model or a specific language
            used to load the diversity model
        :param column_name: the name of column to be analyzed
        :param postproc_func: function to analyze diversity. In default,
            it's function get_diversity
        :param postproc_kwarg: arguments of the postproc_func
        :return:
        """
        # get the lexical tree analysis result
        raw_df = self.compute(lang_or_model=lang_or_model, column_name=column_name)
        # get the result of diversity analysis
        df = postproc_func(raw_df, **postproc_kwarg)

        # export to result report file
        df.to_csv(os.path.join(self.output_path, "diversity.csv"))
        df.to_markdown(os.path.join(self.output_path, "diversity.md"))

        return df
