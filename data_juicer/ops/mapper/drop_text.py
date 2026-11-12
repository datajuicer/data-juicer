"""Remove an extraction mapper's source text from rows and datasets."""

from collections.abc import Mapping, MutableMapping
from copy import deepcopy

from datasets import LargeList, List


def read_text_from_sample(sample, text_key):
    # Dataset initialization imports operators, so defer this dependency.
    from data_juicer.core.data.dj_dataset import nested_query

    # LazyBatch is a Mapping, but nested_query descends through dictionaries.
    return nested_query(dict(sample) if isinstance(sample, Mapping) else sample, text_key)


def _remove_text(value, text_key, source):
    if isinstance(value, (List, LargeList)):
        return _remove_text(value.feature, text_key, source.feature)
    if isinstance(value, list):
        removed = False
        for item in value:
            if _remove_text(item, text_key, item):
                removed = True
        return removed
    if not isinstance(value, MutableMapping) or not isinstance(source, Mapping):
        return False
    if text_key in source:
        if text_key in value:
            del value[text_key]
            return True
        return False
    key, separator, rest = text_key.partition(".")
    if separator and key in source and key in value:
        return _remove_text(value[key], rest, source[key])
    return False


def remove_text_from_sample(sample, text_key):
    _remove_text(sample, text_key, sample)
    return sample


def remove_text_from_dataset(dataset, text_key, source_features=None):
    if source_features is None:
        source_features = dataset.features
    if text_key in source_features:
        return dataset.remove_columns([text_key]) if text_key in dataset.features else dataset

    features = deepcopy(dataset.features)
    if not _remove_text(features, text_key, source_features):
        return dataset

    column, _, rest = text_key.partition(".")

    def remove_nested_field(row):
        value = deepcopy(row[column])
        remove_text_from_sample(value, rest)
        return {column: value}

    # HF map would otherwise merge the old struct schema and restore deleted fields.
    return dataset.map(remove_nested_field, remove_columns=[column], features=features)
