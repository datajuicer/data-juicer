"""Read and remove source text fields in rows and datasets."""

from collections.abc import Mapping, MutableMapping
from copy import deepcopy

from datasets import LargeList, List


def read_text_from_sample(sample, text_key):
    # Dataset initialization imports operators, so defer this dependency.
    from data_juicer.core.data.dj_dataset import NestedQueryDict, nested_query

    # LazyBatch is a Mapping, but nested_query descends through dictionaries.
    row = dict(sample) if isinstance(sample, Mapping) else sample
    text = nested_query(row, text_key)
    if text is None and not isinstance(sample, NestedQueryDict) and not _has_text_path(row, text_key):
        raise KeyError(text_key)
    return text


def _has_text_path(row, text_key):
    if not isinstance(row, Mapping):
        return False
    if text_key in row:
        return True
    key, separator, rest = text_key.partition(".")
    if not separator or key not in row:
        return False
    parent = row[key]
    return parent is None or _has_text_path(parent, rest)


def _empty_struct(feature):
    while isinstance(feature, (List, LargeList)):
        feature = feature.feature
    if isinstance(feature, list):
        return bool(feature) and all(item is None or _empty_struct(item) for item in feature)
    return isinstance(feature, Mapping) and not feature


def _remove_text(value, text_key, source, *, prune_empty=False):
    if isinstance(value, (List, LargeList)):
        return _remove_text(value.feature, text_key, source.feature, prune_empty=prune_empty)
    if isinstance(value, list):
        removed = False
        for item in value:
            if _remove_text(item, text_key, item, prune_empty=prune_empty):
                removed = True
        return removed
    if not isinstance(value, MutableMapping) or not isinstance(source, Mapping):
        return False
    if text_key in source:
        if text_key in value:
            del value[text_key]
            return True
        return prune_empty and _empty_struct(value)
    key, separator, rest = text_key.partition(".")
    if separator and key in source and key in value:
        removed = _remove_text(value[key], rest, source[key], prune_empty=prune_empty)
        if removed and prune_empty and _empty_struct(value[key]):
            del value[key]
        return removed
    return False


def remove_text_from_sample(sample, text_key):
    _remove_text(sample, text_key, sample, prune_empty=True)
    return sample


def _project_to_features(value, features):
    if value is None:
        return None
    if isinstance(features, (List, LargeList)):
        return [_project_to_features(item, features.feature) for item in value]
    if isinstance(features, Mapping):
        return {key: _project_to_features(value[key], feature) for key, feature in features.items()}
    return value


def remove_text_from_dataset(dataset, text_key, source_features=None):
    if source_features is None:
        source_features = dataset.features
    if text_key in source_features:
        return dataset.remove_columns([text_key]) if text_key in dataset.features else dataset

    features = deepcopy(dataset.features)
    if not _remove_text(features, text_key, source_features, prune_empty=True):
        return dataset

    column, _, _ = text_key.partition(".")
    if column not in features:
        return dataset.remove_columns([column])

    def remove_nested_field(row):
        return {column: _project_to_features(row[column], features[column])}

    # HF map would otherwise merge the old struct schema and restore deleted fields.
    return dataset.map(remove_nested_field, remove_columns=[column], features=features)
