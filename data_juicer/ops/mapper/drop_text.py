"""Remove an extraction mapper's source text from rows and datasets."""

from collections.abc import Mapping, MutableMapping
from copy import deepcopy


def _text_path(keys, text_key):
    """Resolve a key using the same literal-first precedence as nested_query."""
    remaining = text_key
    path = []
    current = keys
    while remaining:
        if not isinstance(current, Mapping):
            return None
        if remaining in current:
            path.append(remaining)
            return path
        parts = remaining.split(".")
        for index in range(1, len(parts)):
            prefix = ".".join(parts[:index])
            if prefix in current and isinstance(current[prefix], Mapping):
                path.append(prefix)
                current = current[prefix]
                remaining = ".".join(parts[index:])
                break
        else:
            return None
    return path


def remove_text_from_sample(sample, text_key):
    path = _text_path(sample, text_key)
    if path is None:
        # Batched mappers receive columns as lists (including lists of structs).
        # Apply the remaining path to each struct, preserving sibling fields.
        for prefix, rest in _list_prefix(sample, text_key):
            for value in sample[prefix]:
                if isinstance(value, MutableMapping):
                    remove_text_from_sample(value, rest)
            return sample
        return sample
    parent = sample
    for part in path[:-1]:
        parent = parent[part]
    if isinstance(parent, MutableMapping):
        parent.pop(path[-1], None)
    return sample


def read_text_from_sample(sample, text_key):
    if isinstance(sample, list):
        return [read_text_from_sample(value, text_key) for value in sample]
    path = _text_path(sample, text_key)
    if path is None:
        for prefix, rest in _list_prefix(sample, text_key):
            return read_text_from_sample(sample[prefix], rest)
        raise KeyError(text_key)
    value = sample
    for part in path:
        value = value[part]
    return value


def _list_prefix(sample, text_key):
    if not isinstance(sample, Mapping):
        return
    parts = text_key.split(".")
    for index in range(1, len(parts)):
        prefix = ".".join(parts[:index])
        if prefix in sample and isinstance(sample[prefix], list):
            yield prefix, ".".join(parts[index:])


def remove_text_from_dataset(dataset, text_key):
    path = _text_path(dataset.features, text_key)
    if path is None:
        for prefix, rest in _list_feature_prefix(dataset.features, text_key):
            features = deepcopy(dataset.features)
            list_feature = features[prefix]
            nested_path = _text_path(list_feature.feature, rest)
            if nested_path is None:
                return dataset
            parent = list_feature.feature
            for part in nested_path[:-1]:
                parent = parent[part]
            del parent[nested_path[-1]]

            def remove_list_field(row):
                value = deepcopy(row[prefix])
                if value is not None:
                    for item in value:
                        if isinstance(item, MutableMapping):
                            remove_text_from_sample(item, rest)
                return {prefix: value}

            return dataset.map(remove_list_field, remove_columns=[prefix], features=features)
        return dataset
    if len(path) == 1:
        return dataset.remove_columns([path[0]])

    column = path[0]
    features = deepcopy(dataset.features)
    parent = features[column]
    for part in path[1:-1]:
        parent = parent[part]
    del parent[path[-1]]

    def remove_nested_field(row):
        value = deepcopy(row[column])
        remove_text_from_sample(value, ".".join(path[1:]))
        return {column: value}

    # Remove the original struct before returning the shortened one. Otherwise
    # datasets.map keeps the original nested schema and restores the text as None.
    return dataset.map(remove_nested_field, remove_columns=[column], features=features)


def _list_feature_prefix(features, text_key):
    from datasets import List

    parts = text_key.split(".")
    for index in range(1, len(parts)):
        prefix = ".".join(parts[:index])
        if prefix in features and isinstance(features[prefix], List):
            yield prefix, ".".join(parts[index:])
