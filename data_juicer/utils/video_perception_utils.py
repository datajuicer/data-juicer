"""Frame grouping shared by video perception mappers."""

import os
from hashlib import sha256


def get_video_frame_groups(sample, frame_field, video_key, extractor):
    """Return ordered frame paths per video, using the extractor's actual output."""
    if frame_field in sample:
        frames = sample[frame_field]
    elif sample.get(video_key):
        extracted = extractor.process_single({video_key: list(sample[video_key])})
        frames = extracted[frame_field]
    else:
        return []
    if not frames:
        return []
    if isinstance(frames[0], (str, os.PathLike)):
        frames = [frames]
    groups = []
    for group in frames:
        if not isinstance(group, (list, tuple)) or any(not isinstance(p, (str, os.PathLike)) for p in group):
            raise ValueError("Video frames must be paths grouped by video, or one flat list of paths.")
        groups.append([os.fspath(p) for p in group])
    return groups


def get_video_visualization_dir(root, operator, paths, video_index):
    """Keep artifacts from different input sequences and operators separate."""
    digest = sha256("\0".join(paths).encode()).hexdigest()[:16]
    directory = os.path.join(root, operator, f"{video_index}_{digest}")
    os.makedirs(directory, exist_ok=True)
    return directory


def get_video_output_features(input_features, tag_field_name, output_feature):
    """Keep video result types stable when an HF writer sees empty samples first."""
    if input_features is None:
        return None
    from copy import deepcopy

    from datasets import Features, List

    from data_juicer.utils.constant import Fields

    features = deepcopy(input_features)
    meta = features.get(Fields.meta, {})
    if not isinstance(meta, dict):
        return None
    existing = meta.get(tag_field_name)
    if isinstance(existing, dict) and isinstance(output_feature, dict):
        output_feature = {**existing, **output_feature}
    elif isinstance(existing, List) and isinstance(existing.feature, dict) and isinstance(output_feature, List):
        output_feature = List({**existing.feature, **output_feature.feature})
    features[Fields.meta] = {**meta, tag_field_name: output_feature}
    return Features(features)


def nested_video_feature(dtype, dimensions=1):
    """Describe a variable-size numeric/string result without fixed frame counts."""
    from datasets import List, Value

    feature = Value(dtype)
    for _ in range(dimensions):
        feature = List(feature)
    return feature
