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
