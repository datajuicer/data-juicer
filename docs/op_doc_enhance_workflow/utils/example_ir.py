# Unified example intermediate representation and media objects
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union


@dataclass
class MediaAsset:
    """Media asset representation for different file types"""

    kind: str  # Media type: 'image' | 'audio' | 'video' etc.
    src: Path  # Source path (absolute or relative, stored as-is, no copying)
    rel: str  # Relative path to md_dir (forward slashes)


@dataclass
class ExampleIR:
    """Intermediate representation for operator examples"""

    method: str  # Test method name
    op_code: str  # Operator initialization code
    input: Dict[str, Any] = field(
        default_factory=dict
    )  # Input data, e.g. {'text': '...'} or {'images': [MediaAsset,...]}
    output: Dict[str, Any] = field(default_factory=dict)  # Expected output data
    meta: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


def make_asset(src: Union[str, Path], md_dir: Path, kind: str) -> MediaAsset:
    """Create a MediaAsset with relative path calculation"""
    p = Path(src)
    rel = Path(os.path.relpath(str(p), str(md_dir))).as_posix()  # Convert to forward slashes
    return MediaAsset(kind=kind, src=p, rel=rel)
