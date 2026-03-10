#!/usr/bin/env python3
"""
Operator Searcher - A tool for filtering Data-Juicer operators by tags
"""
import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional

from docstring_parser import parse
from loguru import logger

from data_juicer.format.formatter import FORMATTERS
from data_juicer.ops import OPERATORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

op_type_list = ["aggregator", "deduplicator", "filter", "grouper", "mapper", "pipeline", "selector", "formatter"]


def get_source_path(cls):
    abs_path = Path(inspect.getfile(cls))
    # Get the project root directory

    try:
        # Computes a path relative to the project root directory
        return abs_path.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError("Class is not in the project root directory")


def find_test_by_searching_content(tests_dir, test_class_name):
    """Fallback: brute-force search for test files containing the test class name."""
    for file in tests_dir.rglob("test_*.py"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                if test_class_name in f.read():
                    return Path(file).relative_to(PROJECT_ROOT)
        except Exception:
            continue
    return None


# OP tag analysis functions
def analyze_modality_tag(code, op_prefix):
    """
    Analyze the modality tag for the given code content string. Should be one
    of the "Modality Tags" in `tagging_mappings.json`. It makes the choice by
    finding the usages of attributes `{modality}_key` and the prefix of the OP
    name. If there are multiple modality keys are used, the 'multimodal' tag
    will be returned instead.
    """
    tags = []
    if "self.text_key" in code or op_prefix == "text":
        tags.append("text")
    if "self.image_key" in code or op_prefix == "image":
        tags.append("image")
    if "self.audio_key" in code or op_prefix == "audio":
        tags.append("audio")
    if "self.video_key" in code or op_prefix == "video":
        tags.append("video")
    if len(tags) > 1:
        tags = ["multimodal"]
    return tags


def analyze_resource_tag(cls):
    """
    Analyze resource tags by reading the class attribute_accelerator. Should be one
    of the "Resource Tags" in `tagging_mappings.json`. It makes the choice
    according to their assigning statement to attribute `_accelerator`.
    """
    if "_accelerator" in cls.__dict__:
        accelerator_val = cls._accelerator
        if accelerator_val == "cuda":
            return ["gpu"]
        else:
            return ["cpu"]
    else:
        return []


def analyze_model_tags(cls):
    """
    Analyze the model tag for the given code content string. SHOULD be one of
    the "Model Tags" in `tagging_mappings.json`. It makes the choice by finding
    the `model_type` arg in `prepare_model` method invocation.
    """
    pattern = r"model_type=[\'|\"](.*?)[\'|\"]"
    code = inspect.getsource(cls)
    groups = re.findall(pattern, code)
    tags = []
    for group in groups:
        if group == "api":
            tags.append("api")
        elif group == "vllm":
            tags.append("vllm")
        elif group in {"huggingface", "diffusion", "simple_aesthetics", "video_blip"}:
            tags.append("hf")
    return tags


# def analyze_tag_from_code(content, op_name):
#     """
#     Analyze the tags for the OP from the given code.
#     """
#     tags = []
#     op_prefix = op_name.split('_')[0]

#     tags.extend(analyze_modality_tag(content, op_prefix))
#     tags.extend(analyze_resource_tag(content))
#     tags.extend(analyze_model_tags(content))
#     return tags


def analyze_tag_with_inheritance(op_cls, analyze_func, default_tags=[], other_parm=dict()):
    """
    Universal inheritance chain label analysis function
    """

    mro_classes = op_cls.__mro__[:3]
    for cls in mro_classes:
        try:
            current_tags = analyze_func(cls, **other_parm)
            if len(current_tags) > 0:
                return current_tags
        except (OSError, TypeError):
            continue

    return default_tags


def analyze_tag_from_cls(op_cls, op_name):
    """
    Analyze the tags for the OP from the given cls.
    """
    tags = []
    op_prefix = op_name.split("_")[0]

    content = inspect.getsource(op_cls)

    # Try to find from the inheritance chain
    resource_tags = analyze_tag_with_inheritance(op_cls, analyze_resource_tag, default_tags=["cpu"])
    model_tags = analyze_tag_with_inheritance(op_cls, analyze_model_tags)

    tags.extend(resource_tags)
    tags.extend(model_tags)
    tags.extend(analyze_modality_tag(content, op_prefix))
    return tags


def extract_param_docstring(docstring):
    """
    Extract parameter descriptions from __init__ method docstring.
    """
    param_docstring = ""
    if not docstring:
        return param_docstring
    param_docstring = ":param ".join(docstring.split(":param"))
    if ":param" not in param_docstring:
        return ""
    return param_docstring


class OPRecord:
    """A record class for storing operator metadata"""

    def __init__(self, name: str, op_cls: type, op_type: str = None):
        self.name = name
        self.type = op_type or op_cls.__module__.split(".")[2].lower()
        if self.type not in op_type_list:
            self.type = self._search_mro_for_type(op_cls)
        self.desc = op_cls.__doc__ or ""
        self.tags = analyze_tag_from_cls(op_cls, name)
        self.sig = inspect.signature(op_cls.__init__)
        self.param_desc = extract_param_docstring(op_cls.__init__.__doc__ or "")
        self.param_desc_map = self._parse_param_desc()
        self.source_path = str(get_source_path(op_cls))
        self.test_path = None

        test_path = f"tests/ops/{self.type}/test_{self.name}.py"
        if not (PROJECT_ROOT / test_path).exists():
            test_path = find_test_by_searching_content(PROJECT_ROOT / "tests", op_cls.__name__ + "Test") or test_path

        self.test_path = str(test_path)

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"OPRecord has no attribute: {item}")

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def _search_mro_for_type(self, op_cls: type) -> str:
        """Traverse the inheritance chain to find a valid base class name"""
        for base in op_cls.__mro__:
            base_name = base.__name__.lower()
            if base_name in op_type_list:
                return base_name

        return "unknown"

    def _parse_param_desc(self):
        """
        Parse parameter descriptions from docstring in ':param name: desc' format.
        Return a dict {param_name: description}.
        """
        docstring = parse(self.param_desc)
        return {p.arg_name: p.description.replace("\n", " ") for p in docstring.params}

    def to_dict(self):
        return {
            "type": self.type,
            "name": self.name,
            "desc": self.desc,
            "tags": self.tags,
            "sig": self.sig,
            "param_desc": self.param_desc,
            "param_desc_map": self.param_desc_map,
            "source_path": self.source_path,
            "test_path": self.test_path,
        }


class OPSearcher:
    """Operator search engine"""

    def __init__(self, specified_op_list: Optional[List[str]] = None, include_formatter: bool = False):
        self.all_ops = {}
        if specified_op_list:
            self.op_records = self._scan_specified_ops(specified_op_list)
        else:
            self.op_records = self._scan_all_ops(include_formatter)

    def _scan_specified_ops(self, specified_op_list: List[str]) -> List[OPRecord]:
        """Scan specified operators"""
        records = []
        op_type = None
        for op_name in specified_op_list:
            if op_name in FORMATTERS.modules:
                op_type = "formatter"
                op_cls = FORMATTERS.modules[op_name]
            else:
                op_cls = OPERATORS.modules[op_name]
            record = OPRecord(name=op_name, op_cls=op_cls, op_type=op_type)
            records.append(record)
            self.all_ops[op_name] = record
        return records

    def _scan_all_ops(self, include_formatter: bool = False) -> List[OPRecord]:
        """Scan all operators"""
        all_ops_list = list(OPERATORS.modules.keys())
        if include_formatter:
            all_ops_list.extend(FORMATTERS.modules.keys())
        return self._scan_specified_ops(all_ops_list)

    def search(
        self, tags: Optional[List[str]] = None, op_type: Optional[str] = None, match_all: bool = True
    ) -> List[Dict]:
        """
        Search operators by criteria
        :param tags: List of tags to match
        :param op_type: Operator type (mapper/filter/etc)
        :param match_all: True requires matching all tags, False matches any tag
        :return: List of matched operator records
        """
        results = []
        for record in self.op_records:
            # Filter by type
            if op_type and record.type != op_type:
                continue

            # Filter by tags
            if tags:
                tags = [tag.lower() for tag in tags]
                if match_all:
                    if not all(tag in record.tags for tag in tags):
                        continue
                else:
                    if not any(tag in record.tags for tag in tags):
                        continue

            results.append(record.to_dict())
        return results

    @property
    def records_map(self):
        logger.warning("records_map is deprecated, use all_ops instead")
        return self.all_ops


def main(tags, op_type):
    searcher = OPSearcher(include_formatter=True)

    results = searcher.search(tags=tags, op_type=op_type)

    print(f"\nFound {len(results)} operators:")
    for op in results:
        print(f"\n[{op['type'].upper()}] {op['name']}")
        print(f"Tags: {', '.join(op['tags'])}")
        print(f"Description: {op['desc']}")
        print(f"Parameters: {op['param_desc']}")
        print(f"Parameter Descriptions: {op['param_desc_map']}")
        print(f"Signature: {op['sig']}")
        print("-" * 50)

    print(searcher.records_map["nlpaug_en_mapper"]["source_path"])
    print(searcher.records_map["nlpaug_en_mapper"].test_path)


if __name__ == "__main__":
    tags = []
    op_type = "formatter"
    main(tags, op_type=op_type)
