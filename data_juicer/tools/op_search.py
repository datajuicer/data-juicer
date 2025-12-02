#!/usr/bin/env python3
"""
Operator Searcher - A tool for filtering Data-Juicer operators by tags
"""
import inspect
import re
from typing import Dict, List, Optional

from data_juicer.format.formatter import FORMATTERS
from data_juicer.ops import OPERATORS


class OPRecord:
    """A record class for storing operator metadata"""

    def __init__(self, op_type: str, name: str, desc: str, tags: List[str], sig: inspect.Signature, param_desc: str):
        self.type = op_type
        self.name = name
        self.desc = desc
        self.tags = tags
        self.sig = sig
        self.param_desc = param_desc

    def to_dict(self):
        return {
            "type": self.type,
            "name": self.name,
            "desc": self.desc,
            "tags": self.tags,
            "sig": self.sig,
            "param_desc": self.param_desc,
        }


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


def analyze_resource_tag(code):
    """
    Analyze the resource tag for the given code content string. Should be one
    of the "Resource Tags" in `tagging_mappings.json`. It makes the choice
    according to their assigning statement to attribute `_accelerator`.
    """
    if "_accelerator = " in code:
        if "_accelerator = 'cuda'" in code:
            return ["gpu"]
        else:
            return ["cpu"]
    else:
        return []


def analyze_model_tags(code):
    """
    Analyze the model tag for the given code content string. SHOULD be one of
    the "Model Tags" in `tagging_mappings.json`. It makes the choice by finding
    the `model_type` arg in `prepare_model` method invocation.
    """
    pattern = r"model_type=[\'|\"](.*?)[\'|\"]"
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
            current_code = inspect.getsource(cls)
            current_tags = analyze_func(current_code, **other_parm)
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


class OPSearcher:
    """Operator search engine"""

    def __init__(self, specified_op_list: Optional[List[str]] = None, include_formatter: bool = False):
        if specified_op_list:
            self.op_records = self._scan_specified_ops(specified_op_list)
        else:
            self.op_records = self._scan_all_ops(include_formatter)

    def _scan_specified_ops(self, specified_op_list: List[str]) -> List[OPRecord]:
        """Scan specified operators"""
        records = []
        for op_name in specified_op_list:
            if op_name in FORMATTERS.modules:
                op_type = "formatter"
                op_cls = FORMATTERS.modules[op_name]
            else:
                op_type = op_name.split("_")[-1]
                op_cls = OPERATORS.modules[op_name]
            desc = op_cls.__doc__ or ""
            tags = analyze_tag_from_cls(op_cls, op_name)
            sig = inspect.signature(op_cls.__init__)
            init_param_desc = extract_param_docstring(op_cls.__init__.__doc__ or "")
            records.append(
                OPRecord(op_type=op_type, name=op_name, desc=desc, tags=tags, sig=sig, param_desc=init_param_desc)
            )
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
        return {record.name: record for record in self.op_records}


def main(tags, op_type):
    searcher = OPSearcher(include_formatter=True)

    results = searcher.search(tags=tags, op_type=op_type)

    print(f"\nFound {len(results)} operators:")
    for op in results:
        print(f"\n[{op['type'].upper()}] {op['name']}")
        print(f"Tags: {', '.join(op['tags'])}")
        print(f"Description: {op['desc']}")
        print(f"Parameters: {op['param_desc']}")
        print(f"Signature: {op['sig']}")
        print("-" * 50)


if __name__ == "__main__":
    tags = []
    op_type = "formatter"
    main(tags, op_type=op_type)
