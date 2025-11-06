# -*- coding: utf-8 -*-
import inspect
from data_juicer.tools.op_search import OPSearcher

searcher = OPSearcher(include_formatter=False)

all_ops = searcher.search()

dj_func_info = []
for i, op in enumerate(all_ops):
    class_entry = {
        "index": i,
        "class_name": op["name"],
        "class_desc": op["desc"],
    }
    param_desc = op["param_desc"]
    param_desc_map = {}
    args = ""
    for item in param_desc.split(":param"):
        _item = item.split(":")
        if len(_item) < 2:
            continue
        param_desc_map[_item[0].strip()] = ":".join(_item[1:]).strip()

    if op["sig"]:
        for param_name, param in op["sig"].parameters.items():
            if param_name in ["self", "args", "kwargs"]:
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param_name in param_desc_map:
                args += f"        {param_name} ({param.annotation}): {param_desc_map[param_name]}\n"
            else:
                args += f"        {param_name} ({param.annotation})\n"
    class_entry["arguments"] = args
    dj_func_info.append(class_entry)
