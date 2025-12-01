import datetime
import importlib
import inspect
import json
import logging
import os
from typing import Dict
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from jsonargparse import Namespace
from pydantic import validate_call

from data_juicer.config.config import get_default_cfg
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.exporter import Exporter

DJ_OUTPUT = "outputs"

allowed_methods = {
    "run",
    "process",
    "compute_stats",
    "compute_hash",
    "analyze",
    "compute",
    "process_single",
    "process_batched",
    "compute_stats_single",
    "compute_stats_batched",
}

logger = logging.getLogger("uvicorn.error")
app = FastAPI()


def register_objects_from_init(directory: str):
    """
    Traverse the specified directory for __init__.py files and
    register objects defined in __all__.
    """
    for dirpath, _, filenames in os.walk(os.path.normpath(directory)):
        if "__init__.py" in filenames:
            module_path = dirpath.replace(os.sep, ".")
            module = importlib.import_module(module_path)

            if hasattr(module, "__all__"):
                for name in module.__all__:
                    obj = getattr(module, name)
                    if inspect.isclass(obj):
                        register_class(module, obj)
                    elif callable(obj):
                        register_function(module, obj)


def register_class(module, cls):
    """Register class and its methods as endpoints."""

    def create_class_call(cls, method_name: str):
        async def class_call(request: Request):
            try:
                # wrap init method
                cls.__init__ = validate_call(cls.__init__, config=dict(arbitrary_types_allowed=True))
                # parse json body as cls init args
                init_args = await request.json() if await request.body() else {}
                # create an instance
                instance = cls(**_setup_cfg(init_args))
                # wrap called method

                """
                Optimized according to issue #777 thanks to https://github.com/Night-Quiet
                """
                method = validate_call(getattr(instance, method_name), config=dict(arbitrary_types_allowed=True))

                result = _invoke(method, request)
                return {"status": "success", "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return class_call

    module_path = module.__name__.replace(".", os.sep)
    cls_name = cls.__name__
    for method_name in _get_public_methods(cls, allowed_methods):
        api_path = f"/{module_path}/{cls_name}/{method_name}"
        class_call = create_class_call(cls, method_name)
        app.add_api_route(api_path, class_call, methods=["POST"], tags=["POST"])
        logger.debug(f"Registered {api_path}")


def register_function(module, func):
    """Register a function as an endpoint."""

    def create_func_call(func):
        async def func_call(request: Request):
            try:
                nonlocal func
                func = validate_call(func, config=dict(arbitrary_types_allowed=True))
                result = _invoke(func, request)
                return {"status": "success", "result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return func_call

    module_path = module.__name__.replace(".", os.sep)
    func_name = func.__name__
    api_path = f"/{module_path}/{func_name}"
    func_call = create_func_call(func)
    app.add_api_route(api_path, func_call, methods=["GET"], tags=["GET"])
    logger.debug(f"Registered {api_path}")


def _invoke(callable, request):
    # parse query params as cls method args
    q_params = parse_qs(request.url.query, keep_blank_values=True)
    # flatten lists with a single element
    d_params = dict((k, v if len(v) > 1 else v[0]) for k, v in q_params.items())
    # parse json dumps
    d_params = _parse_json_dumps(d_params)
    # pre-processing
    d_params = _setup_cfg(d_params)
    exporter = _setup_dataset(d_params)
    skip_return = d_params.pop("skip_return", False)
    # invoke callable
    result = callable(**d_params)
    # post-processing
    if exporter is not None:
        exporter.export(result)
        result = exporter.export_path
    if skip_return:
        result = ""
    return result


def _parse_json_dumps(params: Dict, prefix="<json_dumps>"):
    for k, v in params.items():
        if isinstance(v, str) and v.startswith(prefix):
            params[k] = json.loads(v[len(prefix) :])
    return params


def _setup_cfg(params: Dict):
    """convert string `cfg` to Namespace"""
    # TODO: Traverse method's signature and convert any arguments \
    #  that should be Namespace but are passed as str
    if cfg_str := params.get("cfg"):
        if isinstance(cfg_str, str):
            cfg = Namespace(**json.loads(cfg_str))
            params["cfg"] = cfg
    return params


def _setup_dataset(params: Dict):
    """setup dataset loading and exporting"""
    exporter = None
    if dataset_path := params.get("dataset"):
        if isinstance(dataset_path, str):
            cfg = get_default_cfg()
            cfg.dataset_path = dataset_path
            builder = DatasetBuilder(cfg)
            dataset = builder.load_dataset()
            params["dataset"] = dataset
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(DJ_OUTPUT, timestamp, "processed_data.jsonl")
            exporter = Exporter(export_path, keep_stats_in_res_ds=True, keep_hashes_in_res_ds=True, export_stats=False)
    return exporter


def _get_public_methods(cls, allowed=None):
    """Get public methods of a class."""
    all_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    return [name for name, _ in all_methods if not name.startswith("_") and (allowed is None or name in allowed)]


# Specify the directories to search
directories_to_search = [
    "data_juicer",
    # "tools",  # Uncomment to add more directories
]

# Register objects from each specified directory
for directory in directories_to_search:
    register_objects_from_init(directory)
