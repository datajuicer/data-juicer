import datetime
import glob
import importlib
import os
from json import dumps as jdumps
from json import loads as jloads
from typing import Dict, Optional
from urllib.parse import urljoin

import requests
import yaml
from agentscope.message import Msg
from agentscope.service import ServiceToolkit
from loguru import logger
from PIL import Image

DJ_BASE_URL = "http://localhost:8000"
DJ_CONFIG_TEMPLATE = "./configs/dj_config_template.yaml"
DJ_OUTPUT = "outputs"


def call_data_juicer_api(path: str, params: Optional[Dict] = None, json: Optional[Dict] = None):
    url = urljoin(DJ_BASE_URL, path)

    if json is not None:
        response = requests.post(url, params=params, json=json)
    else:
        response = requests.get(url, params=params)

    return jloads(response.text)


def init_config(dataset_path: str, op_name: str, **op_args):
    """
    Initialize Data-Juicer config with operator `op_name`.

    Args:
        dataset_path (`str`):
            The input dataset path.
        op_name: name of the operator.
        op_args: arguments of the operator.
    """
    with open(DJ_CONFIG_TEMPLATE) as fin:
        dj_config = yaml.safe_load(fin)
    dj_config["dataset_path"] = dataset_path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dj_config["export_path"] = os.path.join(DJ_OUTPUT, timestamp, "processed_data.jsonl")
    dj_config["process"].append({op_name: op_args})
    url_path = "/data_juicer/config/get_init_configs"
    try:
        res = call_data_juicer_api(url_path, params={"cfg": jdumps(dj_config)})
    except Exception as e:
        error_msg = f"An unexpected error occurred in calling {url_path}:\n{e}"
        raise RuntimeError(error_msg)
    return res["result"]


def execute_analyzer(dj_config: dict):
    """
    Execute data-juicer analyzer.

    Args:
        dj_config: configs of data-juicer
    """
    logger.chat(Msg(name="system", content="Analyzing data...", role="system"))
    url_path = "/data_juicer/core/Analyzer/run"
    try:
        res = call_data_juicer_api(url_path, params={"skip_return": True}, json={"cfg": jdumps(dj_config)})
        print(res)
        assert res["status"] == "success"
        return dj_config["export_path"]
    except Exception as e:
        error_msg = f"An unexpected error occurred in calling {url_path}:\n{e}"
        raise RuntimeError(error_msg)


def show_analyzed_results(analyzed_result_path: str, require_min=True, require_max=True):
    """
    Show the analyzed results to the users and get the specified thresholds.

    Args:
        analyzed_result_path (`str`):
            The analyzed result path.
    """

    if os.path.isfile(analyzed_result_path):
        analyzed_result_path = os.path.join(os.path.dirname(analyzed_result_path), "analysis")

    hist_file = max(glob.glob(os.path.join(analyzed_result_path, "*hist.png")), key=os.path.getctime, default=None)

    if hist_file is not None:
        img = Image.open(hist_file)
        img.show()
        min_threshold, max_threshold = 0, 0
        if require_min:
            min_threshold = float(
                input("Based on above analyzed results, " "enter the minimum threshold value for filter: ")
            )
        if require_max:
            max_threshold = float(
                input("Based on above analyzed results, " "enter the maximum threshold value for filter: ")
            )
        return min_threshold, max_threshold
    else:
        error_msg = f"Error in showing analyzed result: {analyzed_result_path}"
        raise RuntimeError(error_msg)


def execute_config(dj_config: Dict):
    """
    Execute data-juicer data process.

    Args:
        dj_config: configs of data-juicer
    """
    logger.chat(Msg(name="system", content="Processing data...", role="system"))
    url_path = "/data_juicer/core/Executor/run"
    try:
        res = call_data_juicer_api(url_path, params={"skip_return": True}, json={"cfg": jdumps(dj_config)})
        print(res)
        assert res["status"] == "success"
        return dj_config["export_path"]
    except Exception as e:
        error_msg = f"An unexpected error occurred in calling {url_path}:\n{e}"
        raise RuntimeError(error_msg)


def setup_service_toolkit(module_name, service_toolkit=None):
    if service_toolkit is None:
        service_toolkit = ServiceToolkit()
    module = importlib.import_module(module_name)
    for name in dir(module):
        if name.startswith("execute_"):
            obj = getattr(module, name)
            if callable(obj):
                service_toolkit.add(obj)
    return service_toolkit
