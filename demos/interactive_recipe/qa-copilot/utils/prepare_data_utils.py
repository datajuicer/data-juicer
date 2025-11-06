# -*- coding: utf-8 -*-
"""
Modified from https://github.com/modelscope/agentscope/blob/main/applications/multisource_rag_app/src/utils/prepare_data_utils.py

Data-Juicer adopts Apache 2.0 license, the original license of this file
is as follows:

Copyright 2024 Alibaba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Prepare tutorial data of Data Juicer
"""
import os
from typing import Optional


def prepare_docstring_txt(repo_path: Optional[str] = None, text_dir: Optional[str] = None) -> None:
    """
    If repo_path and text_dir are provided, and text_dir is empty,
    it prepares the docstring in html, and save it.
    Args:
        repo_path (`str`):
            The path of the repo
        text_dir (`str`):
            The path of the text dir
    Returns:
        None
    """
    print(f"DJ repo path: {repo_path}, text_dir: {text_dir}")
    if (
        repo_path
        and text_dir
        and os.path.exists(repo_path)
        and not os.path.exists(text_dir)
    ):
        os.system(
            f"sphinx-build -b text asset/knowledge {text_dir}"
        )