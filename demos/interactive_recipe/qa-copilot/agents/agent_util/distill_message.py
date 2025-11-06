# -*- coding: utf-8 -*-
"""
Copied from https://github.com/modelscope/agentscope/blob/main/applications/multisource_rag_app/src/agents/agent_util/distill_message.py

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
This module includes function(s) for shortening messages.
"""
import re
from agentscope.message import Msg


def rule_based_shorten_msg(
    origin_message: Msg,
    len_per_chunk: int = 50,
) -> Msg:
    """
    Shorten a message to at most len_per_chunk length.
    Args:
        origin_message (`Msg`):
            Original message.
        len_per_chunk (`int`):
            Length of shortened message.
    Returns:
        `Msg`: a message with processed information
    """
    # add "\n" to ensure result content not empty when there are no newlines
    content = origin_message.content + "\n"
    list_content = re.findall("(.*)\n", content)
    truncated_content = []
    for s in list_content:
        if len(s) > len_per_chunk:
            truncated_content.append(s[:len_per_chunk] + "...")
        else:
            truncated_content.append(s)
    return Msg(
        name=origin_message.name,
        role=origin_message.role,
        content="\n".join(truncated_content),
    )
