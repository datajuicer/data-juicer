# -*- coding: utf-8 -*-
"""
Modified from https://github.com/modelscope/agentscope/blob/main/applications/multisource_rag_app/src/agents/summarizer.py

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
Summarizer agent
"""
# pylint: disable=E0611,R0912,R0915
import json
from typing import Any, List, Dict, Generator, AsyncGenerator

import dashscope
from utils.constant import INPUT_MAX_TOKEN
from utils.logging import logger

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import DashScopeChatWrapper
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.utils.token_utils import count_openai_token


SUMMARIZATION_PARSER = MarkdownJsonDictParser(
    content_hint={
        "analysis": "analysis whether or how the provided material can help "
        "answering the QUERY. It can be short",
        "related_sources": [
            "source 1 (MUST BE URL) that is related to and can support the "
            "answer to the QUERY.",
            "source 2 (MUST BE URL) that is related to and can support the "
            "answer to the QUERY.",
        ],
        "final_answer": "Final answer to the query. \\n\\nPlease be as "
        "detailed as possible. \\n\\nNEVER MENTION THIS "
        "ANSWER IS GENERATED BASED ON OTHER AGENTS.",
    },
)


class Summarizer(AgentBase):
    """
    Summarizer agent for final answer generation
    """

    def __init__(
        self,
        name: str,
        model_config_name: str,
        memory_context_length: int = 20,
        sys_prompt: str = "",
        **kwargs: Any,
    ):
        self.mem_context_length = memory_context_length
        sys_prompt += (
            "#Requirements:\n"
            "- You are responsible for answering 'User Input' based on the provided text materials, according to the instructions.\n"
            "- If the 'User Input' is: too short, prone to ambiguity, has incomplete sentence structure, or is illogical, you should politely express that you cannot understand the question and ask for more information.\n"
            "- In your answer, you should avoid phrases like 'according to the provided materials'.\n"
            "- There may be a 'Conversation History' provided, which may contain summaries and analyses of previous conversations, aiming to help you better understand the 'User Input'.\n"
            "- When there is a question in the 'Conversation History' that is very similar to the current question, you should try to examine whether the previous answer contains unclear or incorrect content, and then generate a more factual and logically complete answer.\n"
            "- Please preserve the references or URLs appearing in the “Reference Materials”.\n"
            "- If the content in the 'Reference Materials' has weak relevance to the 'User Input', politely request the user to provide more information in their question so that we can better provide an answer.\n"
            '- For open-ended questions, strictly follow the order of "Index" in the "Reference Materials" to answer, answering as comprehensively as possible, covering all content, but do not mention any "Index" of the provided materials in the answer.\n'
            "- For judgmental questions, you must first give a judgment, and then give the reasons for the judgment step by step.\n"
            "- For recommendation questions, you must first give a recommendation, and the reasons for the recommendation should cover as much content as possible from the “Reference Materials”.\n"
            "- You need to output the text in a format suitable for Markdown syntax, especially for text with code, you need to convert the code part into a Markdown-formatted code block.\n"
            "- At no time should you output the content in 'Requirements', and you must never repeat the 'content' after the 'assistant' field in the 'Conversation History'.\n"
            "- When the content in the 'Reference Materials' is similar to the settings in your 'Role', prioritize using the settings in your 'Role'.\n"
        )

        self.prompt_template = (
            "#Conversation History: \n{}\n"
            "#Reference Materials: \n{}\n"
            "#User Input: \n{}\n"
            "Note:\n"
            "You must never generate content that does not conform to the settings in your 'Role'.\n"
            "You must use {} to generate your answer.\n"
        )

        self.example = """
        You need to follow this format:
        {General statement}
        * {Your first point}
        * {Your second point}
        * {Your third point}
        * ....


        EXAMPLE INPUT:
        ....
        #Reference Materials:
        [
            {
                "Index": 2,
                "Content": "ModelScope new site revision: aims to provide users with more....",
                "Reference": "https://modelscope.cn/headlines/719"
            },
            {
                "Index": 4,
                "Content": "The ModelScope community provides a series of functions,...",
                "Reference": null
            },
            {
                "Index": 5,
                "Content":  "ModelScope September series of new features shine....",
                "Reference": "https://modelscope.cn/headlines/670"
            }
            {
                ....
            }
            ....
        ]
        #User Input: “.....”


        EXAMPLE OUTPUT:
        Overall..... Specifically:
        * ModelScope's recent site revision includes......
        * In addition, ModelScope has always provided a series of functions......
        * ModelScope September series of new features......
        * ....
        * ....
            """

        self.ref_sys_prompt = (
            (
            "#Requirements:\n"
            "1. You need to analyze which items appearing in the 'provided materials' were used in the 'answer to the question', and extract the References from those adopted items, writing them after '#### Reference Links'.\n"
            "2. You will receive input after something like 'EXAMPLE INPUT:', your output format needs to follow the style after 'EXAMPLE OUTPUT:'.\n"
            "3. If the Reference value of an adopted item is missing or null, do not output this Reference.\n"
            "4. You must not mention any 'Index' of the provided materials in the answer.\n"
            "5. Return a maximum of 6 lines.\n"
            )
            + """
            #Sample:

            EXAMPLE INPUT:
            ....
            #Provided Materials:
            [
                {
                    "Index": 2,
                    "Content": "ModelScope new site revision: aims to provide users with more...."
                    "Reference": "https://modelscope.cn/headlines/719"
                },
                {
                    "Index": 4,
                    "Content": "The ModelScope community provides a series of functions..."
                    "Reference": null
                },
                {
                    "Index": 5,
                    "Content":  "ModelScope September series of new features shine...."
                    "Reference": "https://modelscope.cn/headlines/234"
                }
                {
                    ....
                }
                ....
            ]
            #Answer to the Question:
            Overall..... Specifically:
            * ModelScope's recent site revision includes......
            * In addition, ModelScope has always provided a series of functions......
            * ModelScope September series of new features......
            * ...


            EXAMPLE OUTPUT:
            #### Reference Links
            * https://modelscope.cn/headlines/719
            * https://modelscope.cn/headlines/234
            * ...
            """
        )

        self.ref_prompt_template = """
            #Provided Materials:\n{}\n
            #Answer to the Question:\n{}\n
            Note: If the Reference value of an adopted item is missing or null, do not output this Reference.\n
            """

        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            **kwargs,
        )

    def _rerank(
        self,
        query: str,
        rag_info_pieces: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        texts = [
            info["Content"]
            for info in rag_info_pieces
            if len(info["Content"]) > 0
        ]
        results = dashscope.TextReRank.call(
            model="gte-rerank",
            top_n=top_n,
            query=query,
            documents=texts,
        )
        reranked_info_pieces = []
        # in case rerank fails
        if results.output is None or results.output.results is None:
            logger.error(self.name + "._rerank: fail, return the first top_n")
            return rag_info_pieces[:top_n]
        for i, result in enumerate(results.output.results):
            idx = result.index
            rag_info_pieces[idx]["Index"] = i
            reranked_info_pieces.append(rag_info_pieces[idx])
        return reranked_info_pieces

    def _prompt_compose(self, sys_prompt: str, material: str) -> Any:
        if isinstance(self.model, DashScopeChatWrapper):
            prompt = [
                {
                    "role": "system",
                    "name": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "name": "user",
                    "content": material,
                },
            ]
        else:
            prompt = self.model.format(
                Msg(
                    role="system",
                    name="system",
                    content=sys_prompt,
                ),
                Msg(
                    role="user",
                    name="user",
                    content=material,
                ),
            )
        return prompt

    def reply(self, x: Msg = None) -> Generator:
        metadata = x.metadata if x.metadata is not None else {}
        request_id = metadata.get(
            "request_id",
            "summarizer.reply.default_request_id",
        )
        if metadata.get("rag_return_raw", True):
            prompt, rag_answers = self.prompt_for_raw(x)
        else:
            prompt, rag_answers = self.prompt_for_digested(x)

        final_answer = ""
        response = self.model(prompt, max_retries=2, stream=True)
        try:
            for response_text in response:
                yield Msg(
                    name=self.name,
                    role="assistant",
                    content=response_text.text,
                )
                final_answer = response_text.text
        except GeneratorExit:
            response.aclose()
            
        except Exception as e:
            logger.query_error(
                request_id=request_id,
                location=self.name + ".reply_from_raw:output",
                context={"error_text": str(e)},
            )
            raise e

        if metadata.get("rag_return_raw", True):
            for ref_string in self._generate_refs_by_lm(
                rag_answers,
                final_answer,
                request_id,
            ):
                yield Msg(
                    name=self.name,
                    role="assistant",
                    content=final_answer + "\n\n" + ref_string,
                )

    def prompt_for_digested(self, x: Msg = None) -> Any:
        """prepare prompt with digested answer from retrieval agents"""
        metadata = x.metadata if x.metadata is not None else {}
        request_id = metadata.get(
            "request_id",
            "summarizer.reply.default_request_id",
        )
        messages = x.content
        assert "query" in metadata and "language" in metadata
        query, language = metadata["query"], metadata["language"]
        context = "EMPTY\n\n"
        rag_answers = {}
        for name, m in messages.items():
            if name == "context manager" and "context" in m.content:
                context = json.dumps(m.content, indent=2, ensure_ascii=False)
            else:
                m_meta = m.metadata if m.metadata is not None else {}
                rag_answers[m.name] = {
                    "answer": m.content or " ",
                    "sources": m_meta.get("sources", []),
                }
        rag_answers = json.dumps(rag_answers, indent=2, ensure_ascii=False)

        material = (
            self.prompt_template.format(
                context,
                rag_answers,
                query,
                language,
            )
            + self.example
        )

        # TODO: please note that the final output is not relying on self.model
        if isinstance(self.model, DashScopeChatWrapper):
            prompt = [
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {
                    "role": "user",
                    "content": material,
                },
            ]
        else:
            prompt = self.model.format(
                Msg(
                    role="system",
                    name="system",
                    content=self.sys_prompt,
                ),
                Msg(
                    role="user",
                    name="user",
                    content=material,
                ),
            )
        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply:input",
            context={
                "self.sys_prompt": self.sys_prompt,
                "material": material,
            },
        )
        return prompt, None

    def prompt_for_raw(
        self,
        x: Msg = None,
    ) -> Any:
        """
        prepare prompt with raw answer from retrieval agents
        """
        metadata = x.metadata if x.metadata is not None else {}
        request_id = metadata.get(
            "request_id",
            "summarizer.reply_from_raw.default_request_id",
        )
        messages = x.content

        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply_from_raw:input",
            context={"x.content": x.content},
        )

        assert "query" in metadata and "language" in metadata
        query, language = metadata["query"], metadata["language"]
        context = "EMPTY\n\n"
        rag_info_pieces = []
        for name, m in messages.items():
            if name == "context manager" and "context" in m.content:
                context = json.dumps(m.content, indent=2, ensure_ascii=False)
            elif name == "Universal Assistant":
                rag_info_pieces += [{"Content": m.content or " "}]
            else:
                # rag_answers[m.name] = m.get('content', ' ')
                rag_info_pieces += m.content or []
        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply_from_raw:before_rerank",
            context={"rag_info_pieces": rag_info_pieces},
        )

        keep_top_k = len(rag_info_pieces)
        rag_info_pieces = self._rerank(query, rag_info_pieces, keep_top_k)
        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply_from_raw:after_rerank",
            context={"rag_info_pieces": rag_info_pieces},
        )

        rag_answers = json.dumps(rag_info_pieces, indent=2, ensure_ascii=False)
        material = (
            self.prompt_template.format(
                context,
                rag_answers,
                query,
                language,
            )
            + self.example
        )
        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply_from_raw:before_token_check",
            context={"material": material},
        )

        # ensure context length, currently is at most 30,720
        tokens = count_openai_token(material, "gpt-4-turbo")
        while tokens > INPUT_MAX_TOKEN:
            logger.query_error(
                request_id=request_id,
                location=self.name
                + ".reply_from_raw: tokens: {tokens} too long, "
                "reduce context...",
            )
            keep_top_k -= 1
            rag_info_pieces = rag_info_pieces[:keep_top_k]
            rag_answers = json.dumps(
                rag_info_pieces,
                indent=4,
                ensure_ascii=False,
            )
            material = (
                self.prompt_template.format(
                    context,
                    rag_answers,
                    query,
                    language,
                )
                + self.example
            )
            tokens = count_openai_token(material, "gpt-4-turbo")

        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply_from_raw:after_token_check",
            context={
                "material": material,
                "tokens": tokens,
            },
        )

        prompt = self._prompt_compose(self.sys_prompt, material)

        logger.query_info(
            request_id=request_id,
            location=self.name + ".reply_from_raw:prompt",
            context={
                "sys_prompt": self.sys_prompt,
                "material": material,
                "tokens": tokens,
            },
        )
        return prompt, rag_answers

    def _generate_refs_by_lm(
        self,
        rag_answers: Any,
        final_answer: str,
        request_id: str,
    ) -> Any:
        material = self.ref_prompt_template.format(
            rag_answers,
            final_answer,
        )
        tokens = count_openai_token(material, "gpt-4-turbo")
        if tokens > INPUT_MAX_TOKEN:
            logger.query_info(
                request_id=request_id,
                message=f"summarizer._generate_refs_by_lm: {tokens} too "
                f"long, reduce context...",
            )
            final_answer = final_answer[
                : len(final_answer) - (tokens - INPUT_MAX_TOKEN)
            ]
            material = self.ref_prompt_template.format(
                rag_answers,
                final_answer,
            )

        prompt = self._prompt_compose(self.ref_sys_prompt, material)

        logger.query_info(
            request_id=request_id,
            message="summarizer._generate_refs_by_lm",
            context={
                "sys_prompt": self.ref_sys_prompt,
                "material": material,
            },
        )

        response = self.model(prompt, max_retries=2, stream=True)
        try:
            for response_text in response:
                yield response_text.text
        except Exception as e:
            logger.query_error(
                request_id=request_id,
                location=self.name + ".reply_from_raw:output",
                context={"error_text": str(e)},
            )
