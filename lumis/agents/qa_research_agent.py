from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import textwrap
from typing import List, Optional

from lumis.core import Chunk
from lumis.embedding import BaseEmbeddingModel
from lumis.llm.openai_llm import OpenAILLM
from lumis.tools.search import VectorSearchRetrievalEngine

from .base.graph_based_agent import GraphBasedAgent

from openai import pydantic_function_tool
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Queries(BaseModel):
    """Represents a structured request for up to three search queries."""

    queries: list[str] = Field(description="Three or fewer carefully crafted search queries to comprehensively address the user's question.")


class State(TypedDict):
    question: str
    answer: str
    references: list[Chunk]

    attempt: int


Events = None  # Literal[""]


class QAResearchAgent(GraphBasedAgent[State, Events]):
    class ErrorMessages:
        EMPTY_QUESTION = "There does not seem to be a valid question. It seems to be am empty string."
        UNABLE = "I am unable to answer your question at this time."
        UNSUCCESSFUL = "My search was unsuccessful, please try asking another question."
        UNABLE_TO_ANSWER = "I am unable to answer your question, please try asking something else."

        @classmethod
        def is_error_message(cls, message: str) -> bool:
            return message in [
                cls.EMPTY_QUESTION,
                cls.UNABLE,
                cls.UNSUCCESSFUL,
                cls.UNABLE_TO_ANSWER,
            ]

    def __init__(
        self,
        embedding: BaseEmbeddingModel,
        search_engine: VectorSearchRetrievalEngine,
        llm: Optional[OpenAILLM] = None,
        # TODO: Add wiki as another source of data to query
        # wiki: Optional[WikipediaSearcher] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
        max_attempts: int = 3,
        include_citations: bool = False,
        include_datetime: bool = False,
    ):
        super().__init__(llm, verbose=verbose, logger=logger)

        self.embedding = embedding
        self.search_engine = search_engine

        # Options
        self.include_citations = include_citations
        self.include_datetime = include_datetime
        self.max_attempts = max_attempts

    async def setup(self, question: str):
        if self.verbose:
            print(f"User:\n{question}")

        self.graph.set_initial_state({"question": question, "attempt": 0, "references": [], "answer": None})

    def construct_graph(self):
        self.graph.add_node("input", self.input, "start")
        self.graph.add_node("generate_answer", self.generate_answer)

        self.graph.add_edge(
            "input",
            "generate_answer",
            # This condition ensures the we dont move onto generating an answer during
            # if the input was found invalid
            condition=lambda x: x.get("answer", None) is None,
        )

        self.graph.add_edge(
            "generate_answer",
            "generate_answer",
            # This allows us to retry, however the assumption here is that we are
            # populating the answer when we reach our max attempts.
            condition=lambda x: not x.get("answer", None) or self.ErrorMessages.is_error_message(x.get("answer")) and x.get("attempt", 0) < self.max_attempts,
        )

    # TODO: Take configurations to control number of queries and context length? Maybe summarize chunks?

    async def input(self, state: State):
        question = state.get("question")
        new_state = {}
        if not question.strip():
            new_state.update({"answer": self.ErrorMessages.EMPTY_QUESTION})

        return new_state

    def _generate_query_messages(self, question):
        query_messages = []

        # Add additional system messages based on constructor options
        self._include_datetime(query_messages)

        query_messages.append(
            ChatCompletionUserMessageParam(role="user", content=textwrap.dedent(question).strip()),
        )

        return query_messages

    async def generate_answer(self, state: State):  # noqa: C901
        question = state.get("question", "")
        attempt = state.get("attempt", 0) + 1

        queries, tool_call, response = await self._create_queries(self._generate_query_messages(question))
        new_state: dict = {"attempt": attempt}

        # Handle empty tool call (No queries)
        if tool_call is None:
            if attempt < self.max_attempts:
                # return without an answer so we can try again
                return new_state
            else:
                new_state["answer"] = self.ErrorMessages.UNABLE
                return new_state
        # Perform searches and gather chunks. Use the cache to avoid re-searching the same queries.
        search_results = await self._search(queries)
        # Ensure we have some chunks from our searches.
        if len(search_results) < 1:
            new_state["answer"] = self.ErrorMessages.UNSUCCESSFUL
            return new_state

        dumped = json.dumps([self._clean_chunk_for_llm_consumption(r) for r in search_results])

        try:
            messages = self._generate_answer_messages(question, response, dumped, tool_call)
            answer_response = await self.llm.completion(messages=messages)
            answer = answer_response.content

            if self.verbose:
                print(f"Assistant:\n{answer}")

            new_state.update({"answer": answer, "references": search_results})

        except Exception as e:
            self.logger.info(e)
            new_state["answer"] = self.ErrorMessages.UNABLE_TO_ANSWER

        return new_state

    def _generate_answer_messages(
        self,
        question: str,
        response: ChatCompletionAssistantMessageParam,
        dumped: str,
        tool_call: object,
    ):
        """
        Generate the list of messages for the OpenAILLM completion based on the question, OpenAILLM response,
        and retrieved chunks, incorporating options like datetime and citations.
        """
        messages: list[ChatCompletionMessageParam] = []

        self._include_datetime(messages)
        self._include_citations(messages)

        messages.extend(
            [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=textwrap.dedent(
                        """
                    You are a subject matter expert providing answers backed by research. 
                    If the results from the research do not answer the question please respond 
                    with, 'I am unable to answer your question, please try asking something else.'
                    
                    When answering your questions simply answer the question do not finish your response with anything like,
                    'If you have any further questions or need more specific information, feel free to ask!'
                    """
                    ).strip(),
                ),
                ChatCompletionUserMessageParam(role="user", content=textwrap.dedent(question).strip()),
            ]
        )

        # Add the original AI system message and tool context
        messages.append(response)
        messages.append(ChatCompletionToolMessageParam(role="tool", content=dumped, tool_call_id=tool_call.id))

        return messages

    async def _create_queries(self, messages: list[ChatCompletionMessageParam]):
        """
        Use the OpenAILLM to produce structured queries for the search engine.
        Args:
            messages (list[ChatCompletionMessageParam]): The messages representing the current conversation.
        Returns:
            tuple: (Queries, tool_call, response) containing the parsed queries, the tool call details, and the raw OpenAILLM response.
        """

        response = await self.llm.structured_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant. Propose queries to help find information needed in order to answer the users question.",
                },
                *messages,
            ],
            tools=[pydantic_function_tool(Queries)],
            tool_choice="required",
            parallel_tool_calls=False,
        )  # type: ignore
        queries: list[str] = []

        tool_calls = response.tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            return queries, None, response

        tool_call = tool_calls[0]
        queries_response: Queries = tool_call.function.parsed_arguments  # type: ignore
        queries = queries_response.queries

        return queries, tool_call, response

    async def _search(self, queries: list[str]):
        tasks = [self.search_engine.search(query=query, max_results=3, k=3) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results: List[Chunk] = []
        for r in results:
            if isinstance(r, list):
                successful_results.extend(r)

        return successful_results

    def _clean_chunk_for_llm_consumption(self, chunk: Chunk):
        # Ensures that we only display the content and the url to the llm
        chunk_dict = {**chunk.model_dump()}

        #  Keep only the url from the meta data
        chunk_dict["metadata"] = {"url": (chunk.metadata or {}).get("url", "")}

        # remove the identifiers
        chunk_dict.pop("doc_id")
        chunk_dict.pop("parent_id")

        return chunk_dict

    def _include_datetime(self, messages: list[ChatCompletionMessageParam]):
        if self.include_datetime:
            datetime_context = f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=textwrap.dedent(datetime_context).strip(),
                )
            )
        return messages

    def _include_citations(self, messages: list[ChatCompletionMessageParam]):
        if self.include_citations:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="When presenting your answer, use footnotes like '[1]' for citations, and include URLs in a references section without duplicates.",
                )
            )
        return messages
