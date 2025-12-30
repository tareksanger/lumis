from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional, TypeVar

from .models.survey_subjects import Editor

from lumis.core.document import Chunk
from lumis.kit.graph import Graph
from lumis.llm.openai_llm import OpenAILLM
from lumis.tools.search import VectorSearchRetrievalEngine
from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

logger = logging.getLogger("Interview")

T = TypeVar("T", bound=BaseModel)


class Queries(BaseModel):
    """Represents a structured request for up to three search queries."""

    queries: List[str] = Field(description="Three or fewer carefully crafted queries to comprehensively address the user's questions.")

    @field_validator("queries", mode="after")
    def limit_queries(cls, v: List[str]) -> List[str]:
        # Limit the queries to only 3 if more are provided.
        return v[:3]


class State(TypedDict):
    """
    Typed dictionary representing the internal state of the interview process.

    Attributes:
        editor (Editor): The fictional Wikipedia editor with a unique persona.
        messages (list[ChatCompletionMessageParam]): The ongoing conversation messages.
        references (list[Chunk]): A list of reference chunks retrieved from search queries.
        is_done (bool): Indicates whether the interview is finished.
        turns (int): How many turns (Q&A cycles) have been taken.
    """

    editor: Editor
    messages: list[ChatCompletionMessageParam]
    references: list[Chunk]
    is_done: bool
    turns: int


ANSWER_LLM_NAME = "Subject_Matter_Expert"
END_PHRASE = "Thank you so much for your help!"


class InterviewGraph:
    """
    The InterviewGraph orchestrates a conversation (interview) between a fictional Wikipedia editor (with a specific persona)
    and a subject matter expert. The goal is to gather rich, well-sourced information from the expert to help write or improve
    a Wikipedia article on a given topic.

    Workflow:
    1. ask -> The editor asks questions, trying to extract more information.
    2. answer -> The subject matter expert responds using web search results (via a retrieval engine) to provide fact-based answers.
    3. route -> Decide if the interview continues or ends based on conditions (e.g., number of turns, the end phrase).

    The class uses a `Graph[State]` to manage states and transitions, and incorporates a ThreadSafeCache to avoid redundant web searches.
    """

    def __init__(self, llm: OpenAILLM, search_engine: VectorSearchRetrievalEngine, max_turns: int = 3, verbose: bool = False):
        """
        Initialize the InterviewGraph.

        Args:
            llm (OpenAILLM): The language model used for generating system, user, and assistant messages.
            embedding (BaseEmbeddingModel): Embedding model for semantic retrieval.
            search_engine (VectorSearchRetrievalEngine): Web retrieval engine for queries.
            max_turns (int): The maximum number of Q&A cycles before the interview ends.
            verbose (bool): If True, prints debug info and messages to stdout.
        """
        self.initialized = False
        self.search_engine = search_engine
        self.max_turns = max_turns
        self.verbose = verbose

        self.llm = llm
        self.graph = Graph[State]()

        self._construct_graph()

    def _construct_graph(self):
        """Construct the state graph defining the interview workflow."""
        self.graph.add_node("ask", self.ask_questions, "start")
        self.graph.add_node("answer", self.generate_answer)
        self.graph.add_node("route", self.route_messages)

        self.graph.add_edge("ask", "answer")
        self.graph.add_edge("answer", "route", condition=lambda x: not x.get("is_done", False))
        self.graph.add_edge("route", "ask", condition=lambda x: not x.get("is_done", False))

    def initialize(self, editor: Editor, topic: Optional[str] = None):
        """
        Initialize the interview with the given editor and optional topic.

        Args:
            editor (Editor): The Wikipedia editor persona.
            topic (str, optional): The topic of the Wikipedia article being discussed. If None or empty, a fallback message is used.
        """
        self.initialized = True

        messages: list[ChatCompletionMessageParam] = []
        if topic is None or not topic.strip():
            # If no topic is provided, mention the lack of topic context.
            messages.append({"role": "assistant", "content": f"So you said you were writing an article on {topic}?", "name": ANSWER_LLM_NAME})

        self.graph.set_initial_state({"editor": editor, "messages": messages, "is_done": False, "references": [], "turns": 0})

    async def run(self, editor: Editor, topic: Optional[str] = None) -> State:
        """
        Run the full interview process, from initialization through the final route.

        Args:
            editor (Editor): The Wikipedia editor persona.
            topic (str, optional): The topic of the interview.

        Returns:
            State: The final state of the interview after it completes.
        """
        self.initialize(editor=editor, topic=topic)
        await self.graph.traverse()
        return State(**self.graph.state)

    def visualize(self):
        """Visualize the state graph for debugging or review."""
        return self.graph.visualize_graph()

    async def step(self):
        """
        Execute a single step of the interview graph. Useful for manual debugging.

        Raises:
            AssertionError: If the interview has not been initialized.
        """
        assert self.initialized, "Initialize the Interview before running steps (.initialize(...))."
        return await self.graph.step(str(self.graph.current_node))

    async def ask_questions(self, state: State):
        """
        The editor (acting as a Wikipedia writer) asks the expert more questions.

        This node:
        - Prepares the conversation with a system message guiding the editor to ask good questions.
        - The editor persona is applied to generate a single new question.
        - The output is appended to the conversation messages.

        Args:
            state (State): Current state of the interview.

        Returns:
            dict: A dictionary with updated messages.
        """
        editor = state.get("editor")
        swapped_messages = self._swap_roles(editor.name, state)

        # The system message encourages the editor to ask a comprehensive question.
        messages = [
            {
                "role": "system",
                "content": f"""You are an experienced Wikipedia writer researching a topic. 
                You have a unique perspective: {editor.persona}
                
                You are chatting with a subject matter expert. Ask only one question at a time to gather as much unique insight as possible.
                If you have no more questions, say "{END_PHRASE}" to end the conversation.
                """,
            },
            *swapped_messages,
        ]

        response = await self.llm.completion(messages=messages, frequency_penalty=0.1)

        if self.verbose:
            self.logger.info(f"{editor.name}: {response.content}", end="\n" + ("-" * 100) + "\n")

        return {"messages": [*swapped_messages, {"role": "assistant", "content": response.content, "name": editor.name}]}

    async def generate_answer(self, state: State):  # noqa: C901
        """
        The subject matter expert (SME) responds to the editor's question.

        Steps:
        1. Swap roles so that the OpenAILLM sees the last user message as a question from the editor.
        2. If the last question contains END_PHRASE, the interview ends.
        3. Otherwise, query the OpenAILLM to get recommended search engine queries.
        4. Perform web searches (using caching to avoid repeats).
        5. Use the retrieved chunks as references to answer the question, providing citations.

        Args:
            state (State): Current state dictionary.

        Returns:
            dict: Updated messages and references. May include "is_done" if the session ends.
        """

        def _clean_chunk_for_llm_consumption(chunk: Chunk):
            # Ensures that we only display the content and the url to the llm
            chunk_dict = chunk.model_dump()

            #  Keep only the url from the meta data
            chunk_dict["metadata"] = {"url": (chunk.metadata or {}).get("url", "")}

            # remove the identifiers
            chunk_dict.pop("doc_id")
            chunk_dict.pop("parent_id")

            return chunk_dict

        swapped_messages = self._swap_roles(ANSWER_LLM_NAME, state)

        # The last message from the Editor was the end phrase. Ending the task.
        last_question = str(swapped_messages[-1].get("content", ""))
        if END_PHRASE in last_question:
            logger.debug("Ending Interview, last message contained the END_PHRASE")
            return {"is_done": True}

        # Get queries from OpenAILLM to drive web search
        queries, tool_call, response = await self._get_queries(swapped_messages)

        # Perform searches and gather chunks. Use the cache to avoid re-searching the same queries.
        tasks = [self._search_engine(query) for query in queries.queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate successful results
        successful_results: List[Chunk] = []
        for r in results:
            if isinstance(r, list):
                successful_results.extend(r)

        # Convert retrieved chunks to JSON for tool message
        dumped = json.dumps([_clean_chunk_for_llm_consumption(r) for r in successful_results])

        # Now use these references to produce a factually supported answer
        answer_response = await self.llm.completion(
            messages=[
                {
                    "role": "system",
                    "content": """You are a subject matter expert providing answers backed by collected references. 
                    Cite sources as footnotes and provide URLs at the end.""",
                },
                *swapped_messages,
                # The original AI system message from structured completion:
                ChatCompletionAssistantMessageParam(**response.model_dump(), name=ANSWER_LLM_NAME),
                # The retrieved documents are provided as a 'tool' message
                {"role": "tool", "content": dumped, "tool_call_id": tool_call.id},
            ]
        )

        # Update references in state
        references = state.get("references", [])
        references.extend(successful_results)

        if self.verbose:
            self.logger.info(f"{ANSWER_LLM_NAME}: {answer_response.content}", end="\n" + ("-" * 100) + "\n")

        return {"messages": [*swapped_messages, {"role": "assistant", "content": answer_response.content, "name": ANSWER_LLM_NAME}], "references": references}

    def route_messages(self, state: State):
        """
        Decide whether to continue the interview or end it based on conditions:
        - If we've reached the maximum number of turns (Q&A cycles)
        - If the last question contained the end phrase

        Args:
            state (State): Current state.

        Returns:
            dict: "is_done" indicating if the interview should end, and updated "turns" count.
        """
        messages = state.get("messages", [])
        turns = state.get("turns", 0) + 1

        # The second to last message should be the question from the editor
        last_question = str(messages[-2].get("content", "")) if len(messages) >= 2 else ""
        is_done = turns >= self.max_turns or END_PHRASE in last_question

        if is_done:
            logger.debug(f"Ending interview. Turns: {turns}, End Phrase Used: {END_PHRASE in last_question}")

        return {"is_done": is_done, "turns": turns}

    def _swap_roles(self, name: str, state: State) -> list[ChatCompletionMessageParam]:
        """
        Swap the roles in the conversation so that from the OpenAILLM's perspective:
        - Messages from the given 'name' become 'assistant' messages.
        - All others become 'user' messages.

        This helps the OpenAILLM differentiate who is speaking and ensures that the correct context
        is passed to the OpenAILLM for generating the next message.

        Args:
            name (str): The name of the persona considered as the assistant.
            state (State): Current state with messages.

        Returns:
            list[ChatCompletionMessageParam]: Messages with roles swapped.
        """
        messages = state.get("messages", [])
        new_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            # The one with the matching name becomes assistant, others become user
            role = "assistant" if message.get("name") == name else "user"
            new_messages.append({**message, "role": role})  # type: ignore
        return new_messages

    async def _get_queries(self, messages: list[ChatCompletionMessageParam]):
        """
        Use the OpenAILLM to produce structured queries for the search engine.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages representing the current conversation.

        Returns:
            tuple: (Queries, tool_call, response) containing the parsed queries, the tool call details, and the raw OpenAILLM response.
        """
        response = await self.llm.structured_completion(
            messages=[{"role": "system", "content": "You are a helpful research assistant. Propose queries to help find information needed."}, *messages],
            tools=[pydantic_function_tool(Queries)],
            tool_choice="required",
            parallel_tool_calls=False,
        )

        tool_calls = response.tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            raise ValueError("No queries returned by the OpenAILLM.")

        tool_call = tool_calls[0]
        queries: Queries = tool_call.function.parsed_arguments  # type: ignore
        return queries, tool_call, response

    async def _search_engine(self, query: str):
        results = await self.search_engine.search(query=query, max_results=5, k=3)
        return results
