"""Static caller contracts for agents, provider-aware pipelines, and nodes."""

from collections import Counter
from collections.abc import Awaitable
from typing import assert_type

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage

from lumis.agents.base.base_agent import BaseAgent
from lumis.agents.react_agent import ReactAgent, ReActThought
from lumis.core.utils.types import BaseSchema
from lumis.llm.base_llm import BaseLLM
from lumis.llm.ollama_llm import OllamaLLM
from lumis.llm.openai_llm import OpenAILLM
from lumis.pipeline.graph import StateProtocol
from lumis.pipeline.pipeline import Pipeline
from lumis.pipeline.nodes.llm_structured_node import LLMStructuredNode
from lumis.pipeline.prompt_refinement_pipeline import PromptRefinementPipeline
from lumis.pipeline.qa_research_pipeline import QAResearchPipeline


class Decision(BaseSchema):
    answer: str


class LocalPipeline(Pipeline[StateProtocol, str, OllamaLLM]):
    def build(self) -> None:
        pass

    async def setup(self) -> None:
        pass

    async def generate(self) -> None:
        from ollama import GenerateResponse

        assert_type(await self._require_llm().response("question"), GenerateResponse)


async def check_react_contract(agent: ReactAgent, base: BaseAgent[str], messages: list[ChatCompletionMessageParam]) -> None:
    assert_type(agent.llm, OpenAILLM)
    assert_type(base.llm, OpenAILLM)
    response = await agent.llm.structured_completion(response_format=ReActThought, messages=messages)
    assert_type(response, ParsedChatCompletionMessage[ReActThought])
    assert_type(response.parsed, ReActThought | None)
    if response.parsed is not None:
        assert_type(response.parsed.thought, str)
    assert_type(await agent.initialize(messages), ReActThought | None)
    assert_type(await agent.step(response.parsed), ReActThought | None)
    assert_type(agent.token_count, Counter[str])


def check_pipeline_contract(generic: Pipeline[StateProtocol, str], local: LocalPipeline, refinement: PromptRefinementPipeline, qa: QAResearchPipeline, llm: OpenAILLM) -> None:
    assert_type(generic.llm, BaseLLM | None)
    assert_type(local.llm, OllamaLLM | None)
    assert_type(refinement.llm, OpenAILLM | None)
    assert_type(qa.llm, OpenAILLM | None)
    node = LLMStructuredNode(llm=llm, response_format=Decision, state_key="answer")
    assert_type(node, LLMStructuredNode[Decision])
    assert_type(node.response_format, type[Decision])


def check_finish_callbacks(llm: OpenAILLM) -> None:
    async def async_finish() -> str | None:
        return None

    def deferred_finish() -> Awaitable[str | None]:
        return async_finish()

    class Finish:
        async def __call__(self) -> str | None:
            return None

    ReactAgent(llm=llm, finish_condition_callback=async_finish)
    ReactAgent(llm=llm, finish_condition_callback=deferred_finish)
    ReactAgent(llm=llm, finish_condition_callback=Finish())
