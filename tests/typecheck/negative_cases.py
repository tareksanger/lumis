"""Deliberately invalid examples checked for rejection by test_static_contracts.py."""

from lumis.agents.react_agent import ReactAgent, ReActThought
from lumis.llm.base_llm import BaseLLM
from lumis.llm.openai_llm import OpenAILLM


def string_only_middleware(response: str) -> str:
    return response.upper()


async def rejected_examples(llm: OpenAILLM, base: BaseLLM) -> None:
    base.add_middleware(string_only_middleware)  # expected-error: reportArgumentType
    ReactAgent(llm=base)  # expected-error: reportArgumentType
    message = await llm.structured_completion(ReActThought)
    message.parsed.thought  # expected-error: reportOptionalMemberAccess
    wrong_schema: str = message.parsed  # expected-error: reportAssignmentType
    multiple = await llm.structured_completion(ReActThought, n=2)
    multiple.parsed  # expected-error: reportAttributeAccessIssue
