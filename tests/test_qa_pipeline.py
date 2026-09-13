"""QA research node and traversal tests with injected model/search responses."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core import Chunk
from lumis.pipeline.qa_research_pipeline import QAResearchPipeline, Queries

import numpy as np
import pytest
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from openai.types.chat.parsed_function_tool_call import ParsedFunction, ParsedFunctionToolCall
from pydantic import ValidationError


@pytest.fixture
def pipeline():
    return QAResearchPipeline(
        embedding=Mock(),
        search_engine=SimpleNamespace(search=AsyncMock(return_value=[])),
        llm=SimpleNamespace(structured_completion=AsyncMock(), completion=AsyncMock()),
    )


def query_result(queries=None):
    parsed = Queries(queries=["specific query"] if queries is None else queries)
    tool_call = ParsedFunctionToolCall(id="query-1", type="function", function=ParsedFunction(name="Queries", arguments=parsed.model_dump_json(), parsed_arguments=parsed))
    return ParsedChatCompletionMessage[None](role="assistant", tool_calls=[tool_call])


@pytest.mark.parametrize(
    "message",
    [QAResearchPipeline.ErrorMessages.EMPTY_QUESTION, QAResearchPipeline.ErrorMessages.UNABLE, QAResearchPipeline.ErrorMessages.UNSUCCESSFUL, QAResearchPipeline.ErrorMessages.UNABLE_TO_ANSWER],
)
def test_error_messages_are_recognized(message):
    assert QAResearchPipeline.ErrorMessages.is_error_message(message)
    assert not QAResearchPipeline.ErrorMessages.is_error_message("A useful answer")


async def test_setup_initializes_clean_question_state(pipeline):
    await pipeline.setup("Question")
    assert pipeline.state == {"question": "Question", "answer": None, "references": [], "attempt": 0}
    await pipeline.setup("New question")
    assert pipeline.state["question"] == "New question"


@pytest.mark.parametrize("question", ["", " \n\t "])
async def test_input_rejects_blank_questions(pipeline, question):
    assert await pipeline.input({"question": question}) == {"answer": pipeline.ErrorMessages.EMPTY_QUESTION}


async def test_input_accepts_valid_question_without_changing_state(pipeline):
    assert await pipeline.input({"question": "Question?"}) == {}


async def test_empty_question_stops_graph_before_model_or_search(pipeline):
    await pipeline.run(" ")
    assert pipeline.state["answer"] == pipeline.ErrorMessages.EMPTY_QUESTION
    pipeline.llm.structured_completion.assert_not_awaited()
    pipeline.search_engine.search.assert_not_awaited()


def test_build_registers_nodes_and_bounded_retry_condition(pipeline):
    pipeline._graph = Mock()
    pipeline.build()
    assert [call.args[:2] for call in pipeline.graph.add_node.call_args_list] == [("input", pipeline.input), ("generate_answer", pipeline.generate_answer)]
    input_edge, retry_edge = pipeline.graph.add_edge.call_args_list
    assert input_edge.kwargs["condition"]({"answer": None})
    assert not input_edge.kwargs["condition"]({"answer": "stop"})
    condition = retry_edge.kwargs["condition"]
    assert condition({"answer": None, "attempt": 1})
    assert condition({"answer": pipeline.ErrorMessages.UNABLE, "attempt": 1})
    assert not condition({"answer": "Answer", "attempt": 1})
    assert not condition({"answer": None, "attempt": pipeline.max_attempts})
    assert not condition({"answer": "", "attempt": pipeline.max_attempts})


def test_query_message_trims_question_and_optional_datetime(pipeline):
    assert pipeline._generate_query_messages("\n  Question\n") == [{"role": "user", "content": "Question"}]
    pipeline.include_datetime = True
    messages = pipeline._generate_query_messages("Question")
    assert messages[0]["role"] == "system"
    assert "current date and time" in messages[0]["content"]
    assert messages[-1]["content"] == "Question"


def test_answer_messages_link_tool_result_and_optional_context(pipeline):
    pipeline.include_datetime = pipeline.include_citations = True
    original = {"role": "assistant", "content": None}
    tool_call = ChatCompletionMessageFunctionToolCall(id="call-1", type="function", function={"name": "Queries", "arguments": "{}"})
    messages = pipeline._generate_answer_messages("  Question  ", original, "[]", tool_call)
    assert messages[-2] == {"role": "assistant"}
    assert messages[-1] == {"role": "tool", "content": "[]", "tool_call_id": "call-1"}
    assert any(message["role"] == "user" and message["content"] == "Question" for message in messages)
    assert any("citations" in (message.get("content") or "") for message in messages)
    assert any("current date and time" in (message.get("content") or "") for message in messages)


async def test_create_queries_extracts_first_structured_tool_call(pipeline):
    result = query_result(["first", "second"])
    pipeline.llm.structured_completion.return_value = result
    messages = [{"role": "user", "content": "Question"}]
    queries, tool_call, original = await pipeline._create_queries(messages)
    assert queries == ["first", "second"]
    assert tool_call is result.tool_calls[0]
    assert original is result
    options = pipeline.llm.structured_completion.await_args.kwargs
    assert options["tool_choice"] == "required"
    assert options["parallel_tool_calls"] is False
    assert options["messages"][-1] == messages[0]
    assert options["tools"][0]["function"]["name"] == "Queries"


@pytest.mark.parametrize("response", [None, ParsedChatCompletionMessage[None](role="assistant", tool_calls=None), ParsedChatCompletionMessage[None](role="assistant", tool_calls=[])])
async def test_create_queries_handles_missing_model_response_or_tools(pipeline, response):
    pipeline.llm.structured_completion.return_value = response
    assert await pipeline._create_queries([]) == ([], None, response)


async def test_search_combines_successes_and_ignores_failed_queries(pipeline):
    first, second = Chunk(content="first"), Chunk(content="second")
    pipeline.search_engine.search.side_effect = [[first], RuntimeError("search failed"), [second], None]
    assert await pipeline._search(["one", "two", "three", "four"]) == [first, second]
    assert [call.kwargs for call in pipeline.search_engine.search.await_args_list] == [{"query": query, "max_results": 3, "k": 3} for query in ["one", "two", "three", "four"]]


async def test_search_empty_queries_is_noop(pipeline):
    assert await pipeline._search([]) == []
    pipeline.search_engine.search.assert_not_awaited()


@pytest.mark.parametrize("metadata", [None, {"url": "https://example.com", "private": "exclude"}])
def test_clean_chunk_keeps_only_content_and_source_url(pipeline, metadata):
    item = Chunk(content="Evidence", metadata=metadata, embedding=np.array([1.0, 0.0], dtype=np.float32))
    cleaned = pipeline._clean_chunk_for_llm_consumption(item)
    assert cleaned == {"content": "Evidence", "metadata": {"url": (metadata or {}).get("url", "")}}
    json.dumps(cleaned)
    assert item.embedding is not None
    assert item.metadata == metadata


@pytest.mark.parametrize("attempt", [0, 2])
async def test_generate_answer_without_queries_retries_until_limit(pipeline, attempt):
    pipeline._create_queries = AsyncMock(return_value=([], None, None))
    result = await pipeline.generate_answer({"question": "Question", "attempt": attempt})
    assert result["attempt"] == attempt + 1
    assert result.get("answer") == (pipeline.ErrorMessages.UNABLE if attempt == 2 else None)
    pipeline.search_engine.search.assert_not_awaited()


async def test_generate_answer_without_results_reports_unsuccessful_search(pipeline):
    response = query_result()
    pipeline.llm.structured_completion.return_value = response
    result = await pipeline.generate_answer({"question": "Question", "attempt": 0})
    assert result == {"attempt": 1, "answer": pipeline.ErrorMessages.UNSUCCESSFUL}
    pipeline.llm.completion.assert_not_awaited()


async def test_generate_answer_serializes_embedded_chunks_and_returns_references(pipeline):
    item = Chunk(content="Evidence", metadata={"url": "https://example.com"}, embedding=np.array([1.0, 0.0], dtype=np.float32))
    pipeline.llm.structured_completion.return_value = query_result()
    pipeline.search_engine.search.return_value = [item]
    pipeline.llm.completion.return_value = SimpleNamespace(content="Supported answer")
    result = await pipeline.generate_answer({"question": "Question", "attempt": 0})
    assert result == {"attempt": 1, "answer": "Supported answer", "references": [item]}
    messages = pipeline.llm.completion.await_args.kwargs["messages"]
    assert json.loads(messages[-1]["content"]) == [{"content": "Evidence", "metadata": {"url": "https://example.com"}}]
    assert messages[-1]["tool_call_id"] == "query-1"


async def test_answer_model_exception_becomes_error_answer(pipeline):
    pipeline.llm.structured_completion.return_value = query_result()
    pipeline.search_engine.search.return_value = [Chunk(content="Evidence")]
    pipeline.llm.completion.side_effect = RuntimeError("provider down")
    result = await pipeline.generate_answer({"question": "Question", "attempt": 0})
    assert result == {"attempt": 1, "answer": pipeline.ErrorMessages.UNABLE_TO_ANSWER}


@pytest.mark.parametrize("content", [None, "", "   "])
async def test_empty_answer_is_represented_as_retryable_error(pipeline, content):
    pipeline.llm.structured_completion.return_value = query_result()
    pipeline.search_engine.search.return_value = [Chunk(content="Evidence")]
    pipeline.llm.completion.return_value = SimpleNamespace(content=content)
    result = await pipeline.generate_answer({"question": "Question", "attempt": 0})
    assert result["answer"] == pipeline.ErrorMessages.UNABLE_TO_ANSWER


async def test_graph_retries_failed_search_then_stops_on_answer(pipeline):
    pipeline.llm.structured_completion.return_value = query_result()
    item = Chunk(content="Evidence")
    pipeline.search_engine.search.side_effect = [[], [item]]
    pipeline.llm.completion.return_value = SimpleNamespace(content="Answer")
    finished = []
    pipeline.graph.on("finish", lambda graph: finished.append(graph.state["answer"]))
    await pipeline.run("Question")
    assert pipeline.state["attempt"] == 2
    assert pipeline.state["answer"] == "Answer"
    assert pipeline.state["references"] == [item]
    assert finished == ["Answer"]


async def test_graph_stops_after_maximum_failed_search_attempts(pipeline):
    pipeline.llm.structured_completion.return_value = query_result()
    await pipeline.run("Question")
    assert pipeline.state["attempt"] == pipeline.max_attempts
    assert pipeline.state["answer"] == pipeline.ErrorMessages.UNSUCCESSFUL
    assert pipeline.search_engine.search.await_count == pipeline.max_attempts


async def test_graph_bounds_retries_when_answer_model_returns_no_content(pipeline):
    pipeline.llm.structured_completion.return_value = query_result()
    pipeline.search_engine.search.return_value = [Chunk(content="Evidence")]
    pipeline.llm.completion.return_value = SimpleNamespace(content=None)
    await pipeline.run("Question")
    assert pipeline.state["answer"] == pipeline.ErrorMessages.UNABLE_TO_ANSWER
    assert pipeline.state["attempt"] == pipeline.max_attempts
    assert pipeline.llm.completion.await_count == pipeline.max_attempts


async def test_create_queries_validates_dictionary_parsed_arguments(pipeline):
    result = query_result()
    result.tool_calls[0].function.parsed_arguments = {"queries": ["validated query"]}
    pipeline.llm.structured_completion.return_value = result
    queries, tool_call, original = await pipeline._create_queries([])
    assert queries == ["validated query"]
    assert tool_call is result.tool_calls[0]
    assert original is result


@pytest.mark.parametrize("arguments", [None, {"queries": "not a list"}, {"wrong": []}])
async def test_create_queries_rejects_invalid_parsed_arguments(pipeline, arguments):
    result = query_result()
    result.tool_calls[0].function.parsed_arguments = arguments
    pipeline.llm.structured_completion.return_value = result
    with pytest.raises(ValidationError):
        await pipeline._create_queries([])


def test_answer_messages_strip_sdk_parsing_metadata(pipeline):
    response = query_result(["specific query"])
    tool_call = response.tool_calls[0]
    messages = pipeline._generate_answer_messages("Question", response, "[]", tool_call)
    assistant = messages[-2]
    assert assistant == {
        "role": "assistant",
        "tool_calls": [{"id": "query-1", "type": "function", "function": {"name": "Queries", "arguments": tool_call.function.arguments}}],
    }
    assert "parsed" not in assistant
    assert "parsed_arguments" not in assistant["tool_calls"][0]["function"]
    assert messages[-1]["tool_call_id"] == "query-1"
    assert response.tool_calls[0].function.parsed_arguments == Queries(queries=["specific query"])
    json.dumps(messages)


async def test_generate_answer_guards_missing_response_even_with_tool_call(pipeline):
    tool_call = query_result().tool_calls[0]
    pipeline._create_queries = AsyncMock(return_value=(["query"], tool_call, None))
    result = await pipeline.generate_answer({"question": "Question", "attempt": pipeline.max_attempts - 1})
    assert result == {"attempt": pipeline.max_attempts, "answer": pipeline.ErrorMessages.UNABLE}
    pipeline.search_engine.search.assert_not_awaited()
    pipeline.llm.completion.assert_not_awaited()
