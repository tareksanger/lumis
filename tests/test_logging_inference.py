import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core.common.coloured_logger import ColorPrinter
from lumis.core.common.logger_mixin import LoggerMixin
from lumis.core.utils.inference import generate_response, generate_summary

import pytest


def test_color_is_deterministic_and_print_resets(capsys):
    assert ColorPrinter.hash_to_color("agent") == ColorPrinter.hash_to_color("agent")
    assert 0 <= ColorPrinter.hash_to_color("agent") <= 255
    ColorPrinter.print("hello", "agent")
    assert capsys.readouterr().out == f"{ColorPrinter.generate_unique_color('agent')}hello\033[0m\n"


def test_logger_injection_and_exception_details():
    logger = Mock()
    instance = LoggerMixin(logger)
    try:
        raise ValueError("failure")
    except ValueError as exc:
        instance.log_exception(exc, logging.WARNING)
    level, message = logger.log.call_args.args
    assert level == logging.WARNING
    assert "failure" in message and "Traceback" in message
    assert LoggerMixin().logger.name == "LoggerMixin"


@pytest.mark.parametrize("response,expected", [(None, None), (SimpleNamespace(content=None), None), (SimpleNamespace(content="answer"), "answer")])
async def test_generate_response_preserves_optional_content(response, expected):
    llm = SimpleNamespace(completion=AsyncMock(return_value=response))
    messages = [{"role": "user", "content": "question"}]
    assert await generate_response(llm, messages, temperature=0) == expected
    llm.completion.assert_awaited_once_with(messages=messages, temperature=0)


async def test_summary_supplies_source_text():
    llm = SimpleNamespace(completion=AsyncMock(return_value=SimpleNamespace(content="summary")))
    assert await generate_summary(llm, "source text") == "summary"
    messages = llm.completion.call_args.kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "source text"}
    assert messages[0]["role"] == "developer"


async def test_generation_failure_is_not_silently_dropped():
    llm = SimpleNamespace(completion=AsyncMock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        await generate_response(llm, [])
