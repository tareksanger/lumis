"""Backward-compatible OpenAI request inputs normalized at the SDK boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import Timeout, omit
from openai.types import Reasoning
import pytest

from lumis.llm.openai_llm import OpenAILLM


@pytest.fixture
def sdk_client():
    return SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(usage=None))),
        images=SimpleNamespace(generate=AsyncMock(return_value=object())),
    )


@pytest.fixture
def llm(sdk_client):
    return OpenAILLM(client=sdk_client)


@pytest.mark.parametrize(
    ("retention", "expected"),
    [
        ("in-memory", "in_memory"),
        ("in_memory", "in_memory"),
        ("24h", "24h"),
        (None, None),
        (omit, omit),
    ],
)
async def test_cache_retention_normalizes_legacy_spelling_only(llm, sdk_client, retention, expected):
    await llm.response(input="Question", prompt_cache_retention=retention)
    assert sdk_client.responses.create.await_args.kwargs["prompt_cache_retention"] == expected


async def test_legacy_httpx_timeout_preserves_individual_limits(llm, sdk_client):
    original = httpx.Timeout(connect=1.0, read=None, write=3.0, pool=4.0)
    await llm.generate_image(prompt="Scene", timeout=original)
    normalized = sdk_client.images.generate.await_args.kwargs["timeout"]
    assert isinstance(normalized, Timeout)
    assert (normalized.connect, normalized.read, normalized.write, normalized.pool) == (1.0, None, 3.0, 4.0)
    assert original.as_dict() == {"connect": 1.0, "read": None, "write": 3.0, "pool": 4.0}


async def test_sdk_timeout_instance_passes_through_unchanged(llm, sdk_client):
    timeout = Timeout(connect=1.0, read=2.0, write=3.0, pool=None)
    await llm.generate_image(prompt="Scene", timeout=timeout)
    assert sdk_client.images.generate.await_args.kwargs["timeout"] is timeout


@pytest.mark.parametrize("reasoning", [{"effort": "low"}, {"effort": "high", "summary": "auto"}, None, omit])
async def test_reasoning_request_dict_and_sentinels_pass_through(llm, sdk_client, reasoning):
    await llm.response(input="Question", reasoning=reasoning)
    assert sdk_client.responses.create.await_args.kwargs["reasoning"] is reasoning


async def test_reasoning_model_becomes_sdk_request_dict_without_unset_fields(llm, sdk_client):
    model = Reasoning(effort="high", summary="auto", context="all_turns", mode="standard")
    await llm.response(input="Question", reasoning=model)
    assert sdk_client.responses.create.await_args.kwargs["reasoning"] == {
        "effort": "high",
        "summary": "auto",
        "context": "all_turns",
        "mode": "standard",
    }
    assert model.generate_summary is None
    assert model.effort == "high"
