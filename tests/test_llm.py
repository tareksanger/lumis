from __future__ import annotations

import math
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel

from lumis.llm.base_llm import BaseLLM


# ---------------------------------------------------------------------------
# BaseLLM
# ---------------------------------------------------------------------------


class _ConcreteLLM(BaseLLM):
    """Minimal concrete subclass so we can instantiate BaseLLM."""

    pass


class TestBaseLLM:
    def test_initial_token_count_is_empty(self):
        assert _ConcreteLLM().token_count == Counter()

    def test_verbose_defaults_false(self):
        assert _ConcreteLLM().verbose is False

    def test_verbose_flag(self):
        assert _ConcreteLLM(verbose=True).verbose is True

    def test_default_middleware_registered(self):
        llm = _ConcreteLLM()
        assert len(llm._middlewares) == 1
        assert llm._middlewares[0].__name__ == "_count_tokens"

    def test_add_middleware_appends(self):
        llm = _ConcreteLLM()

        def noop(r):
            return r

        llm.add_middleware(noop)
        assert llm._middlewares[-1] is noop

    async def test_apply_middlewares_sync(self):
        llm = _ConcreteLLM()
        llm._middlewares = []
        llm.add_middleware(lambda x: x * 2)
        assert await llm._apply_middlewares(5) == 10

    async def test_apply_middlewares_async(self):
        llm = _ConcreteLLM()
        llm._middlewares = []

        async def add_one(x):
            return x + 1

        llm.add_middleware(add_one)
        assert await llm._apply_middlewares(5) == 6

    async def test_apply_middlewares_chains_in_order(self):
        llm = _ConcreteLLM()
        llm._middlewares = []
        llm.add_middleware(lambda x: x * 2)  # 3 → 6

        async def add_one(x):
            return x + 1  # 6 → 7

        llm.add_middleware(add_one)
        assert await llm._apply_middlewares(3) == 7

    async def test_apply_middlewares_swallows_errors(self):
        """A failing middleware is logged but does not crash the pipeline."""
        llm = _ConcreteLLM()
        llm._middlewares = []

        def bad(x):
            raise ValueError("boom")

        llm.add_middleware(bad)
        llm.add_middleware(lambda x: x + 1)

        # bad raises → response stays 5 → next middleware receives 5 → returns 6
        assert await llm._apply_middlewares(5) == 6


# ---------------------------------------------------------------------------
# OpenAILLM
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    @pytest.fixture
    def llm(self):
        from lumis.llm.openai_llm import OpenAILLM

        return OpenAILLM(client=MagicMock())

    # -- _count_tokens --

    def test_count_tokens_updates_counter(self, llm):
        usage = MagicMock()
        usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        result = llm._count_tokens(MagicMock(usage=usage))

        assert result.usage is usage  # pass-through
        assert llm.token_count["prompt_tokens"] == 10
        assert llm.token_count["completion_tokens"] == 5
        assert llm.token_count["total_tokens"] == 15

    def test_count_tokens_accumulates_across_calls(self, llm):
        def _completion(prompt, comp):
            u = MagicMock()
            u.model_dump.return_value = {
                "prompt_tokens": prompt,
                "completion_tokens": comp,
                "total_tokens": prompt + comp,
            }
            return MagicMock(usage=u)

        llm._count_tokens(_completion(10, 5))
        llm._count_tokens(_completion(20, 8))

        assert llm.token_count["prompt_tokens"] == 30
        assert llm.token_count["completion_tokens"] == 13

    def test_count_tokens_no_usage_is_noop(self, llm):
        llm._count_tokens(MagicMock(usage=None))
        assert llm.token_count == Counter()

    def test_count_tokens_flattens_nested_usage(self, llm):
        usage = MagicMock()
        usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "details": {"cached_tokens": 3},
        }
        llm._count_tokens(MagicMock(usage=usage))
        assert llm.token_count["details.cached_tokens"] == 3

    # -- _parse_tool_arguments --

    def test_parse_tool_arguments_json_string(self, llm):
        assert llm._parse_tool_arguments('{"a": 1}', "fn") == {"a": 1}

    def test_parse_tool_arguments_empty_string(self, llm):
        assert llm._parse_tool_arguments("", "fn") == {}

    def test_parse_tool_arguments_mapping(self, llm):
        assert llm._parse_tool_arguments({"a": 1}, "fn") == {"a": 1}

    def test_parse_tool_arguments_none(self, llm):
        assert llm._parse_tool_arguments(None, "fn") == {}

    def test_parse_tool_arguments_invalid_json_returns_none(self, llm):
        assert llm._parse_tool_arguments("not json", "fn") is None

    # -- _invoke_tool --

    async def test_invoke_tool_sync(self, llm):
        def double(x):
            return x * 2

        assert await llm._invoke_tool("double", {"x": 5}, {"double": double}) == 10

    async def test_invoke_tool_async(self, llm):
        async def add(x, y):
            return x + y

        assert await llm._invoke_tool("add", {"x": 3, "y": 4}, {"add": add}) == 7

    async def test_invoke_tool_missing_returns_error_string(self, llm):
        result = await llm._invoke_tool("nope", {}, {})
        assert result == "No such tool function: nope"

    async def test_invoke_tool_exception_returns_message(self, llm):
        def bad():
            raise ValueError("broken")

        result = await llm._invoke_tool("bad", {}, {"bad": bad})
        assert result == "broken"

    # -- _has_tool_calls --

    def test_has_tool_calls_true(self, llm):
        assert llm._has_tool_calls(MagicMock(tool_calls=[MagicMock()])) is True

    def test_has_tool_calls_false_when_empty(self, llm):
        assert llm._has_tool_calls(MagicMock(tool_calls=[])) is False

    def test_has_tool_calls_false_when_none(self, llm):
        assert llm._has_tool_calls(MagicMock(tool_calls=None)) is False

    # -- perplexity --

    def test_perplexity_no_content(self, llm):
        assert llm.perplexity(MagicMock(content=None)) == 0.0

    def test_perplexity_empty_content(self, llm):
        assert llm.perplexity(MagicMock(content=[])) == 0.0

    def test_perplexity_all_none_logprobs(self, llm):
        content = [MagicMock(logprob=None), MagicMock(logprob=None)]
        assert llm.perplexity(MagicMock(content=content)) == 0.0

    def test_perplexity_known_value(self, llm):
        # avg logprob = -1.0  →  perplexity = e^1 ≈ 2.718
        content = [MagicMock(logprob=-1.0), MagicMock(logprob=-1.0)]
        assert llm.perplexity(MagicMock(content=content)) == pytest.approx(math.e, rel=1e-3)

    def test_perplexity_floored_at_one(self, llm):
        # logprob = 0  →  exp(0) = 1.0  →  max(1.0, 1.0) = 1.0
        content = [MagicMock(logprob=0.0)]
        assert llm.perplexity(MagicMock(content=content)) == 1.0

    # -- confidence_score / token_abstraction --

    def test_confidence_score_certain(self, llm):
        # logprob = 0  →  prob = 1  →  confidence = 100
        assert llm.confidence_score(MagicMock(token="x", logprob=0.0)) == pytest.approx(100.0)

    def test_confidence_score_half(self, llm):
        # logprob = ln(0.5)  →  prob = 0.5  →  confidence ≈ 50
        assert llm.confidence_score(MagicMock(token="x", logprob=math.log(0.5))) == pytest.approx(50.0, abs=0.1)

    def test_token_abstraction_returns_tuple(self, llm):
        lp = MagicMock(token="hi", logprob=-0.5)
        token, logprob, confidence = llm.token_abstraction(lp)

        assert token == "hi"
        assert logprob == -0.5
        assert confidence == pytest.approx(np.round(np.exp(-0.5) * 100, 2))

    # -- completion (mocked client) --

    async def test_completion_single(self, llm):
        msg = MagicMock(content="hi", role="assistant")
        completion = MagicMock(choices=[MagicMock(message=msg)], usage=None)
        llm.client.chat.completions.create = AsyncMock(return_value=completion)

        result = await llm.completion(messages=[{"role": "user", "content": "hello"}])
        assert result is msg

    async def test_completion_multiple(self, llm):
        messages = [MagicMock(content=f"r{i}", role="assistant") for i in range(3)]
        completion = MagicMock(choices=[MagicMock(message=m) for m in messages], usage=None)
        llm.client.chat.completions.create = AsyncMock(return_value=completion)

        result = await llm.completion(messages=[{"role": "user", "content": "hi"}], n=3)
        assert len(result) == 3
        assert result[0] is messages[0]

    async def test_completion_no_choices_raises(self, llm):
        completion = MagicMock(choices=[], usage=None)
        llm.client.chat.completions.create = AsyncMock(return_value=completion)

        with pytest.raises(ValueError, match="No choices"):
            await llm.completion(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# OllamaLLM
# ---------------------------------------------------------------------------


class TestOllamaLLM:
    @pytest.fixture
    def llm(self):
        from lumis.llm.ollama_llm import OllamaLLM

        return OllamaLLM(client=MagicMock())

    # -- factory methods --

    def test_from_client_preserves_client(self):
        from lumis.llm.ollama_llm import OllamaLLM

        mock = MagicMock()
        assert OllamaLLM.from_client(mock).client is mock

    def test_from_host_passes_url(self):
        from lumis.llm.ollama_llm import OllamaLLM

        with patch("lumis.llm.ollama_llm.AsyncClient") as MockClient:
            llm = OllamaLLM.from_host("http://localhost:11434")
            MockClient.assert_called_with(host="http://localhost:11434")
            assert llm.client is MockClient.return_value

    # -- _count_tokens --

    def test_count_tokens_parses_ollama_fields(self, llm):
        # ChatResponse / GenerateResponse are dict-like; use a plain dict
        llm._count_tokens({"prompt_eval_count": 12, "eval_count": 8})

        assert llm.token_count["prompt_tokens"] == 12
        assert llm.token_count["completion_tokens"] == 8
        assert llm.token_count["total_tokens"] == 20

    def test_count_tokens_missing_fields_is_noop(self, llm):
        llm._count_tokens({})
        assert llm.token_count == Counter()

    # -- completion --

    async def test_completion_returns_message(self, llm):
        msg = MagicMock(content="pong", role="assistant")
        response = MagicMock()
        response.message = msg
        response.get = MagicMock(return_value=None)  # _count_tokens calls .get()
        llm.client.chat = AsyncMock(return_value=response)

        result = await llm.completion(messages=[{"role": "user", "content": "ping"}])
        assert result is msg

    async def test_completion_no_message_raises(self, llm):
        response = MagicMock()
        response.message = None
        response.get = MagicMock(return_value=None)
        llm.client.chat = AsyncMock(return_value=response)

        with pytest.raises(ValueError, match="No message"):
            await llm.completion(messages=[{"role": "user", "content": "hi"}])

    # -- structured_response --

    async def test_structured_response_parses_pydantic_model(self, llm):
        class Item(BaseModel):
            name: str
            value: int

        response = MagicMock()
        response.response = '{"name": "widget", "value": 42}'
        response.get = MagicMock(return_value=None)
        llm.client.generate = AsyncMock(return_value=response)

        result = await llm.structured_response(prompt="describe", format=Item)

        assert isinstance(result, Item)
        assert result.name == "widget"
        assert result.value == 42

    async def test_structured_response_empty_response_raises(self, llm):
        class Item(BaseModel):
            name: str

        response = MagicMock()
        response.response = ""  # falsy
        response.get = MagicMock(return_value=None)
        llm.client.generate = AsyncMock(return_value=response)

        with pytest.raises(ValueError, match="No response text"):
            await llm.structured_response(prompt="describe", format=Item)
