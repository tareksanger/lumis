from types import SimpleNamespace
from unittest.mock import AsyncMock

from lumis.pipeline.prompt_refinement_pipeline import PromptRefinementPipeline, Rewrite, RewriteResponse, TextConcisenessAnalyzer

import pytest


@pytest.mark.parametrize("level,expect_rewrite", [("NO MOD", False), ("SOME MOD", True), ("HEAVY MOD", True)])
async def test_refinement_full_pipeline_preserves_rewrite_and_history(monkeypatch, level, expect_rewrite):
    first = Rewrite(rewrite="first", information_added="NO", assumptions=None)
    best = Rewrite(rewrite="best", information_added="NO", assumptions=None)
    result = RewriteResponse(mod_level=level, reason="test", rewrites=[first, best])
    llm = SimpleNamespace(structured_completion=AsyncMock(return_value=SimpleNamespace(parsed=result)))
    monkeypatch.setattr(TextConcisenessAnalyzer, "composite_readability_score", lambda text: {"first": 1, "best": 5}[text])
    pipeline = PromptRefinementPipeline(llm=llm)
    history = [{"role": "user", "content": "previous context"}]
    await pipeline.run(query="question", conversation_history=history)
    assert pipeline.state["rewrite"] == (best if expect_rewrite else None)
    request = llm.structured_completion.call_args.kwargs
    assert request["response_format"] is RewriteResponse
    assert "question" in request["messages"][0]["content"]
    assert "previous context" in request["messages"][0]["content"]
    assert history == [{"role": "user", "content": "previous context"}]


async def test_missing_parsed_content_and_no_rewrite_candidates():
    llm = SimpleNamespace(structured_completion=AsyncMock(return_value=SimpleNamespace(parsed=None)))
    pipeline = PromptRefinementPipeline(llm=llm)
    await pipeline.run(query="question")
    assert pipeline.state["rewrite"] is None
    empty = RewriteResponse(mod_level="SOME MOD", reason="none", rewrites=None)
    assert await pipeline.extract_best_rewrite({"rewrite_response": empty}) == {"rewrite": None}
