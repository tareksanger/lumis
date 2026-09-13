"""Contracts for research source records and response reference normalization."""

from datetime import datetime, timezone
from types import SimpleNamespace

from lumis.agents.research_agent.source_models import SourceType
from lumis.agents.research_agent.types import ResearchAgentResponse

from pydantic import ValidationError
import pytest


def make_source(**overrides):
    return SourceType(**dict(title="A source", summary="Research findings", source_type="web") | overrides)


class TestSourceType:
    @pytest.mark.parametrize("kind", ["arxiv", "wiki", "web", "gemini"])
    def test_supported_source_kinds(self, kind):
        assert make_source(source_type=kind).source_type == kind

    @pytest.mark.parametrize("kind", ["other", "Wikipedia", "", None])
    def test_rejects_unsupported_source_kind(self, kind):
        with pytest.raises(ValidationError) as error:
            make_source(source_type=kind)
        assert error.value.errors()[0]["loc"] == ("source_type",)

    @pytest.mark.parametrize("missing", ["title", "summary", "source_type"])
    def test_required_source_identity(self, missing):
        data = dict(title="A source", summary="Research findings", source_type="web")
        del data[missing]
        with pytest.raises(ValidationError) as error:
            SourceType.model_validate(data)
        assert any(item["loc"] == (missing,) for item in error.value.errors())

    def test_sources_without_urls_are_supported(self):
        source = make_source(source_type="gemini")
        assert source.url is None
        assert source.credibility == 1.0
        assert make_source(url=None).url is None

    def test_metadata_and_keywords_are_not_shared_between_sources(self):
        first, second = make_source(), make_source()
        first.keywords.append("research")
        first.metadata["rank"] = 2
        assert second.keywords == []
        assert second.metadata == {}

    def test_access_time_is_created_when_source_is_instantiated(self):
        before = datetime.now()
        source = make_source()
        after = datetime.now()
        assert before <= source.timestamp_accessed <= after

    def test_json_round_trip_preserves_research_record(self):
        source = make_source(
            url="https://example.org/paper?version=2",
            keywords=["models", "evaluation"],
            credibility=0.75,
            metadata={"authors": ["Ada"], "citations": 7, "reviewed": True},
            timestamp_accessed=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
        )
        assert SourceType.model_validate_json(source.model_dump_json()) == source

    def test_can_import_source_from_attribute_based_object(self):
        record = SimpleNamespace(
            title="A paper",
            summary="Its abstract",
            source_type="arxiv",
            url="https://arxiv.org/abs/1234.5678",
            metadata={"version": 2},
        )
        source = SourceType.model_validate(record)
        assert source.title == record.title
        assert source.url == record.url
        assert source.metadata == {"version": 2}
        assert source.keywords == []


class TestResearchAgentResponse:
    def test_response_text_is_required(self):
        with pytest.raises(ValidationError) as error:
            ResearchAgentResponse()
        assert error.value.errors()[0]["loc"] == ("response",)

    def test_reference_defaults_are_independent(self):
        first = ResearchAgentResponse(response="First finding")
        second = ResearchAgentResponse(response="Second finding")
        first.references.append("https://example.org")
        assert second.references == []

    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            ("https://example.org/paper", "https://example.org/paper"),
            ("http://example.org/paper", "http://example.org/paper"),
            ("example.org/paper", "https://example.org/paper"),
            ("www.example.org/paper", "https://www.example.org/paper"),
            ("Read https://example.org/paper for details", "https://example.org/paper"),
            ('<a href="https://example.org/paper">paper</a>', "https://example.org/paper"),
            ("See www.example.org/paper now", "https://www.example.org/paper"),
            ("https://example.org/paper.,;:", "https://example.org/paper"),
            ("  https://example.org/paper  ", "https://example.org/paper"),
            ("https://example.org/paper?v=2#results", "https://example.org/paper?v=2#results"),
        ],
    )
    def test_normalizes_reference_urls(self, reference, expected):
        result = ResearchAgentResponse(response="Finding", references=[reference])
        assert result.references == [expected]

    @pytest.mark.parametrize("reference", ["", "/relative/path", "https://", "https://[invalid"])
    def test_discards_references_without_a_parseable_host(self, reference):
        result = ResearchAgentResponse(response="Finding", references=[reference])
        assert result.references == []

    def test_invalid_references_do_not_discard_other_references(self):
        references = ["example.org/first", "", "https://[invalid", "http://example.org/last"]
        original = references.copy()
        result = ResearchAgentResponse(response="Finding", references=references)
        assert result.references == ["https://example.org/first", "http://example.org/last"]
        assert references == original

    def test_serialization_contains_normalized_references_and_original_answer(self):
        result = ResearchAgentResponse(response="**Finding**\nDetails", references=["example.org."])
        assert result.model_dump() == {
            "response": "**Finding**\nDetails",
            "references": ["https://example.org"],
        }
        assert ResearchAgentResponse.model_validate_json(result.model_dump_json()) == result
