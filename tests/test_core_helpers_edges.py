"""Model-specific tokenizers and nested context/serialization edge contracts."""

from types import SimpleNamespace
from typing import Annotated
from unittest.mock import Mock
import warnings

from lumis.core.utils import helpers
from lumis.core.utils.types import _metadata_to_dict, BaseSchema

from pydantic import BaseModel
import pytest


@pytest.fixture
def tokenizer_loader(monkeypatch):
    # Isolate the cache so neither fake encodings nor previous suite calls leak.
    monkeypatch.setattr(helpers, "_tokenizer_instances", {})
    loader = Mock(side_effect=lambda model: SimpleNamespace(model=model))
    monkeypatch.setattr(helpers.tiktoken, "encoding_for_model", loader)
    return loader


def test_tokenizer_cache_is_keyed_by_model(tokenizer_loader):
    first = helpers.get_tokenizer("model-first")
    second = helpers.get_tokenizer("model-second")
    assert first.model == "model-first"
    assert second.model == "model-second"
    assert first is not second
    assert helpers.get_tokenizer("model-first") is first
    assert helpers.get_tokenizer("model-second") is second
    assert [entry.args for entry in tokenizer_loader.call_args_list] == [("model-first",), ("model-second",)]


def test_unknown_model_error_is_not_masked_by_cached_other_model(tokenizer_loader):
    helpers.get_tokenizer("known-model")
    tokenizer_loader.side_effect = KeyError("unknown-model")
    with pytest.raises(KeyError, match="unknown-model"):
        helpers.get_tokenizer("unknown-model")


def test_failed_tokenizer_lookup_can_be_retried(tokenizer_loader):
    encoding = object()
    tokenizer_loader.side_effect = [KeyError("Unknown model"), encoding]
    with pytest.raises(KeyError):
        helpers.get_tokenizer("retry-model")
    assert helpers.get_tokenizer("retry-model") is encoding
    assert helpers.get_tokenizer("retry-model") is encoding
    assert tokenizer_loader.call_count == 2


def test_serialize_nested_pydantic_models_without_deprecated_dict_api():
    class Child(BaseModel):
        value: int

    class Parent(BaseModel):
        children: list[Child]

    model = Parent(children=[Child(value=3)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert helpers.serialize(model) == {"children": [{"value": 3}]}


def test_serialize_prefers_custom_dict_contract_over_internal_attributes():
    class Record:
        def __init__(self):
            self.private = "not serialized"

        def dict(self):
            return {"exposed": SimpleNamespace(value=7)}

    assert helpers.serialize(Record()) == {"exposed": {"value": 7}}


@pytest.mark.parametrize("value", [None, True, 42, "text", b"bytes"])
def test_serialize_leaves_atomic_values_unchanged(value):
    assert helpers.serialize(value) is value


def test_metadata_normalizes_keys_and_applies_entries_in_order():
    assert _metadata_to_dict(
        [
            {1: "number-key", "context": False},
            ("context", True),
            {"context": "latest"},
            None,
            "ignored",
            42,
        ]
    ) == {"1": "number-key", "context": "latest"}
    assert _metadata_to_dict([]) == {}


class ChildContext(BaseSchema):
    first_name: Annotated[str, {"context": True}]
    hidden: str = "PRIVATE VALUE"


class NestedContext(BaseSchema):
    people: Annotated[list[ChildContext], {"context": True}]
    pair: Annotated[tuple[str, int], ("context", True)]
    records: Annotated[dict[str, ChildContext | int], {"context": True}]


def test_nested_model_lists_tuples_and_dicts_respect_context_visibility():
    model = NestedContext(
        people=[ChildContext(first_name="Ada"), ChildContext(first_name="Grace")],
        pair=("one", 2),
        records={"owner": ChildContext(first_name="Lin"), "item_count": 4},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        context = model.to_context_str()
    for fragment in ["People:", "First Name: Ada", "First Name: Grace", "Pair:", "- one", "- 2", "Records:", "Owner:", "First Name: Lin", "Item Count: 4"]:
        assert fragment in context
    assert "PRIVATE VALUE" not in context
    assert "Hidden" not in context


def test_custom_context_delimiter_and_depth_apply_to_nested_records():
    model = NestedContext(people=[ChildContext(first_name="Ada")], pair=("one", 2), records={"owner": ChildContext(first_name="Lin")})
    context = model.to_context_str(deliminator=" | ", depth=1)
    assert context.startswith("\tPeople:")
    assert " | " in context
    assert "\tPair:" in context
    assert "\tRecords:" in context
    assert "First Name: Lin" in context


def test_context_without_opted_in_fields_is_empty():
    class PrivateRecord(BaseSchema):
        secret: str

    assert PrivateRecord(secret="private").to_context_str() == ""
