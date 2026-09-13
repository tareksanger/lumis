from types import SimpleNamespace
from typing import Annotated

from lumis.core.document import Chunk, Document
from lumis.core.utils.helpers import merge_pydantic_models, serialize
from lumis.core.utils.types import _metadata_to_dict, BaseSchema
from lumis.pipeline.utils import dict_diff

import numpy as np
from pydantic import BaseModel, Field, ValidationError
import pytest


def test_document_identity_is_independent_of_content():
    first = Document(doc_id="a", content="original")
    second = Document(doc_id="a", content="edited")
    assert first == second
    assert len({first, second}) == 1
    assert first != Document(content="original")
    assert first != {"doc_id": "a"}


def test_document_validation_and_chunk_parent():
    with pytest.raises(ValidationError):
        Document()
    parent = Document(content="parent")
    vector = np.array([1.0, 2.0])
    chunk = Chunk(content="child", parent_id=parent.doc_id, embedding=vector)
    assert chunk.parent_id == parent.doc_id
    assert chunk.embedding is vector
    assert chunk.doc_id != parent.doc_id


def test_recursive_serialization():
    obj = SimpleNamespace(child={"items": [SimpleNamespace(value=3)], "pair": (1, 2)})
    assert serialize(obj) == {"child": {"items": [{"value": 3}], "pair": (1, 2)}}
    assert serialize({1, 2}) == {1, 2}


def test_model_merge_preserves_unset_values_and_inputs():
    class Settings(BaseModel):
        name: str = "default"
        options: dict = Field(default_factory=dict)
        enabled: bool = True

    first = Settings(name="custom", options={"a": 1})
    second = Settings(options={"b": 2}, enabled=False)
    merged = merge_pydantic_models(first, second)
    assert merged.model_dump() == {"name": "custom", "options": {"a": 1, "b": 2}, "enabled": False}
    assert first.options == {"a": 1}
    assert second.options == {"b": 2}


def test_schema_accepts_camel_case_and_ignores_extra_fields():
    class Person(BaseSchema):
        first_name: str

    person = Person.model_validate({"firstName": "Ada", "unknown": "ignored"})
    assert person.model_dump() == {"first_name": "Ada"}
    assert Person.model_validate(SimpleNamespace(first_name="Grace")).first_name == "Grace"


def test_schema_context_only_exposes_opted_in_fields():
    class Child(BaseSchema):
        name: Annotated[str, {"context": True}]

    class Record(BaseSchema):
        secret: str
        child: Annotated[Child, {"context": True}]
        tags: Annotated[list[str], {"context": True}]
        attributes: Annotated[dict, {"context": True}]

    record = Record(secret="DO NOT SHOW", child=Child(name="Ada"), tags=["one", "two"], attributes={"first_key": 3})
    context = record.to_context_str()
    assert "DO NOT SHOW" not in context
    for fragment in ["Child:", "Name: Ada", "- one", "- two", "First Key: 3"]:
        assert fragment in context
    assert _metadata_to_dict([{"context": False}, ("context", True), "ignored"]) == {"context": True}


def test_dict_diff_reports_nested_changes_additions_and_removals():
    before = {"nested": {"a": 1, "b": 2}, "gone": 7, "same": [1]}
    after = {"nested": {"a": 4, "b": 2}, "new": 8, "same": [1]}
    assert dict_diff(before, after) == {"nested": {"a": 4}, "gone": None, "new": 8}
    assert dict_diff(before, before) == {}
