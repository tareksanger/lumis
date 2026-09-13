"""NER wrapper behavior with a fake optional spaCy LLM pipeline."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from lumis.nlp.ner.ner import Ner, relative_path

import pytest


@pytest.fixture
def assemble(monkeypatch):
    factory = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "spacy_llm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "spacy_llm.util", SimpleNamespace(assemble=factory))
    return factory


def test_default_config_uses_bundled_file(assemble):
    Ner()
    assemble.assert_called_once_with(str(relative_path))
    assert relative_path.is_file()


def test_custom_config_is_forwarded(assemble):
    Ner(config="custom.cfg")
    assemble.assert_called_once_with("custom.cfg")


def test_entities_preserve_order_labels_and_duplicates(assemble):
    ner = Ner()
    ner.nlp.return_value = SimpleNamespace(ents=[SimpleNamespace(text="Ada", label_="PERSON"), SimpleNamespace(text="Acme", label_="ORG"), SimpleNamespace(text="Ada", label_="PERSON")])
    assert ner.get_entities("input") == [("Ada", "PERSON"), ("Acme", "ORG"), ("Ada", "PERSON")]
    ner.nlp.assert_called_once_with("input")


def test_empty_entities_returns_empty_list(assemble):
    ner = Ner()
    ner.nlp.return_value = SimpleNamespace(ents=[])
    assert ner.get_entities("") == []


def test_get_doc_returns_unmodified_pipeline_result(assemble):
    ner = Ner()
    document = object()
    ner.nlp.return_value = document
    assert ner.get_doc("input") is document
    ner.nlp.assert_called_once_with("input")


def test_missing_optional_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy_llm.util", None)
    with pytest.raises(ImportError, match=r"pip install lumis-ai\[spacy\]"):
        Ner()


def test_invalid_configuration_propagates(assemble):
    assemble.side_effect = ValueError("invalid config")
    with pytest.raises(ValueError, match="invalid config"):
        Ner("invalid.cfg")


@pytest.mark.parametrize("method", ["get_doc", "get_entities"])
def test_pipeline_errors_propagate(assemble, method):
    ner = Ner()
    ner.nlp.side_effect = RuntimeError("pipeline failed")
    with pytest.raises(RuntimeError, match="pipeline failed"):
        getattr(ner, method)("input")
