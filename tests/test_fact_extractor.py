"""Fact rules exercised with tiny explicit dependency trees, no NLP models."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from lumis.nlp.information_extraction.FactExtractor import FactExtractor

import pytest


def token(text, **kwargs):
    defaults = dict(text=text, dep_="", ent_type_="", pos_="", lemma_="", tag_="", i=0, children=[], lefts=[], rights=[], head=SimpleNamespace(ent_type_=""))
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class Doc(list):
    def __init__(self, tokens=()):
        super().__init__(tokens)
        self.sents = [list(tokens)]
        self._ = SimpleNamespace(facts=None)


@pytest.fixture
def extractor(monkeypatch):
    fake_doc = SimpleNamespace(has_extension=Mock(return_value=False), set_extension=Mock())
    nlp = Mock()
    nlp.vocab = object()
    matcher = Mock(return_value=[])
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=Mock(return_value=nlp)))
    monkeypatch.setitem(sys.modules, "spacy.matcher", SimpleNamespace(Matcher=Mock(return_value=matcher)))
    monkeypatch.setitem(sys.modules, "spacy.tokens", SimpleNamespace(Doc=fake_doc))
    return FactExtractor(model="fake_model")


def svo_doc():
    subject = token("Ada", dep_="nsubj")
    obj = token("tools", dep_="dobj")
    verb = token("builds", pos_="VERB", lemma_="build", lefts=[subject], rights=[obj])
    return Doc([subject, verb, obj])


def test_initialization_loads_requested_model_and_registers_extension(extractor):
    sys.modules["spacy"].load.assert_called_once_with("fake_model")
    sys.modules["spacy.tokens"].Doc.set_extension.assert_called_once_with("facts", default=[])
    assert extractor.matcher.add.call_args.args[0] == "TITLE_PATTERN"


def test_existing_extension_is_not_registered_again(extractor):
    doc = sys.modules["spacy.tokens"].Doc
    doc.has_extension.return_value = True
    doc.set_extension.reset_mock()
    FactExtractor()
    doc.set_extension.assert_not_called()


def test_missing_spacy_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)
    with pytest.raises(ImportError, match=r"pip install lumis-ai\[spacy\]"):
        FactExtractor()


def test_extract_facts_formats_deduplicates_and_attaches_list(extractor):
    doc = svo_doc()
    doc.sents *= 2
    extractor.nlp.return_value = doc
    assert extractor.extract_facts("Ada builds tools") == ["Ada build tools"]
    assert doc._.facts == ["Ada build tools"]


def test_process_parses_once_and_returns_same_document(extractor):
    doc = svo_doc()
    extractor.nlp.return_value = doc
    assert extractor.process("input") is doc
    extractor.nlp.assert_called_once_with("input")
    assert doc._.facts == ["Ada build tools"]


def test_empty_document_has_no_facts(extractor):
    doc = Doc()
    extractor.nlp.return_value = doc
    assert extractor.extract_facts("") == []
    assert doc._.facts == []


def test_compound_noun_preserves_textual_order(extractor):
    root = token("company", i=3, children=[token("large", dep_="amod", i=0), token("software", dep_="compound", i=2), token("the", dep_="det", i=1)])
    assert extractor.get_compound_noun(root) == "large software company"


def test_svo_handles_prepositional_objects(extractor):
    subject = token("Acme", dep_="nsubj")
    partner = token("Beta", dep_="pobj")
    prep = token("with", dep_="prep", children=[partner])
    verb = token("partnered", pos_="VERB", lemma_="partner", lefts=[subject], rights=[prep])
    assert extractor.extract_svo(Doc([verb])) == [("Acme", "partner with", "Beta")]


def test_passive_svo_places_agent_before_verb_and_patient_after(extractor):
    patient = token("Acme", dep_="nsubjpass")
    agent = token("Ada", dep_="pobj")
    by = token("by", dep_="agent", children=[agent])
    verb = token("founded", pos_="VERB", lemma_="found", dep_="ROOT", tag_="VBN", lefts=[patient], rights=[by], children=[patient, by])
    assert extractor.extract_svo(Doc([verb])) == [("Ada", "found", "Acme")]


def test_svo_requires_subject_and_object(extractor):
    assert extractor.extract_svo(Doc([token("runs", pos_="VERB", lemma_="run")])) == []


def test_title_relation_from_apposition(extractor):
    org = token("Acme", ent_type_="ORG")
    title = token("CEO", dep_="appos", head=token("Ada", ent_type_="PERSON"), children=[token("of", dep_="prep", children=[org])])
    assert extractor.extract_title_relations(Doc([title])) == ["Ada is CEO of Acme"]


def test_title_matcher_joins_names_and_skips_incomplete_match(extractor):
    tokens = [
        token("Ada", ent_type_="PERSON"),
        token("Lovelace", ent_type_="PERSON"),
        token("chief", pos_="NOUN"),
        token("scientist", pos_="NOUN"),
        token("Acme", ent_type_="ORG"),
        token("Labs", ent_type_="ORG"),
    ]
    extractor.matcher.return_value = [(1, 0, 6), (1, 0, 2)]
    assert extractor.extract_title_relations(Doc(tokens)) == ["Ada Lovelace is chief scientist of Acme Labs"]


@pytest.mark.parametrize("lemma", ["announce", "declare", "reveal"])
def test_partnership_requires_announcement_subject_and_partner(extractor, lemma):
    subject = token("Acme", dep_="nsubj", ent_type_="ORG")
    partner = token("Beta", ent_type_="ORG")
    obj = token("partnership", dep_="dobj", children=[token("with", dep_="prep", children=[partner])])
    verb = token("announced", lemma_=lemma, children=[subject, obj])
    assert extractor.extract_partnerships(Doc([verb])) == ["Acme announced partnership with Beta"]
    verb.children = [obj]
    assert extractor.extract_partnerships(Doc([verb])) == []


def test_foundation_requires_organization_and_person_agent(extractor):
    org = token("Acme", dep_="nsubjpass", ent_type_="ORG")
    founder = token("Ada", ent_type_="PERSON")
    verb = token("founded", lemma_="found", dep_="ROOT", children=[org, token("by", dep_="agent", children=[founder])])
    assert extractor.extract_foundations(Doc([verb])) == ["Acme was founded by Ada"]
    founder.ent_type_ = "ORG"
    assert extractor.extract_foundations(Doc([verb])) == []
