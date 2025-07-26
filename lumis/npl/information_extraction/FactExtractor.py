from typing import Optional, Set, Tuple

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token


# TODO: Convert this to a spacy factory
class FactExtractor:
    def __init__(self, model: str = "en_core_web_lg") -> None:
        """
        Initialize the FactExtractor with a spaCy NLP pipeline.
        :param model: The name of the spaCy model to load.
        """
        # spaCy handles caching and singleton behavior automatically
        self.nlp: Language = spacy.load(model)
        self.matcher: Matcher = Matcher(self.nlp.vocab)
        self._init_patterns()

        # Register custom extension if not already registered
        if not Doc.has_extension("facts"):
            Doc.set_extension("facts", default=[])

    def _init_patterns(self) -> None:
        """
        Initialize Matcher patterns for specific fact extraction.
        """
        # Pattern for "X is the Y of Z"
        title_pattern = [
            {"ENT_TYPE": "PERSON", "OP": "+"},  # Person's name
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "the"},
            {"POS": "NOUN", "OP": "+"},  # Title
            {"LOWER": "of"},
            {"ENT_TYPE": "ORG", "OP": "+"},  # Organization
        ]
        self.matcher.add("TITLE_PATTERN", [title_pattern])

    def extract_title_relations(self, doc: Doc) -> list[str]:
        """
        Extract facts where a person holds a title at an organization.
        :param doc: A spaCy Doc object.
        :return: A list of fact strings.
        """
        facts: list[str] = []
        for token in doc:
            # Look for appositions indicating a title (e.g., "CEO")
            if token.dep_ == "appos" and token.head.ent_type_ == "PERSON":
                person: str = token.head.text
                title: str = token.text
                # Check for prepositional modifier 'of' leading to an organization
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "of":
                        for org in child.children:
                            if org.ent_type_ == "ORG":
                                fact: str = f"{person} is {title} of {org.text}"
                                facts.append(fact)
        # Use Matcher patterns
        matches: list[Tuple[int, int, int]] = self.matcher(doc)
        for match_id, start, end in matches:
            span: Span = doc[start:end]
            person_tokens: list[str] = []
            title_tokens: list[str] = []
            organization_tokens: list[str] = []
            for token in span:
                if token.ent_type_ == "PERSON":
                    person_tokens.append(token.text)
                elif token.ent_type_ == "ORG":
                    organization_tokens.append(token.text)
                elif token.pos_ == "NOUN":
                    title_tokens.append(token.text)
            if person_tokens and title_tokens and organization_tokens:
                person: str = " ".join(person_tokens)
                title: str = " ".join(title_tokens)
                organization: str = " ".join(organization_tokens)
                fact: str = f"{person} is {title} of {organization}"
                facts.append(fact)
        return facts

    def extract_partnerships(self, doc: Doc) -> list[str]:
        """
        Extract partnership announcements from the text.
        :param doc: A spaCy Doc object.
        :return: A list of fact strings.
        """
        facts: list[str] = []
        for token in doc:
            if token.lemma_ in ["announce", "declare", "reveal"]:
                subject: Optional[str] = None
                partner: Optional[str] = None
                # Find the subject (who announced)
                for child in token.children:
                    if child.dep_ == "nsubj" and child.ent_type_ in ["PERSON", "ORG"]:
                        subject = child.text
                    # Find the object of the announcement
                    if child.dep_ == "dobj" and "partnership" in child.text.lower():
                        # Look for 'with' preposition leading to partner organization
                        for dobj_child in child.children:
                            if dobj_child.dep_ == "prep" and dobj_child.text.lower() == "with":
                                for pobj in dobj_child.children:
                                    if pobj.ent_type_ == "ORG":
                                        partner = pobj.text
                if subject and partner:
                    fact: str = f"{subject} announced partnership with {partner}"
                    facts.append(fact)
        return facts

    def extract_svo(self, doc: Doc) -> list[Tuple[str, str, str]]:
        """
        Extract subject-verb-object triples from the text in a more generic way.
        :param doc: A spaCy Doc object.
        :return: A list of (subject, verb, object) tuples.
        """
        svos: list[Tuple[str, str, str]] = []
        for sentence in doc.sents:
            # For each verb in the sentence
            for token in sentence:
                if token.pos_ == "VERB":
                    subjects = [w for w in token.lefts if w.dep_ in ["nsubj", "nsubjpass"]]
                    objects = [w for w in token.rights if w.dep_ in ["dobj", "dative", "attr", "oprd", "pobj"]]

                    # Handle agent phrases (e.g., "was founded by")
                    if token.dep_ == "ROOT" and token.tag_ in ["VBN", "VBD"]:
                        for child in token.children:
                            if child.dep_ == "agent":
                                agents = [w for w in child.children if w.dep_ == "pobj"]
                                if agents:
                                    subjects.extend(agents)

                    for subj in subjects:
                        subj_text = self.get_compound_noun(subj)
                        verb_text = token.lemma_

                        # Handle direct objects
                        for obj in objects:
                            obj_text = self.get_compound_noun(obj)
                            svos.append((subj_text, verb_text, obj_text))

                        # Handle prepositional objects (e.g., "partnered with")
                        for prep in token.rights:
                            if prep.dep_ == "prep":
                                for pobj in prep.children:
                                    if pobj.dep_ == "pobj":
                                        obj_text = self.get_compound_noun(pobj)
                                        svos.append((subj_text, f"{verb_text} {prep.text}", obj_text))
        return svos

    def get_compound_noun(self, token: Token) -> str:
        """
        Get the full noun phrase (including compounds) for a token.
        :param token: A spaCy Token object.
        :return: The full noun phrase as a string.
        """
        parts = [token.text]
        for child in token.children:
            if child.dep_ in ["compound", "amod"]:
                parts.insert(0, child.text)
        return " ".join(parts)

    def extract_foundations(self, doc: Doc) -> list[str]:
        """
        Extract facts where an organization was founded by a person.
        :param doc: A spaCy Doc object.
        :return: A list of fact strings.
        """
        facts: list[str] = []
        for token in doc:
            if token.lemma_ == "found" and token.dep_ == "ROOT":
                org: Optional[str] = None
                founder: Optional[str] = None
                for child in token.children:
                    if child.dep_ == "nsubjpass" and child.ent_type_ == "ORG":
                        org = child.text
                    elif child.dep_ == "agent":
                        for grandchild in child.children:
                            if grandchild.ent_type_ == "PERSON":
                                founder = grandchild.text
                if org and founder:
                    fact: str = f"{org} was founded by {founder}"
                    facts.append(fact)
        return facts

    def extract_facts(self, text: str) -> list[str]:
        """
        Extract facts from the given text.
        :param text: The text to process.
        :return: A list of fact strings.
        """
        doc: Doc = self.nlp(text)
        facts: Set[str] = set()
        # Extract different types of facts
        # facts.update(self.extract_title_relations(doc))
        # facts.update(self.extract_partnerships(doc))
        # facts.update(self.extract_foundations(doc))
        # Format SVO triples as facts
        svos: list[Tuple[str, str, str]] = self.extract_svo(doc)
        for subj, verb, obj in svos:
            fact: str = f"{subj} {verb} {obj}"
            facts.add(fact)
        # Assign facts to the doc's custom extension
        doc._.facts = facts
        return list(facts)

    def process(self, text: str) -> Doc:
        """
        Process text and attach extracted facts to the Doc object.
        :param text: The text to process.
        :return: The spaCy Doc object with extracted facts.
        """
        doc: Doc = self.nlp(text)
        facts: list[str] = self.extract_facts(text)
        doc._.facts = facts
        return doc
