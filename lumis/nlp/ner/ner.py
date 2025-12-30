from pathlib import Path

from spacy.tokens import Doc
from spacy_llm.util import assemble

relative_path = Path(__file__).parent / "config" / "entity_rel_extraction.cfg"


class Ner:
    def __init__(self, config: str = str(relative_path)):
        self.nlp = assemble(config)

    def get_entities(self, text: str) -> list[tuple[str, str]]:
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_doc(self, text: str) -> Doc:
        return self.nlp(text)
