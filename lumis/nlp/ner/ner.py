from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc

relative_path = Path(__file__).parent / "config" / "entity_rel_extraction.cfg"


class Ner:
    def __init__(self, config: str = str(relative_path)):
        try:
            from spacy_llm.util import assemble
        except ImportError:
            raise ImportError(
                "spacy and spacy-llm are required for Ner. "
                "Install them with: pip install lumis-ai[spacy]"
            )
        self.nlp = assemble(config)

    def get_entities(self, text: str) -> list[tuple[str, str]]:
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_doc(self, text: str) -> Doc:
        return self.nlp(text)
