from __future__ import annotations

from typing import Optional
import uuid

from lumis.embedding.base_embedding import Embedding

from pydantic import BaseModel, ConfigDict, Field


class BaseTextNode(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The content of the text node.")
    metadata: Optional[dict] = Field(default=None, description="Metadata for the text node.")
    embedding: Optional[Embedding] = Field(default=None, description="Embedding for the text node.")

    # Update the configuration to allow arbitrary types and set frozen to True for immutability
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __eq__(self, other):
        if not isinstance(other, BaseTextNode):
            return False
        return self.doc_id == other.doc_id

    def __hash__(self):
        return hash(self.doc_id)


class Document(BaseTextNode):
    pass


class Chunk(BaseTextNode):
    parent_id: Optional[str] = Field(default=None, description="The parent id of the text node.")
