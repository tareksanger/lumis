from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(description="Name of the editor.")
    role: str = Field(
        description="Role of the editor in the context of the topic.",
    )
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )

    @field_validator("name", mode="after")
    def sanitize_name(cls, value: str) -> str:
        # Remove invalid characters
        value = re.sub(r"[^a-zA-Z0-9_-]", "", value)
        # Truncate to 64 characters
        return value[:64]

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: list[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )

    @field_validator("editors", mode="after")
    def sanitize_editors(cls, value: list[Editor]):
        # Remove invalid characters
        return value[:3]


class RelatedSubjects(BaseModel):
    subjects: list[str] = Field(
        description="Comprehensive list of related subjects as background research.",
    )
