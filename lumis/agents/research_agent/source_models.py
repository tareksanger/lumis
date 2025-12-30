from __future__ import annotations

"""
Source models for the ResearchAgent.

This module provides Pydantic models for different types of sources
that can be used by the research agent.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceType(BaseModel):
    """Base model for all source types."""

    url: Optional[str] = None  # Now nullable
    title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    credibility: float = 1.0
    source_type: Literal["arxiv", "wiki", "web", "gemini"] = Field(description="Type of source (arxiv, wiki, web, gemini)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    timestamp_accessed: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(from_attributes=True)
