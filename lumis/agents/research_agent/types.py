from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, TypeVar, Union
from urllib.parse import urlparse

from agents import Agent, RunContextWrapper

from pydantic import BaseModel, Field, field_validator

# Type for callback that can be either sync or async
ResearchCallbackT = TypeVar("ResearchCallbackT", bound=Callable[[RunContextWrapper, Agent, Any], Union[None, Awaitable[None]]])


class ResearchAgentResponse(BaseModel):
    """The response from the research agent."""

    response: str = Field(description="The research response")
    references: list[str] = Field(default_factory=list, description="The list of urls used to generate the response.")

    @field_validator("references", mode="before")
    @classmethod
    def validate_urls(cls, urls: list[str]) -> list[str]:  # noqa: C901
        """Validate and normalize URLs in the references list."""
        validated_urls = []
        for url in urls:
            # Extract URL using regex if it's embedded in text
            url_match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', url)
            if url_match:
                url = url_match.group(0)

            # Ensure URL has a protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Parse URL to validate structure
            try:
                parsed = urlparse(url)
                if not all([parsed.scheme, parsed.netloc]):
                    continue
                # Clean the URL by removing any trailing punctuation or spaces
                cleaned_url = url.strip().rstrip(".,;:")
                validated_urls.append(cleaned_url)
            except Exception:
                continue

        return validated_urls
