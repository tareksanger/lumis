from __future__ import annotations

import json
from typing import Any, cast, Dict, TYPE_CHECKING, Union
import uuid

from django.db import models
from openai.types.chat import ChatCompletionMessageParam


class ChatMemory(models.Model):
    """
    Abstract base model representing a chat memory or message.
    This model combines common functionality from Memory and Message models.
    """

    if TYPE_CHECKING:
        id: int

    class Role(models.TextChoices):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        FUNCTION = "function"
        TOOL = "tool"
        DEVELOPER = "developer"

    uuid: models.UUIDField = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    name = models.CharField(max_length=255, help_text="The name of the user or assistant")
    role = models.CharField(max_length=16, choices=Role.choices, help_text="The role of the message sender")
    _content = models.TextField(db_column="content", help_text="The content of the message")
    _refined_content = models.TextField(db_column="refined_content", help_text="The refined content of the message", null=True, blank=True)

    metadata = models.JSONField(default=dict, help_text="Metadata about the message")
    token_count = models.IntegerField(default=0, help_text="The number of tokens in the message")

    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    @property
    def content(self) -> Union[str, Dict[str, Any], list, None]:
        """Get the content, deserializing from JSON if possible."""
        if not self._content:
            return None

        try:
            return json.loads(self._content)
        except (json.JSONDecodeError, TypeError):
            return self._content

    @content.setter
    def content(self, value: Union[str, Dict[str, Any], list, None]) -> None:
        """Set the content, serializing to JSON if not a string."""
        if value is None:
            self._content = None
        elif isinstance(value, str):
            self._content = value
        else:
            self._content = json.dumps(value)

    @property
    def refined_content(self) -> Union[str, Dict[str, Any], list, None]:
        """Get the refined content, deserializing from JSON if possible."""
        if not self._refined_content:
            return None

        try:
            return json.loads(self._refined_content)
        except (json.JSONDecodeError, TypeError):
            return self._refined_content

    @refined_content.setter
    def refined_content(self, value: Union[str, Dict[str, Any], list, None]) -> None:
        """Set the refined content, serializing to JSON if not a string."""
        if value is None:
            self._refined_content = None
        elif isinstance(value, str):
            self._refined_content = value
        else:
            self._refined_content = json.dumps(value)

    def to_chat_completion_message(self, refinement: bool = False) -> ChatCompletionMessageParam:
        """
        Converts the message to a chat completion message.

        If refinement is True, the refined content is used if it exists - Refinement should only be used for assistant messages.
        """
        content = self.refined_content if refinement and self.refined_content else self.content

        message: ChatCompletionMessageParam = cast(
            ChatCompletionMessageParam,
            {
                "role": self.role,
                "content": content,
                "name": self.name,
            },
        )
        return message

    def get_token_count(self) -> int:
        """Get the token count for the content."""
        if self._content is None:
            return 0

        tokenizable_content = self._prepare_tokenizable_content(self._content)
        if not tokenizable_content:
            return 0

        return self.get_token_count_from_content(tokenizable_content)

    @classmethod
    def _prepare_tokenizable_content(cls, raw_content: Union[str, Dict[str, Any], list, None]) -> str:
        if raw_content is None:
            return ""

        if isinstance(raw_content, str):
            stripped = raw_content.strip()
            if not stripped:
                return ""
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return stripped
            if isinstance(parsed, (str, int, float, bool)):
                return str(parsed).strip()
            return cls._stringify_structured_message(parsed)

        if isinstance(raw_content, (int, float, bool)):
            return str(raw_content).strip()

        return cls._stringify_structured_message(raw_content)

    @classmethod
    def _stringify_structured_message(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            parts: list[str] = []
            for key in ("text", "content", "input", "output", "value"):
                if key in value:
                    part = cls._stringify_structured_message(value[key])
                    if part:
                        parts.append(part)
            if not parts:
                for item in value.values():
                    part = cls._stringify_structured_message(item)
                    if part:
                        parts.append(part)
            return " ".join(parts).strip()
        if isinstance(value, list):
            parts = [cls._stringify_structured_message(item) for item in value]
            return " ".join(part for part in parts if part).strip()
        return ""

    @classmethod
    def get_token_count_from_content(cls, content: str) -> int:
        """Get the token count for a given content string."""
        if not content.strip():
            return 0
        # Note: This method should be implemented by concrete classes
        # as they may have different tokenizer configurations
        raise NotImplementedError("Subclasses must implement get_token_count_from_content")

    def __str__(self) -> str:
        """String representation of the chat memory."""
        content_str = str(self.content)
        if len(content_str) > 50:
            content_str = content_str[:47] + "..."
        return f"{self.role}: {content_str}"

    class Meta:
        abstract = True
        ordering = ["created_at"]
