from __future__ import annotations

import asyncio
from typing import Dict

from .base_memory import BaseMemory, BaseMemoryData

from openai.types.chat import ChatCompletionMessageParam


class SimpleMemory(BaseMemory):
    """
    SimpleMemory is a concrete implementation of BaseMemory that stores messages in memory.

    It maintains a list of messages up to a maximum size, defined by `max_memory_size`.
    If the number of messages exceeds `max_memory_size`, the `get` method returns a subset of messages
    from the beginning and end of the list.

    Attributes:
        max_memory_size (int): The maximum number of messages to retain in memory.
    """

    def __init__(self, max_memory_size: int = 2000, messages: list[ChatCompletionMessageParam] | None = None):
        """
        Initializes a new instance of SimpleMemory.

        Args:
            max_memory_size (int, optional): The maximum number of messages to retain. Defaults to 2000.
        """
        self._lock = asyncio.Lock()
        self.max_memory_size = max_memory_size
        self.messages: list[ChatCompletionMessageParam] = messages if messages is not None else []

    async def add(self, message: ChatCompletionMessageParam):
        """
        Adds a message to the memory.

        Args:
            message (ChatCompletionMessageParam): The message to add.
        """
        async with self._lock:
            self.messages.append(message)

    async def prepend(self, messages: list[ChatCompletionMessageParam] | ChatCompletionMessageParam):
        """
        Inserts one or more messages at the beginning of memory.

        Args:
            messages (list[ChatCompletionMessageParam] | ChatCompletionMessageParam): Messages to place before existing entries.
        """
        if isinstance(messages, dict):
            items = [messages]
        else:
            items = list(messages)

        async with self._lock:
            self.messages = items + self.messages

    async def insert(self, index: int, message: ChatCompletionMessageParam):
        """
        Inserts a single message at the specified index.

        Args:
            index (int): Position where the message should be inserted.
            message (ChatCompletionMessageParam): Message to insert.

        Raises:
            IndexError: If the index is out of range.
        """
        async with self._lock:
            if not 0 <= index <= len(self.messages):
                raise IndexError("Index out of range.")
            self.messages.insert(index, message)

    async def get(self, *args, **kwargs) -> list[ChatCompletionMessageParam]:
        """
        Retrieves messages from memory.

        If the total number of messages is less than or equal to `max_memory_size`,
        all messages are returned. Otherwise, a subset of messages from the start
        and end of the list is returned to maintain context.

        Returns:
            list[ChatCompletionMessageParam]: The list of retrieved messages.
        """
        async with self._lock:
            if len(self.messages) <= self.max_memory_size:
                return self.messages.copy()

            # Calculate the number of items to take from the start and the end
            half_max = self.max_memory_size // 2

            start = self.messages[:half_max]
            end = self.messages[-half_max:]

            return start + end

    async def clear(self):
        """
        Clears all messages from memory.
        """
        async with self._lock:
            self.messages = []

    async def remove(self, index: int):
        """
        Removes a message at the specified index.

        Args:
            index (int): The index of the message to remove.

        Raises:
            IndexError: If the index is out of range.
        """
        async with self._lock:
            if 0 <= index < len(self.messages):
                self.messages.pop(index)
            else:
                raise IndexError("Index out of range.")

    async def update(self, index: int, message: ChatCompletionMessageParam):
        """
        Updates the message at the specified index.

        Args:
            index (int): The index of the message to update.
            message (ChatCompletionMessageParam): The new message to replace the old one.

        Raises:
            IndexError: If the index is out of range.
        """
        async with self._lock:
            if 0 <= index < len(self.messages):
                self.messages[index] = message
            else:
                raise IndexError("Index out of range.")

    @classmethod
    def from_dict(cls, data: BaseMemoryData) -> SimpleMemory:
        max_memory_size = data.get("max_memory_size", 100)
        messages = data.get("messages", [])
        return cls(max_memory_size=max_memory_size, messages=messages)

    @property
    def length(self) -> int:
        """
        Returns the number of messages in memory.

        Returns:
            int: The total number of messages.
        """
        return len(self.messages)

    def to_dict(self):
        """
        Serializes the memory to a dictionary.

        Returns:
            dict: A dictionary representation of the memory.
        """
        return {
            "max_memory_size": self.max_memory_size,
            "messages": self.messages,
        }
