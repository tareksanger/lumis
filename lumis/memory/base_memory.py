from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, TypedDict, TypeVar

from openai.types.chat import ChatCompletionMessageParam


class BaseMemoryData(TypedDict, total=False):
    messages: list[ChatCompletionMessageParam]
    max_memory_size: int


MemoryT = TypeVar("MemoryT", bound="BaseMemory")


class BaseMemory(ABC):
    @abstractmethod
    async def add(self, message: ChatCompletionMessageParam, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def prepend(self, messages: list[ChatCompletionMessageParam] | ChatCompletionMessageParam, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def insert(self, index: int, message: ChatCompletionMessageParam, *args, **kwargs) -> Any: ...

    @abstractmethod
    async def get(self, *args, **kwargs) -> list[ChatCompletionMessageParam]: ...

    @abstractmethod
    async def clear(self): ...

    @abstractmethod
    async def remove(self, index: int): ...

    @abstractmethod
    async def update(self, index: int, message: ChatCompletionMessageParam): ...

    @property
    @abstractmethod
    def length(self) -> int: ...

    @abstractmethod
    def to_dict(self) -> BaseMemoryData:
        """Serializes the memory to a dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls: type[MemoryT], data: BaseMemoryData) -> MemoryT:
        """Deserializes the memory from a dictionary."""
        ...

    # TODO: Move this to some base class that all data classes can conform to..
    def __deepcopy__(self, memo):
        # Create a new instance without calling __init__
        cls = self.__class__
        new_instance = cls.__new__(cls)
        memo[id(self)] = new_instance

        # Recursively deep copy all attributes
        for k, v in self.__dict__.items():
            setattr(new_instance, k, copy.deepcopy(v, memo))

        return new_instance

    @property
    def __dict__(self):
        return self.to_dict()

    def dict(self):
        return self.to_dict()
