from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionMessageParam


class BaseMemory(ABC):
    @abstractmethod
    def add(self, message: ChatCompletionMessageParam): ...

    @abstractmethod
    def get(self, *args, **kwargs) -> list[ChatCompletionMessageParam]: ...

    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def remove(self, index: int): ...

    @abstractmethod
    def update(self, index: int, message: ChatCompletionMessageParam): ...

    @abstractmethod
    def insert(self, index: int, message: ChatCompletionMessageParam): ...

    @property
    @abstractmethod
    def length(self) -> int: ...
