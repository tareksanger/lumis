from __future__ import annotations

from abc import ABC, abstractmethod

from lumis.core.document import Chunk


class BaseVectorDB(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]): ...

    async def aadd_chunks(self, chunks: list[Chunk]):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement aadd_chunks.")

    @abstractmethod
    def add_chunk(self, chunk: Chunk): ...

    async def aadd_chunk(self, chunk: Chunk):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement aadd_chunk.")

    @abstractmethod
    def add_texts(self, texts: list[str], *args, **kwargs): ...

    async def aadd_texts(self, texts: list[str], *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement aadd_texts.")

    @abstractmethod
    def add_text(self, text: str, *args, **kwargs): ...

    async def aadd_text(self, text: str, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement aadd_text.")

    @abstractmethod
    def search(self, query: str, k: int = 5, *args, **kwargs) -> list[Chunk]: ...

    async def asearch(self, query: str, k: int = 5, *args, **kwargs) -> list[Chunk]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement async search.")

    def clear(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement clear.")
