from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

Embedding = NDArray[np.float32]
SimilarityMode = Literal["euclidean", "dot_product", "cosine"]  # noqa: F821


class BaseEmbeddingModel(ABC):
    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        self._dimension = 0  # To be set by subclasses

    @property
    def dimension(self) -> int:
        assert self._dimension is not None and self._dimension > 0, "Dimension not set!"
        return self._dimension

    @abstractmethod
    def embed(self, text: Union[str, list[str]]) -> Embedding:
        """
        Generate embeddings for a single string or a list of strings.
        """
        pass

    @abstractmethod
    async def aembed(self, text: Union[str, list[str]]) -> Embedding:
        """
        Generate embeddings for a single string or a list of strings.
        """
        pass

    def similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        mode: SimilarityMode = "cosine",
    ) -> float:
        """Get embedding similarity."""
        if mode == "euclidean":
            # Using -euclidean distance as similarity to achieve same ranking order
            return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
        elif mode == "dot_product":
            return np.dot(embedding1, embedding2)
        else:
            product = np.dot(embedding1, embedding2)
            norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            return product / norm
