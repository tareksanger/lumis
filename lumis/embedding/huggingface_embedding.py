from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

from .base_embedding import BaseEmbeddingModel, Embedding

import numpy as np


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: Optional[int] = None):
        super().__init__(model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise ImportError("Failed to import from `sentence_transformers` please make sure this package is installed.")

        self.model = SentenceTransformer(model_name)

        self._dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.executor = ThreadPoolExecutor()

    def embed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return np.array(embeddings)

    async def aembed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            text = [text]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(self.executor, self.model.encode, text)
        return np.array(embeddings)
