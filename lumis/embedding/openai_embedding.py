from __future__ import annotations

import logging
from typing import Optional, Union

from lumis.core.utils.coroutine import run_sync

from .base_embedding import BaseEmbeddingModel, Embedding

import numpy as np
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_DIMS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-similarity-ada-001": 1024,
    "text-similarity-babbage-001": 2048,
    "text-similarity-curie-001": 4096,
    "text-similarity-davinci-001": 12288,
    "text-search-ada-doc-001": 1024,
    "text-search-ada-query-001": 1024,
    "text-search-babbage-doc-001": 2048,
    "text-search-babbage-query-001": 2048,
    "text-search-curie-doc-001": 4096,
    "text-search-curie-query-001": 4096,
    "text-search-davinci-doc-001": 12288,
    "text-search-davinci-query-001": 12288,
    "code-search-ada-code-001": 1024,
    "code-search-ada-text-001": 1024,
    "code-search-babbage-code-001": 2048,
    "code-search-babbage-text-001": 2048,
}


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small", dimension: Optional[int] = None):
        super().__init__(model_name)
        self.client = OpenAI()
        self.aclient = AsyncOpenAI()
        # Set the dimension based on the model; for example:
        self._dimension = dimension or _DEFAULT_DIMS.get(model_name, 1536)

    def embed(self, text: Union[str, list[str]]) -> Embedding:
        return run_sync(self.aembed(text))

    async def aembed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            logger.debug("Embedding single string with model %s", self.model_name)
            if text == "":
                logger.debug("Empty string provided; returning zero vector with dimension %d", self.dimension)
                return np.zeros(self.dimension, dtype="float32")

            resp = await self.aclient.embeddings.create(model=self.model_name, input=[text])
            embedding = np.array(resp.data[0].embedding, dtype="float32")
            logger.debug("Received embedding vector length %d", embedding.shape[0])
            return embedding

        if len(text) == 0:
            logger.debug("Empty list provided; returning empty embedding with dimension %d", self.dimension)
            return np.zeros((0, self.dimension), dtype="float32")

        logger.debug("Embedding %d strings with model %s", len(text), self.model_name)
        resp = await self.aclient.embeddings.create(model=self.model_name, input=text)
        embeddings = np.array([d.embedding for d in resp.data], dtype="float32")
        logger.debug("Received %d embeddings with vector length %d", embeddings.shape[0], embeddings.shape[1])
        return embeddings
