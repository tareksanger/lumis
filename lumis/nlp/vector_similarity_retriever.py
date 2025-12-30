from __future__ import annotations

from lumis.core.document import Chunk
from lumis.embedding import BaseEmbeddingModel
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


class VectorSimilarityRetriever:
    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model

    def extract_relevant_chunks(self, query: str, chunks: list[Chunk], k: int = 5) -> list[tuple[float, Chunk]]:
        query_embedding = self.embedding_model.embed(query)
        query_embedding = query_embedding / norm(query_embedding)  # Normalize

        chunk_embeddings = self.embedding_model.embed([chunk.content for chunk in chunks])
        chunk_embeddings = chunk_embeddings / norm(chunk_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = cosine_similarity(chunk_embeddings, query_embedding.reshape(1, -1)).flatten()

        return sorted(zip(similarities, chunks), key=lambda x: -x[0])[:k]  # Rank by similarity

    async def aextract_relevant_chunks(self, query: str, chunks: list[Chunk], k: int = 5) -> list[tuple[float, Chunk]]:
        query_embedding = await self.embedding_model.aembed(query)
        query_embedding = query_embedding / norm(query_embedding)  # Normalize

        chunk_embeddings = await self.embedding_model.aembed([chunk.content for chunk in chunks])
        chunk_embeddings = chunk_embeddings / norm(chunk_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = cosine_similarity(chunk_embeddings, query_embedding.reshape(1, -1)).flatten()

        return sorted(zip(similarities, chunks), key=lambda x: -x[0])[:k]  # Rank by similarity
