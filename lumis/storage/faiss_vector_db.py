from __future__ import annotations

import logging
import pickle
from typing import Literal, Optional

from lumis.core.document import Chunk
from lumis.embedding.base_embedding import BaseEmbeddingModel

from .base_vector_db import BaseVectorDB

from asgiref.sync import sync_to_async
import faiss
import numpy as np

IndexType = Literal["IndexFlatL2", "IndexFlatIP"]

logger = logging.getLogger(__name__)


class FaissVectorDB(BaseVectorDB):
    def __init__(
        self,
        embedding: BaseEmbeddingModel,
        index_type: IndexType = "IndexFlatL2",
        use_gpu: bool = False,
    ):
        """
        Initialize the FAISS vector database.

        Args:
            embedding (BaseEmbeddingModel): Embedding model to use.
            dimension (int): Dimensionality of the vectors.
            index_type (IndexType): Type of FAISS index (default: 'IndexFlatL2').
            use_gpu (bool): Whether to use GPU acceleration (default: False).
        """
        self.embedding: BaseEmbeddingModel = embedding
        self.dimension: int = embedding.dimension
        self.use_gpu: bool = use_gpu

        # Create the FAISS index
        if index_type == "IndexFlatL2":
            # Use IndexHNSWFlat for better scalability and deletion support
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        elif index_type == "IndexFlatIP":
            # Use IndexHNSWFlat with Inner Product
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Wrap the index with IndexIDMap to support IDs
        self.index = faiss.IndexIDMap(self.index)

        # Move to GPU if required
        if use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.id_map: dict[int, Chunk] = {}  # Mapping from FAISS IDs to Chunk objects
        self.chunk_id_to_faiss_id: dict[str, int] = {}  # Mapping from Chunk doc_id to FAISS IDs
        self.next_id: int = 0  # For generating unique numerical IDs for FAISS

    def add_chunks(self, chunks: list[Chunk]):
        """
        Add Chunk instances to the index.

        Args:
            chunks (list[Chunk]): list of Chunk instances to add.
        """
        if not chunks:
            return

        batch_size = 1024
        total = len(chunks)
        offset = 0

        while offset < total:
            logger.debug(f"Processing batch from offset {offset} to {offset + batch_size}")
            batch = chunks[offset : offset + batch_size]

            batch_embeddings: list[np.ndarray | None] = [None] * len(batch)
            to_embed_indices: list[int] = []
            to_embed_contents: list[str] = []

            for i, chunk in enumerate(batch):
                if chunk.embedding is not None:
                    emb = np.asarray(chunk.embedding, dtype="float32")
                    if emb.ndim == 2:
                        if emb.shape[0] != 1:
                            raise ValueError(f"Invalid embedding shape for chunk {chunk.doc_id}: {emb.shape}")
                        emb = emb[0]
                    if emb.shape[0] != self.dimension:
                        raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {emb.shape[0]}")
                    batch_embeddings[i] = emb
                else:
                    to_embed_indices.append(i)
                    to_embed_contents.append(chunk.content)

            if to_embed_indices:
                logger.debug(f"Embedding {len(to_embed_indices)} chunks without precomputed embeddings.")
                new_embeddings = np.asarray(self.embedding.embed(to_embed_contents), dtype="float32")
                if new_embeddings.ndim == 1:
                    new_embeddings = new_embeddings[np.newaxis, :]
                if new_embeddings.shape[1] != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {new_embeddings.shape[1]}")
                for idx, emb in zip(to_embed_indices, new_embeddings):
                    batch_embeddings[idx] = emb

            embeddings_array = np.vstack(batch_embeddings).astype("float32")  # type: ignore[arg-type]

            ids = []
            for chunk in batch:
                faiss_id = self.next_id
                self.next_id += 1
                self.id_map[faiss_id] = chunk
                self.chunk_id_to_faiss_id[chunk.doc_id] = faiss_id
                ids.append(faiss_id)

            ids_array = np.array(ids, dtype="int64")

            if isinstance(self.index, faiss.IndexFlatIP):
                faiss.normalize_L2(embeddings_array)

            self.index.add_with_ids(embeddings_array, ids_array)  # type: ignore

            offset += batch_size

    async def aadd_chunks(self, chunks: list[Chunk]):
        """
        Asynchronously add Chunk instances to the index.

        Args:
            chunks (list[Chunk]): list of Chunk instances to add.
        """
        await sync_to_async(self.add_chunks)(chunks)

    def add_chunk(self, chunk: Chunk):
        """
        Add a single Chunk instance to the index.

        Args:
            chunk (Chunk): The Chunk instance to add.
        """
        self.add_chunks([chunk])

    async def aadd_chunk(self, chunk: Chunk):
        """
        Asynchronously add a single Chunk instance to the index.

        Args:
            chunk (Chunk): The Chunk instance to add.
        """
        await sync_to_async(self.add_chunk)(chunk)

    def add_texts(self, texts: list[str], metadata_list: Optional[list[dict]] = None):
        """
        Add a list of texts to the index by creating Chunk instances.

        Args:
            texts (list[str]): list of text strings to add.
            metadata_list (Optional[list[dict]]): list of metadata dictionaries corresponding to each text.
        """
        chunks = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            chunk = Chunk(content=text, metadata=metadata)
            chunks.append(chunk)
        self.add_chunks(chunks)

    async def aadd_texts(self, texts: list[str], metadata_list: Optional[list[dict]] = None):
        """
        Asynchronously add a list of texts to the index by creating Chunk instances.

        Args:
            texts (list[str]): list of text strings to add.
            metadata_list (Optional[list[dict]]): list of metadata dictionaries corresponding to each text.
        """
        await sync_to_async(self.add_texts)(texts, metadata_list)

    def add_text(self, text: str, metadata: Optional[dict] = None):
        """
        Add a single text string to the index by creating a Chunk instance.

        Args:
            text (str): Text string to add.
            metadata (Optional[dict]): Metadata dictionary for the text.
        """
        self.add_texts([text], metadata_list=[metadata] if metadata else None)

    async def aadd_text(self, text: str, metadata: Optional[dict] = None):
        """
        Asynchronously add a single text string to the index by creating a Chunk instance.

        Args:
            text (str): Text string to add.
            metadata (Optional[dict]): Metadata dictionary for the text.
        """
        await sync_to_async(self.add_text)(text, metadata)

    # TODO: Revisit re-ranking strategies, MMR is not working as expected, also not a fan of the way we're passing in the config
    def search(self, query: str, k: int = 5, rerank: Literal["None", "mmr"] = "None", lambda_: float = 0.7, fetch_k: int | None = None) -> list[Chunk]:
        """
        Search the index for nearest neighbors of the query.

        Args:
            query (Union[str, Chunk]): Query text or Chunk instance.
            k (int): Number of nearest neighbors to return.

        Returns:
            list[Tuple[distance, Chunk]]
        """
        # Compute embedding for query

        query_embedding = self.embedding.embed(query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {query_embedding.shape[1]}")

        # Perform search
        distances, ids = self.index.search(query_embedding, fetch_k or max(k * 4, k))  # type: ignore

        results: list[Chunk] = []
        for dist, id_ in zip(distances[0], ids[0]):
            if id_ == -1:
                continue  # No more results
            chunk = self.id_map.get(id_, None)
            if chunk is None:
                continue
            results.append(chunk)

        print(f"Search results count: {len(results)}")
        if rerank.lower() == "mmr":
            # TODO: Chunk embeddings might be None, handle that case find another way to get the embeddings (or save them to the check on add?)
            order = self._mmr(query_embedding[0], np.vstack([chunk.embedding for chunk in results if chunk.embedding is not None]).astype("float32"), lambda_=lambda_, top_n=k)
            results = [results[i] for i in order]

        return results[:k]

    async def asearch(self, query: str, k: int = 5, rerank: Literal["None", "mmr"] = "None", lambda_: float = 0.7, fetch_k: int | None = None) -> list[Chunk]:
        """
        Asynchronously search the index for nearest neighbors of the query.

        Args:
            query (Union[str, Chunk]): Query text or Chunk instance.
            k (int): Number of nearest neighbors to return.

        Returns:
            list[Tuple[distance, Chunk]]
        """
        return await sync_to_async(self.search)(query, k, rerank=rerank, lambda_=lambda_, fetch_k=fetch_k)

    def save(self, index_path: str, id_map_path: str):
        """
        Save the index and ID map to disk.

        Args:
            index_path (str): Path to save the FAISS index.
            id_map_path (str): Path to save the ID map.
        """
        faiss.write_index(self.index, index_path)
        # We need to serialize the id_map and chunk_id_to_faiss_id
        with open(id_map_path, "wb") as f:
            pickle.dump({"id_map": self.id_map, "chunk_id_to_faiss_id": self.chunk_id_to_faiss_id, "next_id": self.next_id}, f)

    def load(self, index_path: str, id_map_path: str):
        """
        Load the index and ID map from disk.

        Args:
            index_path (str): Path to the FAISS index.
            id_map_path (str): Path to the ID map.
        """
        self.index = faiss.read_index(index_path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        with open(id_map_path, "rb") as f:
            data = pickle.load(f)
            self.id_map = data["id_map"]
            self.chunk_id_to_faiss_id = data["chunk_id_to_faiss_id"]
            self.next_id = data["next_id"]

    def delete_chunk(self, chunk: Chunk):
        """
        Delete a Chunk from the index.

        Args:
            chunk (Chunk): The Chunk to delete.
        """
        faiss_id = self.chunk_id_to_faiss_id.get(chunk.doc_id)
        if faiss_id is not None:
            self.index.remove_ids(np.array([faiss_id], dtype="int64"))
            del self.id_map[faiss_id]
            del self.chunk_id_to_faiss_id[chunk.doc_id]

    def _mmr(
        self,
        query_vec: np.ndarray,
        doc_vecs: np.ndarray,
        lambda_: float = 0.7,
        top_n: int = 8,
    ) -> list[int]:
        """
        query_vec: (D,)
        doc_vecs: (N, D)
        returns indices into doc_vecs of selected items
        """
        # cosine == dot product because vectors are unit norm
        query_sims = doc_vecs @ query_vec  # (N,)

        selected: list[int] = []
        candidate_indices = list(range(doc_vecs.shape[0]))

        while candidate_indices and len(selected) < top_n:
            best_idx = None
            best_score = -1.0

            for i in candidate_indices:
                # similarity to query
                sim_q = query_sims[i]

                # similarity to already-selected docs (max)
                if selected:
                    sims_to_selected = doc_vecs[i] @ doc_vecs[selected].T  # (len(selected),)
                    sim_s = float(np.max(sims_to_selected))
                else:
                    sim_s = 0.0

                score = lambda_ * sim_q - (1.0 - lambda_) * sim_s

                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        return selected
