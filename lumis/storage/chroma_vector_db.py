from __future__ import annotations

# from typing import Optional

# from .base_vector_db import BaseVectorDB

# import chromadb
# from chromadb import EmbeddingFunction
# from chromadb.config import Settings
# from lumis.core.document import Chunk
# from lumis.embedding import BaseEmbeddingModel

# # Assuming BaseEmbeddingModel and Chunk classes are defined as before


# class ChromaEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, embedding):
#         self.embedding = embedding

#     def __call__(self, texts: list[str]) -> list[list[float]]:
#         embeddings = self.embedding.embed(texts)
#         return embeddings.tolist()


# class ChromaVectorDB(BaseVectorDB):
#     def __init__(
#         self,
#         embedding: BaseEmbeddingModel,
#         collection_name: str = "default_collection",
#         persist_directory: Optional[str] = "./chroma_db",
#     ):
#         """
#         Initialize the Chroma vector database.

#         Args:
#             embedding_model (BaseEmbeddingModel): Embedding model to use.
#             collection_name (str): Name of the collection in Chroma.
#             persist_directory (Optional[str]): Directory to persist the database.
#         """
#         self.embedding_model = embedding
#         self.collection_name = collection_name
#         self.persist_directory = persist_directory

#         # Initialize Chroma client
#         self.client = chromadb.Client(
#             Settings(
#                 anonymized_telemetry=False,  # Optional: Disable telemetry
#                 persist_directory=self.persist_directory,
#             )
#         )

#         # Get or create the collection
#         self.collection = self.client.get_or_create_collection(
#             name=self.collection_name,
#             embedding_function=ChromaEmbeddingFunction(embedding),  # We'll define this method
#             metadata={"hnsw:space": "cosine"},
#         )

#     def add_chunks(self, chunks: list[Chunk]):
#         """
#         Add Chunk instances to the collection.

#         Args:
#             chunks (List[Chunk]): List of Chunk instances to add.
#         """

#         ids = []
#         documents = []
#         metadatas = []
#         for chunk in chunks:
#             ids.append(chunk.doc_id)
#             documents.append(chunk.content)
#             metadatas.append(chunk.metadata or {})

#         # Add chunks to the collection
#         self.collection.add(
#             ids=ids,
#             documents=documents,
#             # TODO: Fix Metadata
#             # metadatas=metadatas,
#         )

#     def add_chunk(self, chunk: Chunk):
#         return self.add_chunks([chunk])

#     def add_texts(self, texts: list[str], metadata_list: Optional[list[dict]] = None):
#         """
#         Add a list of texts to the collection by creating Chunk instances.

#         Args:
#             texts (List[str]): List of text strings to add.
#             metadata_list (Optional[List[dict]]): List of metadata dictionaries corresponding to each text.
#         """
#         if metadata_list is None:
#             metadata_list = [{}] * len(texts)
#         elif len(metadata_list) != len(texts):
#             raise ValueError("Length of metadata_list must match length of texts")

#         chunks = [Chunk(content=text, metadata=metadata) for text, metadata in zip(texts, metadata_list)]
#         self.add_chunks(chunks)

#     def add_text(self, text: str, metadata: Optional[dict] = None):
#         """
#         Add a single text string to the collection by creating a Chunk instance.

#         Args:
#             text (str): Text string to add.
#             metadata (Optional[dict]): Metadata dictionary for the text.
#         """
#         self.add_texts([text], metadata_list=[metadata] if metadata else None)

#     def search(
#         self,
#         query: str,
#         k: int = 5,
#         where: Optional[dict] = None,
#     ) -> list[Chunk]:
#         """
#         Search the collection for nearest neighbors of the query.

#         Args:
#             query (Union[str, Chunk]): Query text or Chunk instance.
#             k (int): Number of nearest neighbors to return.
#             where (Optional[dict]): Metadata filter conditions.

#         Returns:
#             List[Tuple[distance, Chunk]]
#         """

#         # Perform search
#         results = self.collection.query(
#             query_texts=[query],
#             n_results=k,
#             where=where,
#             include=["distances", "metadatas", "documents"],  # type: ignore
#         )

#         if results is None:
#             return []

#         hits = []
#         for i in range(len(results["ids"][0])):
#             # distance = results["distances"][0][i] if results["distances"] else None
#             metadata = results["metadatas"][0][i] if results["metadatas"] else None
#             content = results["documents"][0][i] if results["documents"] else None
#             doc_id = results["ids"][0][i]

#             chunk = Chunk(
#                 doc_id=doc_id,
#                 content=content or "",
#                 metadata=metadata.__dict__ if metadata else None,
#             )
#             hits.append(chunk)

#         return hits

#     def delete_chunk(self, chunk: Chunk):
#         """
#         Delete a Chunk from the collection.

#         Args:
#             chunk (Chunk): The Chunk to delete.
#         """
#         self.collection.delete(ids=[chunk.doc_id])

#     def persist(self):
#         """
#         Persist the database to disk.
#         """
#         pass

#     def unload(self):
#         """
#         Unload the collection from memory.
#         """
#         self.client.reset()

#     def load(self):
#         """
#         Reload the collection from disk.
#         """
#         pass
