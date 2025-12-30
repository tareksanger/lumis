from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from asgiref.sync import sync_to_async
from lumis.core.document import Chunk, Document
from lumis.embedding import BaseEmbeddingModel
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from typing_extensions import TypedDict


class SplitSentence(TypedDict):
    sentence: str
    embedding: np.ndarray


class SemanticParser:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        breakpoint_percentile_threshold: int = 95,
        max_tokens: int = 4048,
    ):
        """
        Initialize the SemanticParser.

        Args:
            embedding_model (BaseEmbeddingModel): The embedding model to generate embeddings.
            breakpoint_percentile_threshold (int): The percentile threshold to decide where to create breakpoints.
            max_tokens (int): The maximum number of tokens allowed in a chunk.
        """
        self.embedding_model = embedding_model
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.max_tokens = max_tokens

        self.tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    def split_by_sentence(self, text: str, tokenizer: Any) -> list[str]:
        """Get the spans and then return the sentences.

        Using the start index of each span
        Instead of using end, use the start of the next span if available
        """
        spans = list(tokenizer.span_tokenize(text))
        sentences = []
        for i, span in enumerate(spans):
            start = span[0]
            if i < len(spans) - 1:
                end = spans[i + 1][0]
            else:
                end = len(text)

            sent = text[start:end]

            # Filter non-printable control characters
            filter_chars = "".join([chr(i) for i in range(1, 32)])
            translation_table = str.maketrans(filter_chars, " " * len(filter_chars))
            sent = sent.translate(translation_table)

            sentences.append(sent)
        return sentences

    def _build_sentence_chunks(self, sentences: list[str], embeddings: np.ndarray) -> List[SplitSentence]:
        return [
            SplitSentence(
                sentence=sent.strip(),
                embedding=embedding,
            )
            for sent, embedding in zip(sentences, embeddings.tolist())
        ]

    def _parse(self, text: str, metadata: Optional[dict] = None, parent_id: Optional[str] = None):
        """
        Parse the raw text into Chunk instances with embeddings and metadata.

        Args:
            text (str): The raw text to parse.
            metadata (Optional[Dict]): Metadata to associate with each chunk.

        Returns:
            List[Chunk]: A list of Chunk instances.
        """
        sentences = self.split_by_sentence(text, self.tokenizer)
        embeddings = self.embedding_model.embed(sentences)
        combined_sentences_and_embeddings = self._build_sentence_chunks(sentences, embeddings)

        distances = self._calculate_distances_between_sentence_groups(combined_sentences_and_embeddings)
        chunks = self._build_node_chunks(combined_sentences_and_embeddings, distances, metadata=metadata, parent_id=parent_id)
        return chunks

    async def _aparse(self, text: str, metadata: Optional[dict] = None, parent_id: Optional[str] = None):
        return await sync_to_async(self._parse)(text, metadata, parent_id)

    def parse(self, text: str, metadata: Optional[dict] = None, parent_id: Optional[str] = None):
        """
        Parse the raw text into Chunk instances with embeddings and metadata.

        Args:
            text (str): The raw text to parse.
            metadata (Optional[Dict]): Metadata to associate with each chunk.

        Returns:
            List[Chunk]: A list of Chunk instances.
        """
        return self._parse(text, metadata, parent_id)

    async def aparse(self, text: str, metadata: Optional[dict] = None, parent_id: Optional[str] = None):
        return await sync_to_async(self.parse)(text, metadata, parent_id)

    def _calculate_distances_between_sentence_groups(self, sentences: list[SplitSentence]) -> list[float]:
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["embedding"]
            embedding_next = sentences[i + 1]["embedding"]

            similarity = self.embedding_model.similarity(embedding_current, embedding_next)
            distance = 1 - similarity
            distances.append(distance)

        return distances

    def _build_node_chunks(
        self,
        sentences: list[SplitSentence],
        distances: list[float],
        metadata: Optional[dict] = None,
        parent_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Builds semantic-based chunks, then ensures no chunk exceeds self.max_tokens.
        """
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
            indices_above_threshold = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0
            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                group_text = "".join([d["sentence"] for d in group])
                # Split into sub-chunks if exceeding self.max_tokens
                sub_chunks = self._split_text_by_token_limit(group_text, metadata, parent_id)
                chunks.extend(sub_chunks)
                start_index = index + 1

            if start_index < len(sentences):
                group_text = "".join([d["sentence"] for d in sentences[start_index:]])
                sub_chunks = self._split_text_by_token_limit(group_text, metadata, parent_id)
                chunks.extend(sub_chunks)

        else:
            # If, for some reason we didn't get any distances (i.e. very small documents),
            # treat the whole document as a single chunk—still respecting self.max_tokens.
            combined_text = " ".join([s["sentence"] for s in sentences])
            sub_chunks = self._split_text_by_token_limit(combined_text, metadata, parent_id)
            chunks.extend(sub_chunks)

        return chunks

    def _split_text_by_token_limit(
        self,
        text: str,
        metadata: Optional[dict],
        parent_id: Optional[str],
    ) -> List[Chunk]:
        """
        Splits the given text into chunks that do not exceed the max_tokens limit.
        Returns a list of Chunk instances.
        """
        tokens = word_tokenize(text)
        all_chunks = []
        current_tokens = []

        for token in tokens:
            current_tokens.append(token)
            if len(current_tokens) >= self.max_tokens:
                chunk_text = " ".join(current_tokens)
                chunk = Chunk(content=chunk_text, metadata={**(metadata or {}), "token_count": len(current_tokens)}, parent_id=parent_id)
                all_chunks.append(chunk)
                current_tokens = []

        # Handle any remaining tokens
        if current_tokens:
            chunk_text = " ".join(current_tokens)
            chunk = Chunk(content=chunk_text, metadata={**(metadata or {}), "token_count": len(current_tokens)}, parent_id=parent_id)
            all_chunks.append(chunk)

        return all_chunks

    def parse_document(self, document: Document):
        return self.parse(text=document.content, metadata=document.metadata, parent_id=document.doc_id)

    async def aparse_document(self, document: Document):
        return await sync_to_async(self.parse_document)(document)

    def parse_documents(self, documents: list[Document]):
        raise NotImplementedError("parse_documents is not implemented.")

    async def aparse_documents(self, documents: list[Document]):
        tasks = [self.aparse_document(document) for document in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunks = []
        for result in results:
            if isinstance(result, list):
                chunks.extend(result)
        return chunks
