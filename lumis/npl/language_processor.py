from typing import Optional, Tuple

from lumis.base.common.logger_mixin import LoggerMixin

from ..model_manager import ModelManager

from spacy.tokens import Span


class LanguageProcessor(LoggerMixin):
    """
    A class for handling NLP-related tasks using pre-loaded models from ModelManager.
    """

    def __init__(self):
        """
        Initializes the NLP processor with the specified SpaCy model.

        Args:
            model_name (str): The name of the SpaCy model to use.
        """
        super().__init__()

        self.model_manager = ModelManager.get_instance()

        self.nlp = self.model_manager.get_spacy_model("en_core_web_lg")
        # self.nlp_trf = self.model_manager.get_spacy_model("en_core_web_trf")
        # if not self.nlp_trf.has_pipe("experimental_coref"):
        #     self.nlp_trf.add_pipe("experimental_coref")

    def similarity(self, text1: str, text2: str) -> float:
        """
        Computes the similarity between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The similarity score between 0 and 1.
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

    def relevancy(self, subject: str, string2: str, threshold: Optional[float] = None) -> list[Tuple[str, float]]:
        """
        Extracts sentences from the context that are relevant to the assumption based on similarity.

        Args:
            assumption (str): The assumption text.
            context (str): The context text.
            threshold (float): The similarity threshold.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing relevant sentences and their similarity scores.
        """
        statement_doc = self.nlp(subject)
        paragraph_doc = self.nlp(string2)

        matching_sentences = []

        for sentence in paragraph_doc.sents:
            similarity = statement_doc.similarity(sentence)
            sentence_similarity_tuple = (sentence.text, similarity)
            if threshold is None or (similarity >= threshold):
                matching_sentences.append(sentence_similarity_tuple)

        return matching_sentences

    def chunk_text(self, text: str):
        self.logger.info("Starting text chunking with enhanced heuristics...")
        doc = self.nlp(text)
        sentences = list(doc.sents)
        chunks = []
        current_chunk = ""

        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            if i > 0 and self._is_contextually_linked(sent):
                current_chunk += (" " if current_chunk else "") + sent_text
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent_text

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _is_contextually_linked(self, sentence: Span):
        first_token = sentence[0]
        # Check for pronouns, determiners, or other indicators
        if first_token.pos_ in {"PRON", "DET"}:
            return True
        # Check for dependency relations
        for token in sentence:
            if token.dep_ in {"nsubj", "dobj"} and token.pos_ == "PRON":
                return True
        return False

    # TODO: come back to exploring co-reference models when the co-reference library has been updated
    # def chunk_text(
    #     self,
    #     text: str,
    #     max_chunk_size: int = 500,
    # ) -> list[str]:
    #     """
    #     Chunks a long text into smaller parts, ensuring each chunk contains contextually linked sentences.

    #     Args:
    #         text (str): The input text to chunk.
    #         max_chunk_size (int): The maximum number of characters per chunk.

    #     Returns:
    #         List[str]: A list of text chunks.
    #     """
    #     self.logger.info("Starting text chunking process with co-reference resolution...")
    #     doc = self.nlp_trf(text)
    #     sentences = list(doc.sents)

    #     chunks = []
    #     current_chunk = ""
    #     previous_entities = set()

    #     for i, sent in enumerate(sentences):
    #         sent_text = sent.text.strip()
    #         # Check for co-reference links
    #         if i > 0:
    #             has_coref = self._has_coreference(sent, previous_entities)
    #         else:
    #             has_coref = False

    #         # Determine whether to start a new chunk
    #         if has_coref or len(current_chunk) + len(sent_text) <= max_chunk_size:
    #             current_chunk += (" " if current_chunk else "") + sent_text
    #         else:
    #             if current_chunk:
    #                 chunks.append(current_chunk)
    #             current_chunk = sent_text

    #         # Update previous entities
    #         previous_entities.update(self._get_entities(sent))

    #     # Add the last chunk
    #     if current_chunk:
    #         chunks.append(current_chunk)

    #     self.logger.info(f"Text chunking completed. Total chunks: {len(chunks)}")
    #     return chunks

    # def _has_coreference(self, sentence, previous_entities) -> bool:
    #     """
    #     Checks if the sentence contains pronouns or references to entities in previous sentences.

    #     Args:
    #         sentence (spacy.tokens.Span): The sentence to check.
    #         previous_entities (set): A set of entity texts from previous sentences.

    #     Returns:
    #         bool: True if the sentence has coreference to previous entities, False otherwise.
    #     """
    #     for token in sentence:
    #         if token._.is_coref and token._.coref_chains:
    #             for chain in token._.coref_chains:
    #                 for mention in chain:
    #                     if mention.root.text in previous_entities:
    #                         self.logger.debug(f"Coreference detected: '{token.text}' refers to '{mention.root.text}'")
    #                         return True
    #     return False

    def get_entities(self, text: str) -> list[Tuple[str, str]]:
        """
        Extracts named entities from the input text.

        Args:
            text (str): The input text to extract entities from.

        Returns:
            list[Tuple[str, str]]: A list of tuples containing the entity and its label.
        """
        doc = self.nlp(text)
        ner = self.nlp.get_pipe("ner")
        processed = ner(doc)
        entities: list[Tuple[str, str]] = [(ent.text, ent.label_) for ent in processed.ents]
        return entities

    def sentiment(self, text: str):
        llm = self.nlp.add_pipe("llm_textcat")
        llm.add_label("INSULT")
        llm.add_label("COMPLIMENT")

        doc = self.nlp(text)
        return doc.cats
