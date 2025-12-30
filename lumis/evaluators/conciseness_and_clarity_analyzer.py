from __future__ import annotations

import re
from typing import Dict
import zlib

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import textstat


class TextConcisenessAnalyzer:
    """
    A class to analyze the conciseness of a text string using various methods.

    This class provides static methods to measure different aspects of text conciseness,
    such as average sentence length, filler word counts, compression ratio, information
    density, and paraphrasing comparison.

    Usage:
        - Instantiate the class (optional since methods are class methods).
        - Call the desired method with the text you want to analyze.

    Dependencies:
        - NLTK library for natural language processing tasks.
        - Transformers library for paraphrasing (if using paraphrase method).
        - Standard Python libraries: re, zlib, collections.

    Note:
        - Ensure necessary NLTK data packages are downloaded.
        - For paraphrasing, internet connection is required to load the model.
    """

    @classmethod
    def install_requirements(cls):
        try:
            import nltk
        except ImportError:
            raise ImportError("Please install the 'nltk' library to use this method.")

        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("averaged_perceptron_tagger_eng")

    @classmethod
    def measure_lengths(cls, text: str) -> Dict[str, float]:
        """
        Calculate average sentence and word lengths in the text.

        How to use:
            Call this method with the text you want to analyze.

        When to use:
            Use this method when you want to assess the verbosity of the text
            based on sentence and word lengths.

        What the scores mean:
            - 'average_sentence_length': Average number of words per sentence.
              Lower values may indicate more concise sentences.
            - 'average_word_length': Average number of characters per word.
              Shorter words can contribute to conciseness and readability.

        Args:
            text (str): The input text string to analyze.

        Returns:
            Dict[str, float]: A dictionary containing:
                - 'average_sentence_length' (float)
                - 'average_word_length' (float)
        """
        sentences = re.split(r"[.!?]+", text)
        words = re.findall(r"\b\w+\b", text)

        avg_sentence_length = len(words) / max(len(sentences), 1) if words else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        return {"average_sentence_length": avg_sentence_length, "average_word_length": avg_word_length}

    @classmethod
    def compression_ratio(cls, text: str) -> float:
        """
        Calculate the compression ratio of the text using zlib compression.

        How to use:
            Call this method with the text you want to analyze.

        When to use:
            Use this method to assess redundancy in the text. Highly compressible text
            may contain more redundancy.

        What the score means:
            - A ratio closer to 1 indicates less redundancy (more concise).
            - A lower ratio indicates higher redundancy (less concise).

        Args:
            text (str): The input text string to analyze.

        Returns:
            float: The compression ratio of the text.
        """
        original_size = len(text.encode("utf-8"))
        if original_size == 0:
            return 0.0
        compressed_data = zlib.compress(text.encode("utf-8"))
        compressed_size = len(compressed_data)
        ratio = compressed_size / original_size
        return ratio

    @classmethod
    def information_density(cls, text: str) -> float:
        """
        Calculate the information density of the text.

        How to use:
            Call this method with the text you want to analyze.

        When to use:
            Use this method to determine the proportion of meaningful content words
            (nouns, verbs, adjectives, adverbs) in the text.

        What the score means:
            - A higher value (closer to 1) indicates higher information density (more concise).
            - A lower value indicates more function words and potentially less conciseness.

        Args:
            text (str): The input text string to analyze.

        Returns:
            float: The information density ratio.
        """
        words = word_tokenize(text)
        if not words:
            return 0.0
        tagged_words = pos_tag(words)
        content_words = [word for word, pos in tagged_words if pos.startswith(("N", "V", "J", "R"))]
        density = len(content_words) / len(words)
        return density

    @classmethod
    def composite_readability_score(cls, text: str, alpha: float = 10.0) -> float:
        """
        Calculate a composite readability score for a given text.

        The composite score is defined as:
            Composite Score = Flesch Reading Ease - (alpha * Flesch-Kincaid Grade Level)

        Parameters:
        - text (str): The text to evaluate.
        - alpha (float): The weighting factor for the grade level penalty (default: 10.0).

        Returns:
        - float: The composite readability score. Higher scores indicate better clarity.
        """

        # Compute the readability metrics
        reading_ease = textstat.flesch_reading_ease(text)  # type: ignore
        grade_level = textstat.flesch_kincaid_grade(text)  # type: ignore

        # Calculate the composite score
        score = reading_ease - (alpha * grade_level)

        return score

    @classmethod
    def is_too_verbose(cls, text: str, max_avg_sentence_length: float = 50.0, max_clauses: int = 5) -> bool:
        """
        Determines if a prompt is likely too verbose for an agent to process effectively.
        """
        metrics = cls.measure_lengths(text)
        num_clauses = len(re.split(r"\band\b|\bor\b|,|;", text))  # crude clause count

        return metrics["average_sentence_length"] > max_avg_sentence_length or num_clauses > max_clauses

    @classmethod
    def tokenize_sentence(cls, sentence: str) -> set[str]:
        """Lowercase, remove punctuation, and split into a set of words."""
        return set(re.findall(r"\b\w+\b", sentence.lower()))

    @classmethod
    def jaccard(cls, a: str, b: str) -> float:
        """Compute Jaccard similarity between two sentences as sets of words."""
        set_a = cls.tokenize_sentence(a)
        set_b = cls.tokenize_sentence(b)
        intersection: int = len(set_a & set_b)
        union: int = len(set_a | set_b)
        return intersection / union if union else 0.0
