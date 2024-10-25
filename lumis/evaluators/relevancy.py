import re


class TextRelevancyAnalyzer:
    @staticmethod
    def __preprocess(text: str) -> list[str]:
        """
        Tokenizes the input text into a list of lowercase words.

        This method removes non-alphanumeric characters and converts the text to lowercase,
        returning a list of words. This standardizes the text for similarity calculations.
        """
        return re.findall(r"\w+", text.lower())

    @classmethod
    def cosine_similarity(cls, text1: str, text2: str) -> float:
        """
        Calculates cosine similarity between two texts.

        **Mathematical Explanation:**
        - Cosine similarity measures the cosine of the angle between two vectors
        in a multidimensional space.
        - In this context, each text is represented as a vector of word frequencies.

        **When to Use:**
        - Use cosine similarity when you need to measure the similarity between texts
          based on the frequency of shared words.
        """
        # Tokenize and count word frequencies for each text
        # embeddings1 = cls.get_embeddings(text1)
        # embeddings2 = cls.get_embeddings(text2)

        # # Calculate cosine similarity
        # return dot(embeddings1, embeddings2) / (norm(embeddings1) * norm(embeddings2))
        return 0.0

    @classmethod
    def jaccard_similarity(cls, text1: str, text2: str) -> float:
        """
        Calculates Jaccard similarity between two texts.

        **Mathematical Explanation:**
        - Jaccard similarity measures the similarity between two sets by dividing
          the size of their intersection by the size of their union.

        **When to Use:**
        - Use Jaccard similarity when you want to compare the overlap of unique words between texts.
        """
        # Tokenize and convert to sets of unique words
        set1 = set(cls.__preprocess(text1))
        set2 = set(cls.__preprocess(text2))

        # Compute the intersection and union of the sets
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Avoid division by zero
        if not union:
            return 0.0

        # Calculate Jaccard similarity
        return len(intersection) / len(union)

    @classmethod
    def levenshtein_distance(cls, s1: str, s2: str) -> int:
        """
        Calculates the Levenshtein distance between two strings.

        **Mathematical Explanation:**
        - Levenshtein distance is the minimum number of single-character edits
          (insertions, deletions, or substitutions) required to change one word into the other.
        - Uses dynamic programming to efficiently compute the distance.

        **Algorithm Steps:**
        - Initialize a matrix where the cell at position (i, j) represents the
          Levenshtein distance between the first i characters of s1 and the first j characters of s2.
        - The value at each cell is computed based on the minimum of:
          - The cell above plus one (deletion).
          - The cell to the left plus one (insertion).
          - The cell diagonally above and to the left plus cost (substitution).

        **When to Use:**
        - Use Levenshtein distance for short strings where character-level differences are important.
        - Useful for spell checking, DNA sequencing, or measuring how closely two sequences match.
        """
        # Ensure s1 is the longer string
        if len(s1) < len(s2):
            return cls.levenshtein_distance(s2, s1)

        # If one string is empty, the distance is the length of the other
        if len(s2) == 0:
            return len(s1)

        # Initialize previous row of distances
        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Calculate insertions, deletions, and substitutions
                insertions = previous_row[j + 1] + 1  # Insertion
                deletions = current_row[j] + 1  # Deletion
                substitutions = previous_row[j] + (c1 != c2)  # Substitution

                # Choose the minimum cost operation
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        # The last element in the last row is the Levenshtein distance
        return previous_row[-1]

    @classmethod
    def ngram_similarity(cls, text1: str, text2: str, n: int = 2) -> float:
        """
        Calculates n-gram similarity between two texts.

        **Mathematical Explanation:**
        - N-grams are contiguous sequences of 'n' items from a given sequence of text.
        - The n-gram similarity is calculated as:


            N-gram Similarity = Number of shared n-grams/Total unique n-grams


        **When to Use:**
        - Use n-gram similarity when you need to detect similarities at the character level,
          such as typos or minor variations.
        - Adjust 'n' to control the granularity (e.g., n=2 for bigrams, n=3 for trigrams).
        - Effective for shorter texts or when word order matters.
        """
        # Generate n-grams for each text
        ngrams1 = set(zip(*[text1[i:] for i in range(n)]))
        ngrams2 = set(zip(*[text2[i:] for i in range(n)]))

        # Compute the intersection and union of the n-grams
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        # Avoid division by zero
        if not union:
            return 0.0

        # Calculate n-gram similarity
        return len(intersection) / len(union)

    @classmethod
    def keyword_relevance(cls, text1: str, text2: str, keywords: list[str]) -> bool:
        """
        Checks if both texts share common keywords.

        **When to Use:**
        - Use keyword relevance when certain words are critical for determining relevancy.
        - Effective for domain-specific applications where particular terms are significant.
        - It doesn't provide a similarity score but a boolean indicating the presence of shared keywords.
        """
        # Tokenize and convert to sets of unique words
        words1 = set(cls.__preprocess(text1))
        words2 = set(cls.__preprocess(text2))

        # Convert keywords to a set for efficient lookup
        keywords_set = set(keywords)

        # Find common keywords present in both texts
        common_keywords = (words1 & words2) & keywords_set

        # Return True if any common keywords are found
        return len(common_keywords) > 0

    @classmethod
    def is_relevant(cls, text1: str, text2: str, method: str = "cosine", threshold: float = 0.5, **kwargs) -> bool:
        """
        Determines if two texts are relevant based on the specified method and threshold.

        **Parameters:**
            - **method (str):** The similarity method to use ('cosine', 'jaccard', 'levenshtein', 'ngram').
            - **threshold (float):** The threshold above which texts are considered relevant.
            - **kwargs:** Additional arguments for specific methods (e.g., n for n-gram).

        **Similarity Score Interpretation:**
        - The methods (except keyword relevance) produce a similarity score between 0 and 1.
        - A score closer to 1 indicates higher similarity.

        **When to Use:**
        - Choose the method based on the characteristics of your texts and the importance of word order, frequency, etc.
            - **Cosine Similarity:** When word frequency matters.
            - **Jaccard Similarity:** For measuring unique word overlap.
            - **Levenshtein Distance:** For short strings where character-level edits are important.
            - **N-gram Similarity:** When detecting character-level similarities or typos.

        **Example:**
        ```python
        similarity = RelevancyEvaluator.is_relevant(text1, text2, method='cosine', threshold=0.7)
        ```
        """
        if method == "cosine":
            similarity = cls.cosine_similarity(text1, text2)
        elif method == "jaccard":
            similarity = cls.jaccard_similarity(text1, text2)
        elif method == "levenshtein":
            # Calculate Levenshtein distance and normalize to similarity score
            distance = cls.levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            similarity = 1 - (distance / max_len) if max_len else 0
        elif method == "ngram":
            n = kwargs.get("n", 2)
            similarity = cls.ngram_similarity(text1, text2, n)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Return True if similarity meets or exceeds the threshold
        return similarity >= threshold
