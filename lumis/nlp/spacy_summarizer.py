from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


class SpacySummarizer:
    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initializes the Summarizer with spaCy and pyTextRank.

        :param model_name: Name of the spaCy model to use.
        """
        try:
            import spacy
        except Exception:
            raise ImportError("Please install the `sapcy` package to use the SpacySummarizer.")
        try:
            import pytextrank  # noqa: F401
        except ImportError:
            raise ImportError("Please install the `pytextrank` package to use the SpacySummarizer.")

        # Load the spaCy model
        self.nlp = spacy.load(model_name)
        # Add pyTextRank to the spaCy pipeline if not already added
        if "textrank" not in self.nlp.pipe_names:
            self.nlp.add_pipe("textrank", last=True)
        # Initialize ThreadPoolExecutor for asynchronous execution
        self.executor = ThreadPoolExecutor()

    async def summarize(self, text: str, summary_ratio: float = 0.2) -> str:
        """
        Asynchronously summarizes the given text.

        :param text: The text to summarize.
        :param summary_ratio: The ratio of sentences to include in the summary.
                              For example, 0.2 means include 20% of the sentences.
        :return: Summarized text.
        """
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(self.executor, self._sync_summarize, text, summary_ratio)
        return summary

    def _sync_summarize(self, text: str, summary_ratio: float) -> str:
        """
        Synchronously summarizes the given text.

        :param text: The text to summarize.
        :param summary_ratio: The ratio of sentences to include in the summary.
        :return: Summarized text.
        """
        # Process the text with spaCy
        doc = self.nlp(text)

        # Calculate the number of sentences to include in the summary
        sentences = list(doc.sents)
        num_sentences = max(1, int(len(sentences) * summary_ratio))

        # Get the top-ranked sentences
        top_sentences = []
        for sent in doc._.textrank.summary(limit_sentences=num_sentences):
            top_sentences.append(sent)

        # Sort the sentences based on their position in the original text
        top_sentences = sorted(top_sentences, key=lambda s: s.start_char)

        # Join the sentences to form the summary
        summary = " ".join([sent.text for sent in top_sentences])

        return summary

    def __del__(self):
        """
        Clean up the executor when the object is deleted.
        """
        self.executor.shutdown(wait=False)
