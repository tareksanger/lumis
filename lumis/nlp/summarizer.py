from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Literal

logger = logging.getLogger(__name__)

SummarizerModels = Literal[
    # BART: A larger, more accurate model suitable when the highest quality of summarization is required.
    # "facebook/bart-large-cnn",
    # DISTILBART: A faster model suitable for applications where speed and resource efficiency are crucial.
    "sshleifer/distilbart-cnn-12-6",
]

MAX_TOKENS = {
    # "facebook/bart-large-cnn": 1024,
    "sshleifer/distilbart-cnn-12-6": 1024,
}

# TODO: Revisit


class Summarizer:
    def __init__(
        self,
        model_name: SummarizerModels = "sshleifer/distilbart-cnn-12-6",
        use_gpu: bool = False,
    ):
        """
        Initialize the summarizer with a specified Hugging Face model.

        :param model_name: Hugging Face model name for summarization.
        :param use_gpu: Whether to use GPU for inference.
        """
        try:
            import torch
        except Exception:
            raise ImportError("Torch is not installed.")

        try:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except Exception:
            raise ImportError("Transformers is not installed.")

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_name: str = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        self.max_model_length = MAX_TOKENS.get(self.model_name, 1024)
        logger.info(f"Maximum model length: {self.max_model_length}")

        # Initialize ThreadPoolExecutor
        self.executor = ThreadPoolExecutor()

    async def summarize(self, text: str, split_long_text: bool = False) -> str:
        """
        Summarizes the given text.

        :param text: The text to summarize.
        :param split_long_text: Whether to split long texts into chunks for summarization.
        :return: Summarized text.
        """
        loop = asyncio.get_running_loop()
        summary: str = await loop.run_in_executor(self.executor, self._sync_summarize, text, split_long_text)
        return summary

    def _sync_summarize(self, text: str, split_long_text: bool) -> str:
        """
        Synchronously summarizes the text.

        :param text: The text to summarize.
        :param split_long_text: Whether to split long texts into chunks for summarization.
        :return: Summarized text.
        """
        # Tokenize without truncation to get the true token count
        tokenized_text = self.tokenizer(
            text,
            truncation=False,
            return_tensors="pt",
        )

        num_tokens = tokenized_text["input_ids"].size(1)  # type: ignore
        logger.info(f"Number of tokens in input text: {num_tokens}")

        if num_tokens <= self.max_model_length:
            # Text fits within the model's context window
            inputs = self.tokenizer(
                text,
                max_length=self.max_model_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            summary = self._generate_summary(inputs)
            return summary
        else:
            if split_long_text:
                # Split text into manageable chunks
                chunks: list[str] = self._split_text_into_chunks(text)

                # Summarize each chunk
                summaries = [self._sync_summarize(chunk, split_long_text=False) for chunk in chunks]

                # Combine summaries into one
                combined_summary = " ".join(summaries)

                # If combined summary is still too long, summarize it without splitting
                return self._sync_summarize(combined_summary, split_long_text=False)
            else:
                # Truncate the text to the maximum input length
                logger.warning("Input text is too long; truncating to fit the model's maximum input length.")
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_model_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                summary = self._generate_summary(inputs)
                return summary

    def _generate_summary(self, inputs) -> str:
        """
        Generates summary from tokenized inputs.

        :param inputs: Tokenized inputs.
        :return: Summary text.
        """
        # Adjust generation parameters based on the model
        if self.model_name == "facebook/bart-large-cnn":
            num_beams = 4
            max_length = 150
            min_length = 40
        elif self.model_name == "allenai/led-base-16384":
            num_beams = 4
            max_length = 512
            min_length = 50
        else:
            num_beams = 1
            max_length = 130
            min_length = 30

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
        logger.info(f"Generated summary IDs: {summary_ids}")

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def _split_text_into_chunks(self, text: str) -> list[str]:  # noqa: C901
        """
        Splits text into chunks that fit within the model's maximum input length.

        :param text: The text to split.
        :return: List of text chunks.
        """
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty sentences
            sentence += ". "  # Add back the period
            sentence_length = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            if current_length + sentence_length <= self.max_model_length:
                current_chunk += sentence
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_length = sentence_length

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def __del__(self):
        """
        Clean up the executor when the object is deleted.
        """
        self.executor.shutdown(wait=False)
