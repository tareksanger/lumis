from __future__ import annotations

import logging
import os
from typing import Optional, TypeVar, Union

from lumis.core.utils.helpers import merge_pydantic_models
from lumis.llm.base_llm import BaseLLM

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    ContentListUnion,
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    GenerateImagesConfigOrDict,
    GenerateImagesResponse,
    GenerateVideosConfigOrDict,
    GenerateVideosResponse,
    GoogleSearch,
    ImageOrDict,
    Tool,
)
import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


async def _get_final_url(url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308) and "location" in response.headers:
                return response.headers["location"]
            return str(response.url)
    except Exception as e:
        logging.error(f"Error getting final URL for {url}: {e}")
        return url


CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

ResponseT = TypeVar("ResponseT")


API_KEY = os.getenv("GEMINI_API_KEY")


"""
Docs: https://ai.google.dev/gemini-api/docs/text-generation
"""


class SearchSegment(BaseModel):
    """Model for a segment of text from a Gemini response with its confidence score."""

    text: str = Field(description="The text segment from the response")
    confidence: float = Field(description="Confidence score for this segment (0.0 to 1.0)", ge=0.0, le=1.0)


class GeminiSource(BaseModel):
    """Model for a source used in a Gemini response."""

    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    confidence: float = Field(
        description="Overall confidence score for this source (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    segments: list[SearchSegment] = Field(default_factory=list, description="List of text segments that used this source")
    summary: str = Field(default="", description="Summary of the source content")


class ExtractedSearchResponse(BaseModel):
    """Model for a complete Gemini response with answer and sources."""

    answer: str = Field(description="The complete answer text")
    sources: list[GeminiSource] = Field(default_factory=list, description="List of sources used in the response")


class Gemini(BaseLLM):
    """
    Gemini LLM
    """

    google_search_tool = Tool(
        google_search=GoogleSearch(),
    )

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize BaseLLM
        super().__init__(verbose=verbose, logger=logger)

        if client is None:
            client = genai.Client(api_key=API_KEY)

        self.__client = client

    @property
    def client(self) -> genai.Client:
        return self.__client

    def _count_tokens(  # noqa: C901
        self,
        response: (GenerateContentResponse | GenerateImagesResponse | GenerateVideosResponse),
    ) -> GenerateContentResponse | GenerateImagesResponse | GenerateVideosResponse:
        """
        Counts the tokens in the response.
        """
        # with sentry_sdk.start_span(op="llm.token.count", name="Gemini LLM") as span:
        if not isinstance(response, GenerateContentResponse):
            self.logger.debug(f"Skipping token count for {type(response)} token count not yet supported for this response type.")
            return response

        try:
            if usage := response.usage_metadata:
                # For Sentry tracing WILL ADD LATER
                # prompt_tokens = usage.prompt_token_count
                # cache_tokens = usage.cache_token_count
                # total_tokens = usage.total_token_count

                usage_dict = usage.model_dump(exclude_none=True)

                if "prompt_tokens_details" in usage_dict:
                    usage_dict.pop("prompt_tokens_details")
                if "cache_tokens_details" in usage_dict:
                    usage_dict.pop("cache_tokens_details")
                if "candidates_tokens_details" in usage_dict:
                    usage_dict.pop("candidates_tokens_details")

                self._token_count.update(usage_dict)

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
        return response

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_content(
        self,
        model: str = CHAT_MODEL,
        contents: Union[ContentListUnion, ContentListUnionDict, str] = [],
        config: Optional[GenerateContentConfigOrDict] = None,
        use_search: bool = True,
    ):
        if isinstance(contents, str):
            contents = [contents]

        # Always merge with search config
        if use_search:
            config = self.__default_search_config(config)

        try:
            self.logger.debug(f"Generating text with model={model}, contents_type={type(contents)} contents_count={len(contents) if isinstance(contents, list) else 1}")

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            self.logger.debug("Received response from Gemini API")
            response = await self._apply_middlewares(response)
            return response

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_content_stream(  # noqa: C901
        self,
        model: str = CHAT_MODEL,
        contents: Union[ContentListUnion, ContentListUnionDict, str] = [],
        config: Optional[GenerateContentConfigOrDict] = None,
        use_search: bool = False,
    ):
        self.logger.debug(f"Generating text with model={model}, contents_type={type(contents)} contents_count={len(contents) if isinstance(contents, list) else 1}")

        try:
            if use_search:
                config = self.__default_search_config(config)

            stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            async for chunk in await stream:
                chunk = await self._apply_middlewares(chunk)
                yield chunk

        except ClientError as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

        except ServerError as e:
            self.log_exception(e, level=logging.ERROR)
            return
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str = "veo-2.0-generate-001",
        image: Optional[ImageOrDict] = None,
        config: Optional[GenerateVideosConfigOrDict] = None,
    ):
        config_keys = config.model_dump(exclude_none=True).keys() if isinstance(config, BaseModel) else "None" if config is None else config.keys()
        self.logger.debug(f"Generating videos with model={model}, prompt_type={type(prompt)} image_type={type(image)} config_keys={config_keys}")
        try:
            response = await self.client.aio.models.generate_videos(
                model=model,
                prompt=prompt,
                image=image,
                config=config,
            )

            self.logger.debug("Received response from Gemini API")
            response = await self._apply_middlewares(response)
            return response

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    # TODO: Configure model typing
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_images(
        self,
        prompt: str,
        model: str = "imagen-3.0-generate-002",
        config: Optional[GenerateImagesConfigOrDict] = None,
    ):
        self.logger.debug(
            f"Generating image with model={model}, prompt_type={type(prompt)} config_keys={config.model_dump(exclude_none=True).keys() if isinstance(config, BaseModel) else 'None' if config is None else config.keys()}"  # noqa: E501
        )

        try:
            response = await self.client.aio.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )

            self.logger.debug("Received response from Gemini API")
            response = await self._apply_middlewares(response)
            return response

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    def __default_search_config(self, config: Optional[GenerateContentConfigOrDict] = None) -> GenerateContentConfig:
        """
        Adds a Google Search tool to the config if it is not already present.
        """
        self.logger.debug("Adding Google Search tool to config")
        config_with_search = GenerateContentConfig(
            tools=[self.google_search_tool],
            response_modalities=["TEXT"],
        )

        if config is None:
            self.logger.debug("Config is None, returning config with search")
            return config_with_search

        self.logger.debug("Config is not None, merging config with search")
        config = config if isinstance(config, GenerateContentConfig) else GenerateContentConfig(**config)  # type: ignore
        return merge_pydantic_models(config, config_with_search)

    @classmethod
    async def extract_response_sources_and_answer(cls, response: GenerateContentResponse):  # noqa: C901
        """
        Extract answer and sources with their supporting segments from a Gemini response object.
        Also fetches and summarizes content from URLs using UrlContext.

        Args:
            response: The Gemini GenerateContentResponse object

        Returns:
            ExtractedSearchResponse containing answer and sources with summaries
        """
        try:
            # Extract the answer text from the first candidate
            answer = response.text
            # If the answer is empty, try to extract it from the first candidate
            if not answer and response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts or []:
                    if part.text:
                        answer = part.text
                        break

            # Extract sources from grounding metadata
            sources = []
            source_map = {}  # url -> source index

            if response.candidates and response.candidates[0].grounding_metadata:
                grounding = response.candidates[0].grounding_metadata

                # First pass: collect all unique sources
                for chunk in grounding.grounding_chunks or []:
                    if chunk.web and chunk.web.uri and chunk.web.uri not in source_map:
                        source_map[chunk.web.uri] = len(sources)
                        sources.append(
                            {
                                "title": chunk.web.title or "Unknown Source",
                                "url": await _get_final_url(chunk.web.uri),
                                "confidence": 0.0,
                                "segments": [],
                            }
                        )

                # Second pass: process grounding supports to add segments
                if grounding.grounding_supports:
                    for support in grounding.grounding_supports:
                        if support.segment and support.grounding_chunk_indices and support.confidence_scores:
                            segment_text = support.segment.text
                            for idx, score in zip(
                                support.grounding_chunk_indices,
                                support.confidence_scores,
                            ):
                                if idx < len(sources):
                                    sources[idx]["confidence"] = max(sources[idx]["confidence"], float(score))
                                    # Add segment to source
                                    sources[idx]["segments"].append(
                                        {
                                            "text": segment_text,
                                            "confidence": float(score),
                                        }
                                    )

            # Convert to Pydantic models
            gemini_sources = [GeminiSource(**source) for source in sources]
            return ExtractedSearchResponse(answer=str(answer), sources=gemini_sources)

        except Exception as e:
            logging.error(f"Error extracting Gemini response: {e}")
            return ExtractedSearchResponse(answer="", sources=[])
