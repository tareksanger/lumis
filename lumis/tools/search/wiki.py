import asyncio
import logging
from typing import Optional

from lumis.common import LoggerMixin
from lumis.npl import LanguageProcessor

from pydantic import BaseModel, Field
import wikipedia


class WikiResult(BaseModel):
    page_id: str
    title: str
    summary: str
    content: str
    url: str


class ReleventWikiResult(WikiResult):
    similarity: float = Field(default=0)


class WikipediaSearcher(LoggerMixin):
    """
    A class to handle Wikipedia search operations using the 'wikipedia' package with integrated logging.
    """

    def __init__(self, user_agent: Optional[str] = "lumis/1.0 (https://lumis.com):tech@lumis.com", logger: Optional[logging.Logger] = None):
        """
        Initializes the WikipediaSearcher.

        Args:
            language_processor (LanguageProcessor): An instance of LanguageProcessor for NLP tasks.
            logger (Optional[logging.Logger]): An existing logger to use. If None, a new logger is created.
        """
        super().__init__(logger=logger)
        self.language_processor = LanguageProcessor()
        self.logger.info("WikipediaSearcher initialized.")

        # Set a default User-Agent without contact information (not recommended)
        self.user_agent = user_agent
        wikipedia.set_user_agent(self.user_agent)
        self.logger.debug(f"User-Agent set to: {self.user_agent}")

    async def search(self, query: str, num_results: int = 5, lang: str = "en") -> list[WikiResult]:
        """
        Searches Wikipedia for the given query and retrieves page details.

        Args:
            query (str): The search query.
            num_results (int, optional): Number of search results to return. Defaults to 5.
            lang (str, optional): Language code for Wikipedia. Defaults to "en".

        Returns:
            List[Dict[str, Any]]: A list of search results, each containing 'title', 'url', and 'summary'.
        """
        wikipedia.set_lang(lang)
        self.logger.info(f"Initiating Wikipedia search for query: '{query}' in language '{lang}'")

        try:
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(None, wikipedia.search, query, num_results)

            self.logger.debug(f"Retrieved {len(search_results)} search results from Wikipedia for query: '{query}'")

            if not search_results:
                self.logger.warning(f"No search results found on Wikipedia for query: '{query}'")
                return []

            results: list[WikiResult] = []
            for idx, title in enumerate(search_results, start=1):
                try:
                    page = await loop.run_in_executor(None, wikipedia.page, title)
                    content = page.content.strip()
                    summary = page.summary.strip()
                    url = page.url
                    page_id = page.pageid

                    result = WikiResult(page_id=page_id, title=title, url=url, summary=summary, content=content)

                    results.append(result)
                    self.logger.info(f"Result {idx}: '{title}' retrieved successfully.")
                except wikipedia.DisambiguationError as e:
                    self.logger.warning(f"DisambiguationError for title '{title}': {e.options}. Skipping.")
                except wikipedia.PageError:
                    self.logger.warning(f"PageError: The page '{title}' does not exist. Skipping.")
                except Exception as e:
                    self.log_exception(e)

            self.logger.info(f"Total results retrieved: {len(results)}")
            return results

        except Exception as e:
            self.log_exception(e)
            return []

    async def get_page_details(self, title: str, lang: str = "en") -> Optional[WikiResult]:
        """
        Retrieves detailed information about a specific Wikipedia page.

        Args:
            title (str): The title of the Wikipedia page.
            lang (str, optional): Language code for Wikipedia. Defaults to "en".

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing 'title', 'url', and 'summary' if the page exists; otherwise, None.
        """
        wikipedia.set_lang(lang)
        self.logger.info(f"Retrieving details for Wikipedia page: '{title}' in language '{lang}'")

        try:
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(None, wikipedia.page, title)
            content = page.content.strip()
            summary = page.summary.strip()
            url = page.url
            page_id = page.pageid

            result = WikiResult(page_id=page_id, title=title, url=url, summary=summary, content=content)

            self.logger.info(f"Wikipedia page '{title}' retrieved successfully.")
            return result

        except wikipedia.DisambiguationError as e:
            self.logger.warning(f"DisambiguationError for title '{title}': {e.options}.")
        except wikipedia.PageError:
            self.logger.warning(f"PageError: The page '{title}' does not exist.")
        except Exception as e:
            self.log_exception(e)

        return None

    async def search_and_get_relevant(self, query: str, num_results: int = 5, threshold: float = 0.8, lang: str = "en") -> list[ReleventWikiResult]:
        """
        Searches Wikipedia and returns the most relevant results based on similarity scores.

        Args:
            query (str): The search query.
            num_results (int, optional): Number of top relevant results to return. Defaults to 5.
            threshold (float, optional): Similarity threshold to determine relevance. Defaults to 0.8.
            lang (str, optional): Language code for Wikipedia. Defaults to "en".

        Returns:
            List[Dict[str, Any]]: A list of the most relevant search results, each containing 'title', 'url', 'summary', and 'similarity'.
        """
        self.logger.info(f"Starting search and relevance extraction for query: '{query}'")

        try:
            search_results = await self.search(query, num_results=num_results * 2, lang=lang)

            if not search_results:
                self.logger.warning(f"No search results to process for query: '{query}'")
                return []

            relevant_results = []

            for idx, result in enumerate(search_results, start=1):
                title = result.title
                summary = result.summary
                combined_text = f"{title}. {summary}"

                # Compute similarity using the LanguageProcessor
                similarity = self.language_processor.similarity(query, combined_text)
                self.logger.debug(f"Result {idx}: '{title}' has similarity {similarity:.2f}")

                if similarity >= threshold:
                    relevant_result = ReleventWikiResult(**result.model_dump(), similarity=similarity)
                    relevant_results.append(relevant_result)
                    self.logger.info(f"Result {idx} ('{title}') deemed relevant with similarity {similarity:.2f}")

                    if len(relevant_results) >= num_results:
                        self.logger.debug(f"Desired number of relevant results ({num_results}) achieved.")
                        break

            if not relevant_results:
                self.logger.warning(f"No relevant Wikipedia results met the threshold of {threshold} for query: '{query}'")

            self.logger.info(f"Total relevant results found: {len(relevant_results)}")
            return relevant_results

        except Exception as e:
            self.log_exception(e)
            return []
