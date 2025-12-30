from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from lumis.llm import Gemini
from lumis.tools import WikipediaSearcher
from lumis.tools.search.arxiv import ArxivResult, ArxivSearcher

from arxiv import SortCriterion, SortOrder
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
wiki = WikipediaSearcher()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60), retry=retry_if_exception_type((Exception,)), reraise=True)
async def _read_arxiv_pdf(query: str, content: ArxivResult, summary_length: str = "concise", semaphore: Optional[asyncio.Semaphore] = None):
    """
    Reads and summarizes an arXiv PDF with retry logic.

    This is an internal helper function that handles the actual PDF reading and summarization process.
    It includes retry logic to handle potential failures and uses a semaphore for rate limiting.

    Args:
        content (ArxivResult): The arXiv paper result containing the PDF content
        summary_length (str): The desired length of the summary ("concise", "detailed", or "comprehensive")
        semaphore (Optional[asyncio.Semaphore]): Optional semaphore for rate limiting concurrent operations

    Returns:
        Optional[str]: The generated summary of the paper, or None if summarization fails
    """
    if semaphore:
        async with semaphore:
            return await _generate_summary(query, content, summary_length)
    return await _generate_summary(query, content, summary_length)


async def _generate_summary(query: str, content: ArxivResult, summary_length: str) -> Optional[str]:
    """
    Generates a summary of an academic paper using the Gemini LLM.

    This internal function creates a structured prompt for the LLM to generate a summary
    of the paper based on the specified length requirement.

    Args:
        content (ArxivResult): The arXiv paper result containing the paper's content
        summary_length (str): The desired length of the summary ("concise", "detailed", or "comprehensive")

    Returns:
        Optional[str]: The generated summary, or None if generation fails
    """
    gemini = Gemini()

    # Create a prompt based on the summary length
    length_instructions = {
        "concise": "Provide a brief summary (2-3 paragraphs) focusing on the key findings and contributions.",
        "detailed": "Provide a detailed summary (4-5 paragraphs) covering the main methodology, findings, and implications.",
        "comprehensive": "Provide a comprehensive summary (6-8 paragraphs) covering all major aspects including background, methodology, findings, implications, and future work.",
    }

    prompt = f"""
    Please analyze this academic paper with a specific focus on answering the following query:
    "{query}"
    
    Provide a {length_instructions[summary_length]} that addresses both the query and the paper's key aspects.
    
    Title: {content.title}
    Authors: {", ".join(content.authors)}
    Abstract: {content.abstract}
    
    Full paper content:
    {content.content}

    Please structure your analysis to include:
    1. Relevance to Query
       - How does this paper specifically address or relate to the query?
       - What key information from the paper answers the query?
    
    2. Main Research Context
       - Main research question/objective
       - Key methodology/approach
       - Major findings/results
    
    3. Query-Specific Analysis
       - Direct answers or insights related to the query
       - Supporting evidence from the paper
       - Any limitations or caveats in addressing the query
    
    4. Additional Context
       - Significance/contributions
       - Limitations/future work (if mentioned)
       - Other relevant information that provides broader context

    Format the analysis in clear paragraphs with appropriate section headers.

    IMPORTANT: Respond ONLY with the formatted summary. Do not include any meta-commentary, introductions, or explanations about what you are going to do. Start directly with the summary content.
    """

    response = await gemini.generate_content(contents=prompt)
    return response.text if response is not None else None


async def search_arxiv(  # noqa: C901
    query: str,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    max_results: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Searches arXiv for scientific and technical research papers.

    This function is specifically designed for accessing peer-reviewed scientific papers and technical documentation
    from arXiv, a repository of scientific research. It's best used for:
    - Technical specifications and methodologies
    - Scientific research findings
    - Mathematical proofs and theoretical frameworks
    - Computer science algorithms and implementations
    - Physics, mathematics, statistics, and quantitative biology research

    This tool should NOT be used for:
    - General knowledge questions
    - Non-technical topics
    - Opinion pieces or news articles
    - Historical or cultural information

    The function performs a comprehensive search of arXiv papers, downloads their PDFs,
    and generates summaries using an LLM. It handles concurrent operations with rate limiting
    to prevent overwhelming the arXiv API and the LLM service.

    Query Guidelines:
    - Focus on technical and scientific terminology
    - Use field-specific keywords and concepts
    - Include relevant scientific categories using "cat:category_name"
    - Use Boolean operators (AND, OR, NOT) for precise filtering
    - Use quotes for exact phrase matching

    Query Examples:
    Technical Research:
     - "What are the recent developments in transformer architecture?"
     - "cat:cs.AI transformer architecture attention mechanism"
     - "quantum error correction cat:quant-ph"

    Specific Paper:
     - "id:2303.08774"

    Args:
        query (str): The search query string formatted as a technical question or specific paper identifier
        sort_by (str): Sort criterion - one of "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order (str): Sort order - either "ascending" or "descending"
        max_results (Optional[int]): Maximum number of results to return

    Returns:
        list[dict[str, Any]]: List of dictionaries containing paper information including:
            - type: Always "arxiv"
            - title: Paper title
            - authors: List of author names
            - abstract: Paper abstract
            - url: URL to the PDF
            - arxiv_id: arXiv identifier
            - published_date: Publication date
            - updated_date: Last update date
            - categories: List of arXiv categories
            - primary_category: Primary arXiv category
            - comment: Any comments
            - journal_ref: Journal reference if available
            - doi: Digital Object Identifier if available
            - content: Generated summary of the paper
            - metadata: Additional metadata
    """
    searcher = ArxivSearcher(max_concurrent_pdfs=5)
    summary_semaphore = asyncio.Semaphore(4)

    try:
        # Convert string sort parameters to enums
        sort_criterion = SortCriterion(sort_by)
        sort_order_enum = SortOrder(sort_order)

        results = await searcher.search(
            query=query,
            sort_by=sort_criterion,
            sort_order=sort_order_enum,
            max_results=max_results,
            read_pdfs=True,
        )

        # Create tasks for concurrent summarization
        summary_tasks = []
        for result in results:
            if result.content is None:
                continue
            summary_tasks.append(_read_arxiv_pdf(query=query, content=result, summary_length="concise", semaphore=summary_semaphore))

        # Wait for all summaries to complete
        if summary_tasks:
            summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
            for result, summary in zip(results, summaries):
                if isinstance(summary, Exception):
                    logger.error(f"Failed to generate summary: {summary}")
                    result.content = None
                else:
                    result.content = str(summary)

        # Convert results to dictionaries with all available fields
        return [
            {
                "type": "arxiv",
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "url": result.pdf_url,
                "arxiv_id": result.arxiv_id,
                "published_date": result.published_date,
                "updated_date": result.updated_date,
                "categories": result.categories,
                "primary_category": result.primary_category,
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "doi": result.doi,
                "content": result.content,
                "metadata": result.metadata,
            }
            for result in results
        ]
    finally:
        await searcher.close()


async def wiki_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Searches Wikipedia for articles matching the given query.

    This function performs a Wikipedia search and retrieves basic information about
    the matching articles, including their titles, summaries, and URLs.

    Args:
        query (str): The search query string
        num_results (int): Number of results to return (default: 3, maximum: 5)

    Returns:
        list[dict]: List of dictionaries containing article information:
            - title: Article title
            - summary: Brief summary of the article
            - url: URL to the Wikipedia page
            - content: The article content
            - categories: List of Wikipedia categories the article belongs to
    """
    if num_results > 5:
        num_results = 5

    results = await wiki.search(query, num_results)
    return [
        {
            "title": page.title,
            "summary": page.summary,
            "url": page.url,
            "content": page.content,
            "categories": page.categories,
        }
        for page in results
    ]


async def web_search(query: str) -> Optional[dict]:
    """
    Performs a web search using Gemini's search capabilities.

    Uses LLM Powered Web Search to generate a comprehensive answer based on the search results.

    Query Guidelines:
    - Format your query as a clear, specific question
    - Include relevant context and scope
    - Use precise terminology and avoid ambiguity
    - Focus on one main topic or aspect per query
    - Avoid overly broad or vague questions

    Examples of good queries:
    - "What are the key differences between Python's asyncio and threading for concurrent programming?"
    - "How does React's virtual DOM improve performance compared to direct DOM manipulation?"
    - "What are the best practices for implementing JWT authentication in a microservices architecture?"

    Examples of poor queries:
    - "Tell me about Python" (too broad)
    - "How to code?" (too vague)
    - "What's the best way?" (lacks context)

    Args:
        query (str): The search query string, formatted as a specific, well-formed question.

    Returns:
        Optional[dict]: Dictionary containing:
            - answer: A comprehensive answer to the query
            - results: List of search results with title, snippet, and url
        Returns None if the search fails
    """
    gemini = Gemini()

    try:
        response = await gemini.generate_content(
            contents=query,
            use_search=True,
        )

        extracted_response = await gemini.extract_response_sources_and_answer(response)

        return extracted_response.model_dump()

    except Exception as e:
        # Log error and update trace
        logger.error(f"Error during web search: {e}")
        return None
