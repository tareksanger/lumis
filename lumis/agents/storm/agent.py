from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Union

from lumis.core.document import Chunk
from lumis.core.utils.thread_runner import ThreadRunner
from lumis.embedding import BaseEmbeddingModel
from lumis.llm.openai_llm import OpenAILLM
from lumis.storage import BaseVectorDB
from lumis.tools import WikiPage, WikipediaSearcher
from lumis.tools.search.vector_search_retrieval_engine import VectorSearchRetrievalEngine

from ..base.graph_based_agent import GraphBasedAgent
from .interview_graph import InterviewGraph, State
from .models.outline import Outline, WikiSection
from .models.survey_subjects import Editor, Perspectives, RelatedSubjects

from typing_extensions import TypedDict


class InterviewResult(TypedDict):
    editor: str
    result: State  # Replace `Any` with the specific type returned by `interview.run`


class InterviewError(TypedDict):
    editor: str
    error: str


InterviewOutcome = Union[InterviewResult, InterviewError]


class StormState(TypedDict):
    """
    Typed dictionary that represents the internal state of the StormAgent's knowledge
    and workflow. This state is maintained across the steps of the agent's execution.

    Attributes:
        topic (str): The main subject or topic that the agent is building a Wikipedia-like article about.
        outline (Outline): The structured outline of the article, detailing sections and structure.
        editors (list[Editor]): A list of fictional Wikipedia editors (representing different perspectives)
                                who contribute to shaping the article.
        interview_results (list[State]): The results of interviews conducted with these editors. Each
                                         `State` typically contains the conversation and any references mentioned.
        sections (list[WikiSection]): The final drafted sections of the Wikipedia article, each represented
                                      as a `WikiSection`.
        final_refs (list[Chunk]): The final reference chunks collected during the drafting phase.
        article (str): The fully written, integrated Wikipedia-style article composed after refining the
                       outline, indexing references, and writing section drafts.
    """

    topic: str
    outline: Outline
    editors: list[Editor]
    interview_results: list[InterviewOutcome]

    sections: list[WikiSection]
    final_refs: list[Chunk]
    article: str


Events = None


class StormAgent(GraphBasedAgent[StormState, Events]):
    """
    A specialized agent designed to conduct web research and produce a Wikipedia-style article on a given topic.

    The StormAgent:
    - Initializes a workflow (graph) that outlines the steps to produce a comprehensive article.
    - Interacts with an OpenAILLM (Language Model) to generate outlines, refine them, and write drafts.
    - Conducts "interviews" with fictional editors, each providing different perspectives and insights.
    - Uses vector search retrieval to find and index references relevant to the topic.
    - Iteratively refines the outline and drafts until a complete article is produced.

    The agent uses an underlying `Graph` to manage its workflow steps (nodes) and transitions (edges).
    Each node represents a logical step: initialization, interviews, refinement, reference indexing,
    drafting, and writing the final article.

    Attributes:
        embedding (BaseEmbeddingModel): The embedding model used for semantic understanding and retrieval.
        vector_db (BaseVectorDB): A vector database interface used to store and retrieve semantic embeddings of chunks.
        llm (Optional[OpenAILLM]): The language model interface for generating text. If None, it defaults to a new OpenAILLM.
        memory (BaseMemory): The memory interface to store and retrieve conversation states or results.
        logger (Optional[logging.Logger]): A logger instance for debugging and monitoring agent behavior.
        verbose (bool): If True, logs additional debug information.

    Usage:
        agent = StormAgent(embedding=embedding_model, vector_db=vector_database)
        await agent.run("Your Topic")
    """

    def __init__(
        self,
        embedding: BaseEmbeddingModel,
        vector_db: BaseVectorDB,
        llm: Optional[OpenAILLM] = None,
        wiki: Optional[WikipediaSearcher] = None,
        logger: Optional[logging.Logger] = None,
        max_concurrent_interviews: int = 5,
        interview_timeout: float = 300.0,  # 5 minutes per interview
        verbose: bool = False,
    ) -> None:
        """
        Initialize the StormAgent with the required models and tools.

        Args:
            embedding (BaseEmbeddingModel): Embedding model for semantic understanding.
            vector_db (BaseVectorDB): Vector database to store and retrieve semantic chunks.
            llm (Optional[OpenAILLM]): Language model for generating outlines, drafts, and refined articles.
            memory (BaseMemory): Memory implementation to store agent states and/or conversations.
            logger (Optional[logging.Logger]): Logger instance for monitoring internal states.
            verbose (bool): If True, print additional debug information.
        """
        super().__init__(llm, verbose=verbose, logger=logger)

        self.vector_db = vector_db
        self.embedding = embedding

        self.wiki = wiki if wiki is not None else WikipediaSearcher()

        self.max_concurrent_interviews = max_concurrent_interviews
        self.interview_timeout = interview_timeout

    def construct_graph(self):
        """
        Construct the internal workflow graph of the StormAgent.

        This graph outlines the sequence of steps (nodes) and their transitions:
        1. "initialize" -> Initialize the state with a given topic, outline, and editor perspectives.
        2. "interview" -> Conduct interviews with the editors to gather insights and references.
        3. "refine_outline" -> Refine the outline based on the interviews.
        4. "index_references" -> Index references extracted from interviews into the vector DB.
        5. "draft" -> Write draft sections of the article using the refined outline and references.
        6. "write_article" -> Produce the final integrated article from the drafted sections.
        """
        self.graph.add_node("initialize", self.initialize_storm, "start")
        self.graph.add_node("interview", self.conduct_interviews)
        self.graph.add_node("refine_outline", self.refine_outline)
        self.graph.add_node("index_references", self.index_references)
        self.graph.add_node("draft", self.write_draft)
        self.graph.add_node("write_article", self.write_article)

        self.graph.chain("initialize", "interview", "refine_outline", "index_references", "draft", "write_article")

    async def setup(self, topic: str):
        """
        Initialize the agent with a topic. Sets the initial state for the graph and marks the agent as initialized.

        Args:
            topic (str): The main topic to build the Wikipedia-like article about.
        """
        self.graph.set_initial_state({"topic": topic})

    async def initialize_storm(self, state: StormState):
        """
        Workflow step: Initialize the storm (the research and writing process).

        This step:
        - Generates an initial outline for the topic.
        - Surveys subjects (editors/perspectives) to get a diverse set of viewpoints.

        Args:
            state (StormState): Current state dictionary.

        Returns:
            dict: Keys "outline" and "editors" to update the state with the generated outline and editor perspectives.
        """
        topic = state["topic"]
        results = await asyncio.gather(*[self._outline(topic), self._survey_subjects(topic)])

        # The results can come in any order due to gather
        # One is an Outline, the other is Perspectives, both contain editors.
        r1, r2 = results
        if isinstance(r1, Outline):
            outline = r1
            editors = r2.editors
        else:
            outline = r2
            editors = r2.editors

        return {"outline": outline, "editors": editors}

    async def conduct_interviews(self, state: StormState):
        """
        Workflow step: Conduct interviews with editors.

        Each editor represents a different perspective or domain of expertise. The interviews aim to gather
        additional context, references, and viewpoints that will shape the article.

        Args:
            state (StormState): Current state dictionary, expected to contain "topic" and "editors".

        Returns:
            dict: A dictionary with "interview_results", a list of interview states/results from each editor.
        """
        engine = VectorSearchRetrievalEngine(embedding=self.embedding, use_cache=True)

        topic: Optional[str] = state.get("topic")
        editors: Optional[List[Editor]] = state.get("editors")

        async def run_interview(editor: Editor, topic: str):
            self.logger.debug(f"Starting Interview for Editor: {editor.name}")
            interview = InterviewGraph(llm=self.llm, search_engine=engine)
            return await interview.run(editor, topic)

        # Prepare arguments for each editor
        tasks_args = [(editor, topic) for editor in editors]

        thread_runner = ThreadRunner(
            max_concurrency=self.max_concurrent_interviews,
            timeout=300,  # Max 5 minutes
            logger=self.logger,
        )

        # Run all interviews concurrently with timeouts
        results = await thread_runner.run_all(run_interview, tasks_args)

        return {"interview_results": results}

    async def refine_outline(self, state: StormState):
        """
        Workflow step: Refine the article's outline based on insights gathered from interviews.

        The method sends the old outline and the content of the interviews to the OpenAILLM and requests a refined outline.

        Args:
            state (StormState): Current state dictionary, expected to contain "topic", "outline", and "interview_results".

        Returns:
            dict: Updated "outline" after refinement.
        """
        formatted_conversations = []
        for interview_result in state.get("interview_results", []):
            results = interview_result.get("result", None)
            if results is not None:
                formatted_conversations.append(self._format_conversation(results))

        convos = "\n\n".join(formatted_conversations)

        topic = state.get("topic")
        old_outline = state.get("outline").as_str
        response = await self.llm.structured_completion(
            response_format=Outline,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a Wikipedia writer. You have gathered information from experts and search engines. Now, refine the outline of the Wikipedia page.
                    Topic: {topic}

                    Old outline:

                    {old_outline}""",
                },
                {"role": "user", "content": f"Refine the outline based on these editor interviews:\n\n{convos}"},
            ],
        )

        return {"outline": response.parsed}

    async def index_references(self, state: StormState):
        """
        Workflow step: Index references found during interviews into the vector database.

        Args:
            state (StormState): Current state dictionary, expected to contain "interview_results" which
                                may include references (chunks).

        Returns:
            dict: Empty, as no additional data is stored in state here.
        """
        interview_results = state.get("interview_results", [])

        chunks: list[Chunk] = []
        for ir in interview_results:
            if ir.get("result"):
                ir_state = ir.get("result")
                if ir_state is not None:
                    chunks.extend(ir_state.get("references", []))

        self.logger.debug(f"{len(chunks)} references extracted from interviews")
        await self.vector_db.aadd_chunks(list(chunks))

        return {}

    async def write_draft(self, state: StormState):
        """
        Workflow step: Write draft sections for the article.

        Using the refined outline, this step requests the OpenAILLM to write each section.
        Each section is returned along with references used.

        Args:
            state (StormState): Current state dictionary, expected to contain "topic" and "outline".

        Returns:
            dict: Dictionary with "sections" (list of WikiSection objects) and "final_refs" (list of all reference chunks).
        """
        outline = state["outline"]

        # Write each section in parallel
        tasks = await asyncio.gather(*[self._write_section(section=section.as_str, outline=outline.as_str) for section in outline.sections])

        sections, final_refs = zip(*tasks)

        sections = list(sections)
        final_refs = list(final_refs)

        # Combine all references from all sections
        references = set()
        for refs in final_refs:
            references.update(refs)

        return {"sections": sections, "final_refs": list(references)}

    async def write_article(self, state: StormState):
        """
        Workflow step: Write the final integrated article.

        This compiles all drafted sections into a final cohesive Wikipedia-style article.

        Args:
            state (StormState): Current state dictionary, expected to contain "topic" and "sections".

        Returns:
            dict: Contains the "article" as a single integrated text.
        """
        topic = state.get("topic")
        sections = state.get("sections")

        draft = "\n\n".join([section.as_str for section in sections])

        response = await self.llm.completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert Wikipedia author. Write the complete wiki article on the topic:
                    
                    "{topic}"
                    
                    Using the following section drafts:
                    
                    {draft}
                    
                    Follow Wikipedia format guidelines strictly.""",
                },
                {
                    "role": "user",
                    "content": 'Write the complete Wiki article using markdown. Use footnotes like "[1]" for citations, and include URLs in a references section without duplicates.',
                },
            ],
            frequency_penalty=0.4,
            presence_penalty=0.2,
        )

        return {"article": response.content}

    async def _outline(self, topic: str):
        """
        Private method: Generate an initial outline for the given topic using the OpenAILLM.

        Args:
            topic (str): The main topic of the article.

        Returns:
            Outline: The generated outline structure.
        """
        response = await self.llm.structured_completion(
            response_format=Outline,
            messages=[
                {"role": "system", "content": "You are a Wikipedia writer. Write a comprehensive and specific outline for a Wikipedia page on the topic."},
                {"role": "user", "content": topic},
            ],
        )
        return response.parsed

    async def _survey_subjects(self, topic: str):
        """
        Private method: Expand the topic by finding related subjects, then use these related subjects
        and their Wikipedia pages as inspiration for determining a set of diverse editors (perspectives).

        Args:
            topic (str): The main topic of the article.

        Returns:
            Perspectives: An object containing a list of editor perspectives.
        """
        related = await self._expand(topic)
        if related is None:
            raise ValueError("No related subjects found for the given topic.")

        examples = await self._wiki_examples(related.subjects)
        perspectives = await self._generate_perspectives(topic, examples)

        return perspectives

    async def _wiki_examples(self, related_topics: list[str]):
        """
        Private method: Search Wikipedia for each related topic and compile a formatted summary.

        Args:
            related_topics (list[str]): List of related topics.

        Returns:
            list[str]: Summaries of related Wikipedia pages, useful for inspiring editor perspectives.
        """

        def format_result(result: WikiPage, max_length: int = 1000):
            related_cats = "- " + "\n - ".join(result.categories)
            # Truncate to max_length as a safeguard
            return f"### {result.title}\n\nSummary: {result.summary}\n\nRelated\n{related_cats}"[:max_length]

        tasks = [self.wiki.search(rt, num_results=1) for rt in related_topics]
        results = await asyncio.gather(*tasks)
        wiki_results: list[WikiPage] = []
        for result in results:
            wiki_results.extend(result)

        return [format_result(w) for w in wiki_results]

    async def _generate_perspectives(self, topic: str, examples: list[str]):
        """
        Private method: Generate a set of editor perspectives based on related topic examples.

        Args:
            topic (str): The main topic of the article.
            examples (list[str]): Formatted summaries of related Wikipedia pages.

        Returns:
            Perspectives: Object containing a list of editors with distinct perspectives.
        """
        response = await self.llm.structured_completion(
            response_format=Perspectives,
            messages=[
                {
                    "role": "system",
                    "content": f"""Select a set of distinct Wikipedia editors to cover the topic from multiple angles. Each editor should focus on a unique aspect.
                    
                    Use the following wiki page outlines of related topics for inspiration:
                    
                    {examples}""",
                },
                {"role": "user", "content": topic},
            ],
        )
        return response.parsed

    async def _expand(self, topic: str):
        """
        Private method: Identify related subjects to the given topic using the OpenAILLM.

        Args:
            topic (str): The main topic.

        Returns:
            RelatedSubjects: Contains a list of related subjects and their associated URLs.
        """
        response = await self.llm.structured_completion(
            response_format=RelatedSubjects,
            messages=[
                {
                    "role": "user",
                    "content": f"""Identify some related Wikipedia subjects and URLs that provide insights into commonly associated aspects of the topic:
                    
                    Topic: {topic}""",
                }
            ],
        )
        return response.parsed

    def _format_conversation(self, interview_state: State):
        """
        Private method: Format a conversation log from an interview state.

        Args:
            interview_state (State): The state from an interview node, containing messages and editor info.

        Returns:
            str: A nicely formatted text representation of the entire conversation.
        """
        messages = interview_state["messages"]
        convo = "\n".join(f"{m.get('name')}: {m.get('content')}" for m in messages)
        return f"Conversation with {interview_state['editor'].name}\n\n" + convo

    async def _write_section(self, section: str, outline: str):
        """
        Private method: Write a single WikiSection draft using the OpenAILLM.

        Given a section name and the full outline, the method retrieves relevant chunks from the vector DB
        and provides them as references to the OpenAILLM. The OpenAILLM then produces a structured WikiSection output.

        Args:
            topic (str): The main topic.
            section (str): The specific section title or outline portion to draft.
            outline (str): The full outline string for context.

        Returns:
            tuple[WikiSection, list[Chunk]]: The drafted WikiSection and the reference chunks used.
        """
        query = f"{section}"
        chunks = await self.vector_db.asearch(query)

        docs = "\n".join([f'<Document href="{(doc.metadata or {}).get("url", "")}"/>\n{doc.content}\n</Document>' for doc in chunks])

        response = await self.llm.structured_completion(
            response_format=WikiSection,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert Wikipedia writer. Complete the following WikiSection from the outline:
                    
                    Outline:
                    {outline}
                    
                    Cite your sources using the references below:
                    
                    <Documents>
                    {docs}
                    </Documents>
                    """,
                }
            ],
        )

        return response.parsed, chunks

    async def _reset(self):
        """
        Reset the agent's internal state, graph, and OpenAILLM. Clears stored data so a new run can occur.

        NOTE: The vector database clearing is TODO and might not be implemented here.

        This is intended for debugging or starting a fresh process with the same agent instance.
        """
        self.graph.reset()
        self.llm = OpenAILLM()
        self.initialized = False
        self.vector_db.clear()
