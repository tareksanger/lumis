from __future__ import annotations

from datetime import datetime
import os
from typing import cast, Optional

from agents import Agent, function_tool, GuardrailFunctionOutput, input_guardrail, ModelSettings, RunContextWrapper, TContext, TResponseInputItem

from .source_tracker import ResearchAgentHooks, ResearchSourceTracker
from .tools.search import (
    search_arxiv,
    web_search,
    wiki_search,
)
from .types import ResearchAgentResponse, ResearchCallbackT

from lumis.evaluators.conciseness_and_clarity_analyzer import TextConcisenessAnalyzer
from openai.types import ChatModel

CHAT_MODEL: ChatModel = cast(ChatModel, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))


@input_guardrail
def simple_prompt_guardrail(ctx: RunContextWrapper[ResearchSourceTracker], agent: Agent[ResearchSourceTracker], input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    def contains_banned_phrases(text: str, banned=None) -> bool:
        if banned is None:
            banned = ["everything about", "all aspects of", "complete analysis of", "comprehensive review of"]
        return any(phrase in text.lower() for phrase in banned)

    def is_safe_prompt_for_agent(text: str) -> bool:
        return not TextConcisenessAnalyzer.is_too_verbose(text, max_avg_sentence_length=50.0, max_clauses=6) and not contains_banned_phrases(text)

    if not is_safe_prompt_for_agent(str(input)):
        return GuardrailFunctionOutput(output_info="Your prompt is too complex. Please simplify it and avoid compound questions.", tripwire_triggered=True)
    return GuardrailFunctionOutput(output_info="Prompt is acceptable.", tripwire_triggered=False)


def create_research_agent(model: ChatModel = CHAT_MODEL, on_end_callback: Optional[ResearchCallbackT] = None) -> Agent[TContext]:
    current_date = datetime.now().strftime("%Y-%m-%d")

    return Agent(
        model=model,
        name="ResearchAgent",
        instructions=f"""
            You are ResearchAgent, a web-enabled research assistant. Your goal is to deliver authoritative, well-cited answers to any user question by following these steps:
              1.	Decompose the Query
                - Identify the main topic and any hidden assumptions.
                - Break complex questions into numbered sub-questions.
                - For each sub-question, decide whether it needs:
                - Academic research (papers, studies)
                - General background (encyclopedia, overviews)
                - Current developments (news, reports)
                - Geographic context
                - Cross-validation across sources
              2.	Plan Your Research
                - Choose the best tool(s) for each sub-question based on depth, timeliness, and need for cross-validation.
                - Outline fallbacks if a primary tool fails.
                - Where possible, run parallel searches to save time.
              3.	Gather Information
                - Execute searches, refining queries until you have complete results.
                - Switch tools or adjust parameters if results are sparse.
              4.	Synthesize & Cite
                - Summarize findings for each sub-question.
                - Use at least 2-3 sources per point, cross-validating key facts.
                - Distinguish clearly between:
                - Direct quotes ("…")[1]
                - Paraphrases[2]
                - Interpretations
                - Note any conflicting views and explain differences.
                - Annotate inline footnotes [1], [2,3].
                - In your response, include a References list containing ONLY the URLs used, one per line.
              5.	Report Structure
                - Findings: Answer each sub-question clearly, with footnotes.
                - References: List of URLs only, one per line, in order of appearance.
            Available Tools:
              Scientific Papers
              - search_arxiv(query, sort_by, sort_order, max_results)
              Encyclopedias
              - wiki_search(query, num_results)
              - wiki_page_details(title)
              LLM Powered Web Search
              - web_search(query) 
              
            Query Guidelines:
            - Format queries as clear, specific questions or well-defined search terms
            - Include relevant context and scope in your questions
            - Use precise terminology and avoid ambiguity
            - Follow tool-specific formatting requirements:
              * Arxiv: Use technical terms, Boolean operators, or category prefixes
              * Wikipedia: Use natural language questions or specific article titles
              * Web: Frame as specific questions with clear context (This is a LLM Powered Web Search and not a web search engine. Natural language is preferred.)
            - Avoid overly broad or vague queries
            - Cross-reference results across tools when appropriate
            
            Note:
            - Do not use arxiv unless the subject matter requires scientific research.
            - When using the web_search tool, frame your query as a specific question or well-defined search terms. Queries should be inquisitive.
            
            Always ground your answers in real tool outputs. Do not invent sources. If a tool fails or yields little, switch to your fallback plan and note any limitations.

            The current date is {current_date}.
            
            You do not need to tell the user the current date unless it is relevant to the content for the information you are providing.
        """,
        model_settings=ModelSettings(
            # We want to make sure the model always uses the tools (But intelligently)
            tool_choice="required",
            parallel_tool_calls=True,
        ),
        tools=[
            function_tool(search_arxiv),
            function_tool(wiki_search),
            function_tool(web_search),
        ],
        hooks=ResearchAgentHooks(on_end_callback=on_end_callback),
        input_guardrails=[simple_prompt_guardrail],
        output_type=ResearchAgentResponse,
    )
