from __future__ import annotations

import logging
from typing import cast, Literal, Optional, TypedDict

from lumis.agents.base import GraphBasedAgent
from lumis.evaluators.conciseness_and_clarity_analyzer import TextConcisenessAnalyzer
from lumis.llm import OpenAILLM
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

ModLevel = Literal["NO MOD", "SOME MOD", "HEAVY MOD"]
Level = Literal["LOW", "MID", "HIGH"]
YesNo = Literal["YES", "NO"]


class Assumption(BaseModel):
    assumption: str = Field(description="The assumption about the user's goals that need to be made")
    salience: Level = Field(description="The salience of the assumption")
    plausibility: Level = Field(description="The plausibility of the assumption")


class Rewrite(BaseModel):
    rewrite: str = Field(description="The Rewritten Query. Make sure to include ALL relevant information from the original Query and the Conversational History")
    information_added: Optional[YesNo] = Field(description="Whether information beyond what's present in the Query or the Conversational History needs to be added in the rewrite. Reply YES or NO")
    assumptions: Optional[list[Assumption]] = Field(
        description=(
            "If there's additional information needed to be added to the user's query for it to be effective (information_added is YES), "
            "then those are assumptions about the user's goals that need to be made."
        )
    )


class RewriteResponse(BaseModel):
    mod_level: ModLevel = Field(description="The level of modification needed to the query. Can be NO MOD, SOME MOD or HEAVY MOD")
    reason: str = Field(description="The reason for the modification level")
    rewrites: Optional[list[Rewrite]] = Field(
        description="The list of rewrites that can be made to the query. Make sure to include ALL relevant information from the original Query and the Conversational History"
    )


REWRITING_PROMPT = """
Refine a user's query by analyzing it and the provided conversational history to identify aspects of improvement or to highlight its effectiveness.

Analyze the query to determine if any modifications are required. Classify the necessity of modification as NO MOD, SOME MOD, or HEAVY MOD. If classified as NO MOD, present effective aspects of the query in a table form. If modifications are needed, propose rewritten queries that communicate the user's needs and goals more effectively, without altering the intended purpose. If new information is needed for enhancement, list the assumptions made to include this information.

Evaluation Procedure

1. Determine Modification Necessity
   - Assess the query's effectiveness based on its clarity, completeness, and alignment with the user's intent.
   - Classify the outcome as:
     - NO MOD: The query is effective as is.
     - SOME MOD: The query needs slight improvements.
     - HEAVY MOD: The query requires substantial rewrites.

2. Effective Aspects Table
   - If classified as NO MOD, present a table highlighting effective aspects:
     - Aspect: Clarity - Describe how the query clearly conveys the question.
     - Aspect: Completeness - Indicate if the query covers all necessary details.
     - Aspect: Relevance - Confirm alignment with user's goals.

3. Propose Rewrites
   - If modifications are necessary:
     - Present a list of potential rewrites ordered from the most to least effective, maintaining all relevant original information.
     - Specify if any new information or assumptions have been made for clarity.

Output Format

- Classification (NO MOD, SOME MOD, HEAVY MOD)
- Table of Effective Aspects (if NO MOD)
- List of Rewrites (for SOME MOD or HEAVY MOD):
  - Rewrite: <Rewritten query>
  - Information Added: <YES/NO>
  - Assumptions: <Details of any assumptions made>

Notes

- Context may vary. Consider conversational history when relevant to ensure alignment with user's goals, but disregard unrelated context.
- Focus on preserving the user's original intent in each rewrite.
- Ensure rewrites enhance clarity and comprehensiveness without adding unnecessary complexity.

Conversational History: {conversation_history}
Query: {query}
"""  # noqa: E501


class PromptRefinementState(TypedDict):
    query: str
    conversation_history: Optional[list[ChatCompletionMessageParam]]

    # The full response from the OpenAILLM with all potential rewrites
    rewrite_response: Optional[RewriteResponse]

    # The best rewrite from the OpenAILLM response
    rewrite: Optional[Rewrite]


class PromptRefinementPipeline(GraphBasedAgent[PromptRefinementState, None]):
    def __init__(
        self,
        llm: Optional[OpenAILLM] = None,
        logger: Optional[logging.Logger] = None,  # 5 minutes per interview
        verbose: bool = False,
    ) -> None:
        super().__init__(llm, logger, verbose)

    def construct_graph(self):
        self.graph.add_node("check_for_rewrite", self.check_for_rewrite, "start")
        self.graph.add_node("extract_best_rewrite", self.extract_best_rewrite)

        # If the OpenAILLM response is not None, extract the best rewrite
        self.graph.add_edge(
            "check_for_rewrite",
            "extract_best_rewrite",
            condition=lambda state: state.get("rewrite_response") is not None,
        )

    async def setup(
        self,
        query: str,
        conversation_history: Optional[list[ChatCompletionMessageParam]] = None,
    ):
        self.graph.set_initial_state(
            {
                "query": query,
                "conversation_history": conversation_history,
                # The response from the OpenAILLM
                "rewrite_response": None,
                # The best rewrite from the OpenAILLM response
                "rewrite": None,
            }
        )

    async def check_for_rewrite(self, state: PromptRefinementState):
        query = state.get("query")
        conversation_history = state.get("conversation_history")

        response = await self.llm.structured_completion(
            response_format=RewriteResponse,
            model="gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": REWRITING_PROMPT.format(
                        conversation_history=conversation_history,
                        query=query,
                    ),
                },
            ],
        )

        # Only accept the response for rewrites if the mod_level is not NO MOD
        if content := response.parsed:
            if content.mod_level != "NO MOD":
                return {"rewrite_response": content}

        return {"rewrite_response": None}

    async def extract_best_rewrite(self, state: PromptRefinementState):
        # Enforce the typing because the condition in the edge above ensures that the response is not None
        rewrite_response: RewriteResponse = cast(RewriteResponse, state.get("rewrite_response"))
        # query = state.get("query")

        # Calculate scores for all rewrites and store them
        scored_rewrites = [(TextConcisenessAnalyzer.composite_readability_score(r.rewrite), r) for r in rewrite_response.rewrites or []]

        self.logger.debug(f"Scored rewrites: {scored_rewrites}")
        # Find the best rewrite and its score
        if not scored_rewrites:
            return {"rewrite": None}

        best_rewrite_score, best_rewrite = max(scored_rewrites, key=lambda x: x[0])
        self.logger.debug(f"Best rewrite score: {best_rewrite_score}")
        self.logger.debug(f"Best rewrite: {best_rewrite}")

        return {"rewrite": best_rewrite}
