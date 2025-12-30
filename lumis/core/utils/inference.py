from __future__ import annotations

import logging
import textwrap

from lumis.llm import OpenAILLM

from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


async def generate_response(llm: OpenAILLM, messages: list[ChatCompletionMessageParam], *args, **kwargs):
    response = await llm.completion(messages=messages, *args, **kwargs)  # Note: the completion already has tenacity retries built in

    if response is not None and response.content is not None:
        return response.content
    else:
        return None


async def generate_summary(llm: OpenAILLM, text: str):
    return await generate_response(
        llm,
        [
            {
                "role": "developer",
                "content": textwrap.dedent("""
                Can you provide a comprehensive summary of the given text? The summary should cover all the key points and main ideas presented in the original text, 
                while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples 
                that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity 
                of the original text, providing a clear and accurate overview without omitting any important information.
                """),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
