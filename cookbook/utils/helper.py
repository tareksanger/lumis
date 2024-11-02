#!/usr/bin/env python3

def setup_notebook():
    """
    Sets up the environment for the notebook by applying nest_asyncio,
    setting environment variables, and initializing the Llama Index.

    This function:
    1. Applies `nest_asyncio` to allow nested event loops.
    2. Sets environment variables for OpenAI API key, Redis URL, and Tavily API key.
    3. Calls `setup_llama_index` to configure the Llama Index.

    """
    import os

    from .config import config

    import nest_asyncio

    print("Applying Nest Asyncio to allow nested event loops in Jupyter.")
    nest_asyncio.apply()

    env_vars = {
        "OPENAI_API_KEY": config.openai_api_key,
    }

    print("Setting Environment Variables")
    for key, value in env_vars.items():
        print(f"- {key}")
        os.environ[key] = value
