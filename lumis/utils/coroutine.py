import asyncio
from typing import Awaitable, TypeVar

ReturnType = TypeVar("ReturnType")

_get_running_loop = asyncio.get_running_loop
_run_coroutine_threadsafe = asyncio.run_coroutine_threadsafe
_run = asyncio.run


def run_sync(coro: Awaitable[ReturnType]) -> ReturnType:
    try:
        loop = _get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an existing async loop → use thread executor
        return _run_coroutine_threadsafe(coro, loop).result()  # type: ignore
    else:
        # Not inside an event loop → safe to call asyncio.run
        return _run(coro)  # type: ignore
