import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, TypeVar

ReturnType = TypeVar("ReturnType")

_get_running_loop = asyncio.get_running_loop
_run = asyncio.run


def run_sync(coro: Awaitable[ReturnType]) -> ReturnType:
    """Run an awaitable synchronously, using a worker loop if a loop is active.

    The awaitable must not depend on work or futures bound to the caller's
    running loop, which remains blocked until this function returns.
    """

    async def await_result() -> ReturnType:
        return await coro

    try:
        loop = _get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run, await_result()).result()
    else:
        # Not inside an event loop → safe to call asyncio.run
        return _run(await_result())
