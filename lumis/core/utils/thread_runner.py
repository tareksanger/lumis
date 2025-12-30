from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from inspect import iscoroutinefunction
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

R = TypeVar("R")


class TaskSuccess(TypedDict, Generic[R]):
    args: Tuple[Any, ...]
    result: R


class TaskError(TypedDict):
    args: Tuple[Any, ...]
    error: str


RunResult = Union[TaskSuccess[R], TaskError]


class ThreadRunner(Generic[R]):
    def __init__(
        self,
        max_concurrency: int,
        timeout: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        # Create one executor for all tasks.
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrency)

    async def run_all(  # noqa: C901
        self,
        func: Union[Callable[..., R], Callable[..., Awaitable[R]]],
        tasks_args: list[Tuple[Any, ...]],
    ) -> list[RunResult[R]]:
        """
        Schedules up to max_concurrency tasks in parallel.
        Each element in tasks_args is passed as *args into func.

        :param func: A function or coroutine that returns an R.
        :param tasks_args: A list of argument-tuples for func.
        :return: A list of RunResult[R] containing either the successful
                 result or an error string.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run_single_task(args: Tuple[Any, ...]) -> RunResult[R]:  # noqa: C901
            async with semaphore:
                try:
                    # If 'func' is an async function (or __call__ is async),
                    # call it directly. Otherwise, run it in a thread.
                    if iscoroutinefunction(func) or iscoroutinefunction(getattr(func, "__call__", None)):
                        if self.timeout is not None:
                            result = await asyncio.wait_for(func(*args), timeout=self.timeout)
                        else:
                            result = await func(*args)  # type: ignore
                    else:
                        loop = asyncio.get_running_loop()
                        if self.timeout is not None:
                            result = await asyncio.wait_for(
                                loop.run_in_executor(self.executor, func, *args),
                                timeout=self.timeout,
                            )
                        else:
                            result = await loop.run_in_executor(self.executor, func, *args)

                    return {
                        "args": args,
                        "result": result,
                    }
                except asyncio.TimeoutError:
                    self.logger.error(f"Task with args {args} timed out after {self.timeout} seconds.")
                    return {"args": args, "error": "Task timed out."}
                except Exception as e:
                    self.logger.error(f"Task with args {args} failed: {e}")
                    return {"args": args, "error": str(e)}

        tasks = [_run_single_task(args) for args in tasks_args]
        return await asyncio.gather(*tasks)
