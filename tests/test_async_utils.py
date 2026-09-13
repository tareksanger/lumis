"""Concurrency tests use event coordination and bounded deadlock detection."""

import asyncio
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading

from lumis.core.utils.coroutine import run_sync
from lumis.core.utils.thread_runner import ThreadRunner

import pytest


@pytest.fixture
def runner():
    instance = ThreadRunner(max_concurrency=2)
    yield instance
    instance.executor.shutdown(wait=True, cancel_futures=True)


def test_run_sync_returns_result_without_active_loop():
    async def work():
        await asyncio.sleep(0)
        return 42

    assert run_sync(work()) == 42


def test_run_sync_accepts_custom_awaitable():
    class Value:
        def __await__(self):
            async def value():
                return "result"

            return value().__await__()

    assert run_sync(Value()) == "result"


def test_run_sync_propagates_exception_without_active_loop():
    async def work():
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        run_sync(work())


def test_run_sync_inside_active_loop_completes_and_propagates_errors():
    # A separate process bounds the regression: the previous implementation
    # blocked its own event loop forever, defeating in-loop timeout guards.
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # Explicit import path also supports interpreters using safe-path mode.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(project_root), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import threading
from lumis.core.utils.coroutine import run_sync

async def main():
    caller_thread = threading.get_ident()
    original_threads = set(threading.enumerate())
    async def work():
        await asyncio.sleep(0)
        assert threading.get_ident() != caller_thread
        return 42
    assert run_sync(work()) == 42
    async def fail():
        raise ValueError("expected")
    try:
        run_sync(fail())
    except ValueError as error:
        assert str(error) == "expected"
    else:
        raise AssertionError("exception was lost")
    class Value:
        def __await__(self):
            return work().__await__()
    assert run_sync(Value()) == 42
    assert set(threading.enumerate()) == original_threads, "worker thread leaked"
    await asyncio.sleep(0)
asyncio.run(main())
""",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


async def test_empty_task_list_does_not_call_function(runner):
    def unexpected():
        pytest.fail("empty task list invoked callable")

    assert await runner.run_all(unexpected, []) == []


async def test_sync_callable_runs_in_worker_and_receives_positional_args(runner):
    caller_thread = threading.get_ident()

    def add(left, right):
        assert threading.get_ident() != caller_thread
        return left + right

    assert await runner.run_all(add, [(1, 2), (3, 4)]) == [{"args": (1, 2), "result": 3}, {"args": (3, 4), "result": 7}]


async def test_async_callable_object_is_awaited(runner):
    class Multiply:
        async def __call__(self, value):
            await asyncio.sleep(0)
            return value * 2

    assert await runner.run_all(Multiply(), [(3,)]) == [{"args": (3,), "result": 6}]


@pytest.mark.parametrize("async_callable", [False, True])
async def test_successful_task_with_timeout_returns_result(runner, async_callable):
    runner.timeout = 2

    def sync_work():
        return None

    async def async_work():
        return None

    assert await runner.run_all(async_work if async_callable else sync_work, [()]) == [{"args": (), "result": None}]


async def test_async_tasks_respect_concurrency_and_preserve_input_order(runner):
    started = asyncio.Event()
    release_first = asyncio.Event()
    active = 0
    peak = 0
    completed = []

    async def work(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if value == 0:
                started.set()
                await release_first.wait()
            else:
                await started.wait()
                await asyncio.sleep(0)
            completed.append(value)
            if value == 3:
                release_first.set()
            return value * 10
        finally:
            active -= 1

    result = await asyncio.wait_for(runner.run_all(work, [(n,) for n in range(4)]), 2)
    assert peak == 2
    assert active == 0
    assert completed[-1] == 0
    assert result == [{"args": (n,), "result": n * 10} for n in range(4)]


async def test_sync_tasks_respect_worker_limit(runner):
    lock = threading.Lock()
    release = threading.Event()
    both_started = asyncio.Event()
    loop = asyncio.get_running_loop()
    active = 0
    peak = 0

    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                loop.call_soon_threadsafe(both_started.set)
        try:
            assert release.wait(3), "test did not release workers"
            return value
        finally:
            with lock:
                active -= 1

    pending = asyncio.create_task(runner.run_all(work, [(n,) for n in range(6)]))
    try:
        await asyncio.wait_for(both_started.wait(), 2)
        assert peak == 2
    finally:
        release.set()
        result = await asyncio.wait_for(pending, 3)
    assert peak == 2
    assert active == 0
    assert result == [{"args": (n,), "result": n} for n in range(6)]


@pytest.mark.parametrize("async_callable", [False, True])
async def test_callable_errors_are_isolated_from_successes(runner, caplog, async_callable):
    def sync_work(value):
        if value == "bad":
            raise ValueError("invalid task")
        return value.upper()

    async def async_work(value):
        return sync_work(value)

    with caplog.at_level(logging.ERROR):
        result = await runner.run_all(async_work if async_callable else sync_work, [("ok",), ("bad",)])
    assert result == [{"args": ("ok",), "result": "OK"}, {"args": ("bad",), "error": "invalid task"}]
    assert "invalid task" in caplog.text


async def test_async_timeout_cancels_task_and_reports_error(runner, caplog):
    runner.timeout = 0.02
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    result = await asyncio.wait_for(runner.run_all(blocked, [()]), 2)
    assert started.is_set()
    assert cancelled.is_set()
    assert result == [{"args": (), "error": "Task timed out."}]
    assert "timed out" in caplog.text


async def test_sync_timeout_reports_error_while_worker_finishes_after_release(runner):
    runner.timeout = 0.1
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked():
        started.set()
        try:
            assert release.wait(3), "test did not release worker"
        finally:
            finished.set()

    try:
        result = await asyncio.wait_for(runner.run_all(blocked, [()]), 2)
        assert started.is_set()
        assert result == [{"args": (), "error": "Task timed out."}]
        assert not finished.is_set()
    finally:
        release.set()
    assert await asyncio.to_thread(finished.wait, 2)


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_concurrency_is_rejected(value):
    with pytest.raises(ValueError):
        ThreadRunner(value)
