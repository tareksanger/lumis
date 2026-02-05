from __future__ import annotations

import pytest

from lumis.core.event_emitter import EventEmitter


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestEventEmitterRegistration:
    def test_on_registers_single_handler(self):
        emitter = EventEmitter()
        handler = lambda: None  # noqa: E731
        emitter.on("click", handler)

        assert emitter._event_handlers["click"] == [handler]

    def test_on_registers_multiple_handlers_for_same_event(self):
        emitter = EventEmitter()
        h1 = lambda: None  # noqa: E731
        h2 = lambda: None  # noqa: E731
        emitter.on("click", h1)
        emitter.on("click", h2)

        assert emitter._event_handlers["click"] == [h1, h2]

    def test_on_keeps_handlers_isolated_across_events(self):
        emitter = EventEmitter()
        h_click = lambda: None  # noqa: E731
        h_hover = lambda: None  # noqa: E731
        emitter.on("click", h_click)
        emitter.on("hover", h_hover)

        assert emitter._event_handlers["click"] == [h_click]
        assert emitter._event_handlers["hover"] == [h_hover]


# ---------------------------------------------------------------------------
# Sync handlers
# ---------------------------------------------------------------------------


class TestEventEmitterSyncEmit:
    async def test_sync_handler_is_called(self):
        emitter = EventEmitter()
        calls: list[str] = []
        emitter.on("ping", lambda: calls.append("pong"))

        await emitter.emit("ping")

        assert calls == ["pong"]

    async def test_sync_handler_receives_args_and_kwargs(self):
        emitter = EventEmitter()
        received: list = []

        def capture(*args, **kwargs):
            received.append((args, kwargs))

        emitter.on("data", capture)
        await emitter.emit("data", 1, "two", key="value")

        assert received == [((1, "two"), {"key": "value"})]


# ---------------------------------------------------------------------------
# Async handlers
# ---------------------------------------------------------------------------


class TestEventEmitterAsyncEmit:
    async def test_async_handler_is_called(self):
        emitter = EventEmitter()
        calls: list[str] = []

        async def handler():
            calls.append("async_pong")

        emitter.on("ping", handler)
        await emitter.emit("ping")

        assert calls == ["async_pong"]

    async def test_async_handler_receives_args_and_kwargs(self):
        emitter = EventEmitter()
        received: list = []

        async def capture(*args, **kwargs):
            received.append((args, kwargs))

        emitter.on("data", capture)
        await emitter.emit("data", 42, flag=True)

        assert received == [((42,), {"flag": True})]


# ---------------------------------------------------------------------------
# Mixed sync + async handlers
# ---------------------------------------------------------------------------


class TestEventEmitterMixedHandlers:
    async def test_sync_and_async_handlers_both_run_in_registration_order(self):
        emitter = EventEmitter()
        order: list[str] = []

        def sync_first():
            order.append("sync")

        async def async_second():
            order.append("async")

        emitter.on("mixed", sync_first)
        emitter.on("mixed", async_second)

        await emitter.emit("mixed")

        assert order == ["sync", "async"]


# ---------------------------------------------------------------------------
# Emit with no handlers
# ---------------------------------------------------------------------------


class TestEventEmitterNoHandlers:
    async def test_emit_unregistered_event_does_not_raise(self):
        emitter = EventEmitter()
        # Should complete without error
        await emitter.emit("ghost")


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestEventEmitterErrorIsolation:
    async def test_failing_handler_does_not_prevent_subsequent_handlers(self):
        emitter = EventEmitter()
        calls: list[str] = []

        def bad_handler():
            raise RuntimeError("boom")

        def good_handler():
            calls.append("survived")

        emitter.on("risky", bad_handler)
        emitter.on("risky", good_handler)

        # emit should not propagate the exception
        await emitter.emit("risky")

        assert calls == ["survived"]

    async def test_failing_async_handler_does_not_prevent_subsequent_handlers(self):
        emitter = EventEmitter()
        calls: list[str] = []

        async def bad_async():
            raise ValueError("async boom")

        async def good_async():
            calls.append("async survived")

        emitter.on("risky", bad_async)
        emitter.on("risky", good_async)

        await emitter.emit("risky")

        assert calls == ["async survived"]

    async def test_failing_handler_error_is_logged(self, caplog):
        emitter = EventEmitter()

        def bad_handler():
            raise RuntimeError("logged error")

        emitter.on("err", bad_handler)

        import logging

        with caplog.at_level(logging.ERROR):
            await emitter.emit("err")

        assert "logged error" in caplog.text


# ---------------------------------------------------------------------------
# Multiple events / handlers end-to-end
# ---------------------------------------------------------------------------


class TestEventEmitterEndToEnd:
    async def test_each_event_only_triggers_its_own_handlers(self):
        emitter = EventEmitter()
        a_calls: list[int] = []
        b_calls: list[int] = []

        emitter.on("a", lambda: a_calls.append(1))
        emitter.on("b", lambda: b_calls.append(1))

        await emitter.emit("a")
        await emitter.emit("a")
        await emitter.emit("b")

        assert a_calls == [1, 1]
        assert b_calls == [1]

    async def test_handler_registered_after_emit_is_not_called_retroactively(self):
        emitter = EventEmitter()
        calls: list[str] = []

        await emitter.emit("late")  # no handlers yet

        emitter.on("late", lambda: calls.append("late"))

        # only the second emit should trigger it
        await emitter.emit("late")

        assert calls == ["late"]
