from __future__ import annotations

from inspect import iscoroutinefunction
import logging
from typing import Any, Callable, Coroutine, Generic, TypeVar, Union

from lumis.core.common.logger_mixin import LoggerMixin

E = TypeVar("E", bound=str | None)
EventHandler = Union[Callable[..., Any], Callable[..., Coroutine[Any, Any, Any]]]


class EventEmitter(LoggerMixin, Generic[E]):
    def __init__(self) -> None:
        self._event_handlers: dict[E, list[EventHandler]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def on(self, event_name: E, handler: EventHandler) -> None:
        """
        Registers a handler for a specific event.

        Args:
            event_name (str): The name of the event.
            handler (EventHandler): The function to call when the event is emitted.
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        self.logger.debug(f"Registered handler '{handler.__name__}' for event '{event_name}'")

    async def emit(self, event_name: E, *args: Any, **kwargs: Any) -> None:
        """
        Emits an event and calls all registered handlers.

        Args:
            event_name (str): The name of the event to emit.
            *args (Any): Variable length argument list for handlers.
            **kwargs (Any): Arbitrary keyword arguments for handlers.
        """
        handlers: list[EventHandler] = self._event_handlers.get(event_name, [])
        self.logger.debug(f"Emitting event '{event_name}' to {len(handlers)} handler(s)")
        for handler in handlers:
            try:
                if iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
                self.logger.debug(f"Handler '{handler.__name__}' executed successfully for event '{event_name}'")
            except Exception as e:
                self.logger.error(f"Error in handler '{handler.__name__}' for event '{event_name}': {e}")
                self.logger.exception(e)
