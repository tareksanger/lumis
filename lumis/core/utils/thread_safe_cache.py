from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class ThreadSafeCache(Generic[T]):
    """
    A thread-safe cache with a configurable maximum size. If the cache exceeds `max_size`,
    the oldest inserted item is evicted.

    This ensures the cache doesn't grow indefinitely.
    """

    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize the ThreadSafeCache.

        Args:
            max_size (int): The maximum number of items to store in the cache. Defaults to 100.
        """
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, T]" = OrderedDict()
        self.max_size = max_size

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: str) -> T:
        with self._lock:
            return self._cache[key]

    def __setitem__(self, key: str, value: T) -> None:
        with self._lock:
            # Move to end or insert fresh
            self._cache[key] = value
            self._cache.move_to_end(key)
            self._evict_if_needed()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._cache[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __iter__(self):
        with self._lock:
            # Return a copy of keys to avoid runtime errors while iterating
            return iter(list(self._cache.keys()))

    def __repr__(self) -> str:
        with self._lock:
            return f"{self.__class__.__name__}({list(self._cache.items())!r})"

    def __bool__(self) -> bool:
        return len(self) > 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThreadSafeCache):
            return NotImplemented
        with self._lock, other._lock:
            return list(self._cache.items()) == list(other._cache.items())

    def get_content(self, key: str) -> Optional[T]:
        with self._lock:
            return self._cache.get(key)

    def set_content(self, key: str, content: T) -> None:
        with self._lock:
            self._cache[key] = content
            self._cache.move_to_end(key)
            self._evict_if_needed()

    def fetch_or_parse(self, key: str, parse_function: Callable[[str], T]) -> T:
        with self._lock:
            content = self._cache.get(key)
            if content is not None:
                return content
        # Parse outside lock
        new_content = parse_function(key)
        with self._lock:
            # Double-check after parse
            if key not in self._cache:
                self._cache[key] = new_content
                self._cache.move_to_end(key)
                self._evict_if_needed()
                return new_content
            else:
                return self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self):
        with self._lock:
            return list(self._cache.keys())

    def values(self):
        with self._lock:
            return list(self._cache.values())

    def items(self):
        with self._lock:
            return list(self._cache.items())

    def copy(self) -> "ThreadSafeCache[T]":
        new_cache = ThreadSafeCache[T](max_size=self.max_size)
        with self._lock:
            for k, v in self._cache.items():
                new_cache._cache[k] = v
        return new_cache

    def _evict_if_needed(self):
        """
        Evict the oldest item if the cache size exceeds `max_size`.
        """
        while len(self._cache) > self.max_size:
            # FIFO eviction
            oldest_key, _ = self._cache.popitem(last=False)
            # Could add a log here if desired: logging.debug(f"Evicted oldest item from cache: {oldest_key}")
