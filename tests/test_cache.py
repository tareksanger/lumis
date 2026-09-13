from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

from lumis.core.utils.thread_safe_cache import ThreadSafeCache

import pytest


def test_fifo_eviction_and_overwrite_order():
    cache = ThreadSafeCache(max_size=2)
    cache["a"], cache["b"] = 1, 2
    assert cache["a"] == 1
    cache["c"] = 3
    assert cache.items() == [("b", 2), ("c", 3)]
    cache["b"] = 4
    cache["d"] = 5
    assert cache.items() == [("b", 4), ("d", 5)]


def test_collection_interface_and_snapshot_iteration():
    cache = ThreadSafeCache()
    assert not cache
    assert 1 not in cache
    assert cache.get_content("absent") is None
    with pytest.raises(KeyError):
        _ = cache["absent"]
    cache.set_content("a", 1)
    iterator = iter(cache)
    cache["b"] = 2
    assert list(iterator) == ["a"]
    assert cache.values() == [1, 2]
    assert "a" in cache
    del cache["a"]
    assert cache.keys() == ["b"]
    cache.clear()
    assert len(cache) == 0


def test_copy_is_independent_and_equality_handles_self():
    cache = ThreadSafeCache(max_size=2)
    cache["a"] = 1
    copied = cache.copy()
    assert cache == cache
    assert cache == copied
    assert cache != {}
    copied["b"] = 2
    assert "b" not in cache
    assert copied.max_size == 2
    assert cache != copied


def test_fetch_caches_results_and_does_not_cache_errors():
    cache = ThreadSafeCache()
    parser = Mock(return_value=0)
    assert cache.fetch_or_parse("a", parser) == 0
    assert cache.fetch_or_parse("a", parser) == 0
    parser.assert_called_once_with("a")
    with pytest.raises(ValueError, match="broken"):
        cache.fetch_or_parse("b", Mock(side_effect=ValueError("broken")))
    assert "b" not in cache


def test_concurrent_fetch_returns_single_committed_value():
    cache = ThreadSafeCache()
    barrier = Barrier(2)

    def parse(key):
        value = object()
        barrier.wait(timeout=3)
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cache.fetch_or_parse, "key", parse) for _ in range(2)]
        values = [f.result(timeout=5) for f in futures]
    assert values[0] is values[1] is cache["key"]


def test_zero_capacity_never_retains_entries():
    cache = ThreadSafeCache(max_size=0)
    cache["a"] = 1
    assert not cache
    assert cache.fetch_or_parse("b", lambda _: 2) == 2
    assert not cache


def test_negative_capacity_is_rejected():
    with pytest.raises(ValueError, match="max_size"):
        ThreadSafeCache(max_size=-1)


def test_self_equality_completes_even_when_its_lock_is_held():
    from threading import Thread

    cache = ThreadSafeCache()
    outcomes = []
    # A daemon thread and bounded join keep a self-deadlock regression from
    # hanging the suite, even though the underlying lock is not reentrant.
    with cache._lock:
        worker = Thread(target=lambda: outcomes.append(cache == cache), daemon=True)
        worker.start()
        worker.join(timeout=3)
        assert not worker.is_alive(), "Self-comparison attempted to acquire its own lock"
    assert outcomes == [True]


def test_simultaneous_reversed_comparisons_complete_without_deadlock():
    from threading import Thread

    first, second = ThreadSafeCache(), ThreadSafeCache()
    first["item"] = second["item"] = 1
    start = Barrier(3)
    outcomes, errors = [], []

    def compare(left, right):
        try:
            start.wait(timeout=3)
            outcomes.append(all(left == right for _ in range(100)))
        except BaseException as error:
            errors.append(error)

    workers = [Thread(target=compare, args=pair, daemon=True) for pair in [(first, second), (second, first)]]
    for worker in workers:
        worker.start()
    start.wait(timeout=3)
    for worker in workers:
        worker.join(timeout=3)
    assert not any(worker.is_alive() for worker in workers), "Reversed comparisons deadlocked"
    assert errors == []
    assert outcomes == [True, True]


def test_equality_blocks_writers_to_both_caches_until_comparison_finishes():
    from threading import Event, Thread, current_thread

    entered_comparison, release_comparison = Event(), Event()
    outcomes, errors = [], []

    class GatedValue:
        def __eq__(self, other):
            entered_comparison.set()
            if not release_comparison.wait(timeout=5):
                raise TimeoutError("Comparison was not released")
            return isinstance(other, GatedValue)

    class ObservedLock:
        def __init__(self, lock):
            self.lock = lock
            self.writer_attempted = Event()
            self.writer_was_blocked = False

        def __enter__(self):
            if current_thread().name.startswith("cache-writer-"):
                acquired = self.lock.acquire(blocking=False)
                self.writer_was_blocked = not acquired
                self.writer_attempted.set()
                if not acquired:
                    self.lock.acquire()
            else:
                self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    first, second = ThreadSafeCache(), ThreadSafeCache()
    first["item"], second["item"] = GatedValue(), GatedValue()
    locks = [ObservedLock(cache._lock) for cache in (first, second)]
    first._lock, second._lock = locks
    completed = [Event(), Event()]

    def compare():
        try:
            outcomes.append(first == second)
        except BaseException as error:
            errors.append(error)

    def write(cache, done):
        try:
            cache["new"] = 2
            done.set()
        except BaseException as error:
            errors.append(error)

    comparison = Thread(target=compare, daemon=True)
    writers = [Thread(target=write, args=(cache, done), name=f"cache-writer-{index}", daemon=True) for index, (cache, done) in enumerate(zip((first, second), completed))]
    comparison.start()
    try:
        assert entered_comparison.wait(timeout=3), "Value comparison never started"
        for writer in writers:
            writer.start()
        for lock in locks:
            assert lock.writer_attempted.wait(timeout=3), "Writer never attempted its cache lock"
            assert lock.writer_was_blocked, "Equality released a cache lock before comparing values"
        assert not any(done.is_set() for done in completed)
    finally:
        release_comparison.set()
        comparison.join(timeout=3)
        for writer in writers:
            if writer.ident is not None:
                writer.join(timeout=3)
    assert not comparison.is_alive()
    assert not any(writer.is_alive() for writer in writers)
    assert errors == []
    assert outcomes == [True]
    assert all(done.is_set() for done in completed)
    assert first["new"] == second["new"] == 2
