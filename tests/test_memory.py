"""Memory contracts, exercised without model clients or external services."""

from __future__ import annotations

import asyncio
import copy

from lumis.memory import BaseMemory, SimpleMemory

import pytest


def message(content: str):
    return {"role": "user", "content": content}


def test_base_memory_requires_a_concrete_implementation():
    with pytest.raises(TypeError, match="abstract"):
        BaseMemory()


async def test_new_instances_have_independent_empty_histories():
    first, second = SimpleMemory(), SimpleMemory()
    await first.add(message("first"))
    assert first.max_memory_size == 2000
    assert first.length == 1
    assert second.length == 0
    assert await second.get() == []


async def test_add_preserves_order_and_get_returns_a_separate_list():
    memory = SimpleMemory()
    messages = [message("first"), message("second")]
    for item in messages:
        await memory.add(item)
    result = await memory.get()
    assert result == messages
    result.clear()
    assert await memory.get() == messages
    assert memory.length == 2


@pytest.mark.parametrize("prefix", [message("first"), [message("first"), message("second")], []])
async def test_prepend_accepts_single_message_or_ordered_batch(prefix):
    tail = message("tail")
    memory = SimpleMemory(messages=[tail])
    await memory.prepend(prefix)
    expected = prefix if isinstance(prefix, list) else [prefix]
    assert await memory.get() == expected + [tail]
    assert memory.length == len(expected) + 1


@pytest.mark.parametrize("index", [0, 1, 2])
async def test_insert_supports_beginning_middle_and_end(index):
    original = [message("first"), message("last")]
    memory = SimpleMemory(messages=original.copy())
    added = message("inserted")
    await memory.insert(index, added)
    assert await memory.get() == original[:index] + [added] + original[index:]


async def test_insert_into_empty_memory():
    memory = SimpleMemory()
    added = message("first")
    await memory.insert(0, added)
    assert await memory.get() == [added]


@pytest.mark.parametrize("operation", ["insert", "remove", "update"])
@pytest.mark.parametrize("index", [-1, 3])
async def test_invalid_mutation_index_raises_without_changing_history(operation, index):
    original = [message("first"), message("last")]
    memory = SimpleMemory(messages=original.copy())
    args = (index,) if operation == "remove" else (index, message("replacement"))
    with pytest.raises(IndexError, match="Index out of range"):
        await getattr(memory, operation)(*args)
    assert await memory.get() == original
    # An exception inside the lock must not prevent subsequent writes.
    await memory.add(message("after failure"))
    assert memory.length == 3


@pytest.mark.parametrize("operation", ["remove", "update"])
async def test_remove_and_update_reject_index_equal_to_length(operation):
    memory = SimpleMemory(messages=[message("only")])
    args = (1,) if operation == "remove" else (1, message("replacement"))
    with pytest.raises(IndexError):
        await getattr(memory, operation)(*args)
    assert memory.length == 1


@pytest.mark.parametrize("index", [0, 1, 2])
async def test_remove_deletes_only_the_selected_message(index):
    original = [message(str(i)) for i in range(3)]
    memory = SimpleMemory(messages=original.copy())
    await memory.remove(index)
    assert await memory.get() == original[:index] + original[index + 1 :]
    assert memory.length == 2


@pytest.mark.parametrize("index", [0, 1, 2])
async def test_update_replaces_only_the_selected_message(index):
    original = [message(str(i)) for i in range(3)]
    memory = SimpleMemory(messages=original.copy())
    replacement = message("updated")
    await memory.update(index, replacement)
    assert await memory.get() == original[:index] + [replacement] + original[index + 1 :]
    assert memory.length == 3


async def test_clear_is_idempotent_and_allows_reuse():
    memory = SimpleMemory(messages=[message("old")])
    await memory.clear()
    await memory.clear()
    assert await memory.get() == []
    assert memory.length == 0
    await memory.add(message("new"))
    assert await memory.get() == [message("new")]


@pytest.mark.parametrize("count", [0, 3, 4])
async def test_get_returns_all_messages_at_or_below_capacity(count):
    history = [message(str(i)) for i in range(count)]
    assert await SimpleMemory(max_memory_size=4, messages=history).get() == history


async def test_over_capacity_get_retains_start_and_end_without_destroying_history():
    history = [message(str(i)) for i in range(8)]
    memory = SimpleMemory(max_memory_size=4, messages=history)
    assert await memory.get() == history[:2] + history[-2:]
    assert memory.length == 8
    memory.max_memory_size = 8
    assert await memory.get() == history


@pytest.mark.parametrize("capacity, expected", [(0, []), (1, [message("3")])])
async def test_small_capacity_returns_only_the_allowed_recent_context(capacity, expected):
    memory = SimpleMemory(max_memory_size=capacity, messages=[message(str(i)) for i in range(4)])
    assert await memory.get() == expected


async def test_concurrent_adds_do_not_lose_messages():
    memory = SimpleMemory()
    await asyncio.gather(*(memory.add(message(str(i))) for i in range(100)))
    contents = [item["content"] for item in await memory.get()]
    assert len(contents) == 100
    assert set(contents) == {str(i) for i in range(100)}


async def test_serialization_round_trip_preserves_configuration_and_history():
    history = [message("hello"), {"role": "assistant", "content": "world"}]
    memory = SimpleMemory(max_memory_size=12, messages=history)
    expected = {"max_memory_size": 12, "messages": history}
    assert memory.to_dict() == expected
    assert memory.dict() == expected
    assert memory.__dict__ == expected
    restored = SimpleMemory.from_dict(expected)
    assert restored.max_memory_size == 12
    assert await restored.get() == history
    await restored.add(message("continued"))
    assert restored.length == 3


async def test_deserialization_accepts_omitted_messages():
    memory = SimpleMemory.from_dict({"max_memory_size": 6})
    assert memory.max_memory_size == 6
    assert await memory.get() == []


def test_deepcopy_copies_nested_message_content_and_preserves_shared_references():
    shared = {"role": "user", "content": [{"type": "text", "text": "original"}]}
    original = SimpleMemory(max_memory_size=10, messages=[shared, shared])
    duplicate = copy.deepcopy(original)
    duplicate.messages[0]["content"][0]["text"] = "changed"
    assert original.messages[0]["content"][0]["text"] == "original"
    assert duplicate.messages[0] is duplicate.messages[1]
    assert duplicate.max_memory_size == 10


async def test_deepcopy_remains_usable_for_reads_and_writes():
    original = SimpleMemory(messages=[message("original")])
    duplicate = copy.deepcopy(original)
    await duplicate.add(message("copied"))
    assert await duplicate.get() == [message("original"), message("copied")]
    assert await original.get() == [message("original")]


async def test_deepcopy_of_locked_memory_can_be_used_independently():
    original = SimpleMemory(messages=[message("original")])
    async with original._lock:
        duplicate = copy.deepcopy(original)
        await asyncio.wait_for(duplicate.add(message("new")), timeout=1)
        assert await asyncio.wait_for(duplicate.get(), timeout=1) == [message("original"), message("new")]
    assert await original.get() == [message("original")]
