import pytest


@pytest.fixture(scope="module")
def memory_class():
    import django
    from django.conf import settings

    if not settings.configured:
        settings.configure(INSTALLED_APPS=[], DATABASES={}, SECRET_KEY="unit-test-only")
    django.setup()
    from lumis.core.django.models.chat_memory import ChatMemory

    class Memory(ChatMemory):
        class Meta:
            app_label = "lumis_unit_tests"

        @classmethod
        def get_token_count_from_content(cls, content):
            return len(content.split())

    return Memory


@pytest.mark.parametrize("value", [None, "plain text", {"text": "hello"}, ["a", "b"]])
def test_content_and_refinement_roundtrip(memory_class, value):
    memory = memory_class(role="assistant", name="agent")
    memory.content = value
    memory.refined_content = value
    assert memory.content == value
    assert memory.refined_content == value


def test_message_conversion_prefers_requested_refinement(memory_class):
    memory = memory_class(role="assistant", name="agent")
    memory.content = "original"
    memory.refined_content = "refined"
    assert memory.to_chat_completion_message() == {"role": "assistant", "name": "agent", "content": "original"}
    assert memory.to_chat_completion_message(refinement=True)["content"] == "refined"
    memory.refined_content = None
    assert memory.to_chat_completion_message(refinement=True)["content"] == "original"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ""),
        ("   ", ""),
        (" plain text ", "plain text"),
        ('"text"', "text"),
        ("42", "42"),
        (False, "False"),
        ({"text": "hello", "ignored": "metadata"}, "hello"),
        ({"other": ["one", {"content": "two"}]}, "one two"),
        (["one", None, True, 3], "one True 3"),
        (object(), ""),
    ],
)
def test_tokenizable_content_handles_nested_formats(memory_class, raw, expected):
    assert memory_class._prepare_tokenizable_content(raw) == expected


def test_count_and_string_representation(memory_class):
    memory = memory_class(role="user", name="reader")
    memory.content = [{"text": "one two"}, {"text": "three"}]
    assert memory.get_token_count() == 3
    memory.content = None
    assert memory.get_token_count() == 0
    memory.content = " " * 5
    assert memory.get_token_count() == 0
    memory.content = "a" * 60
    assert str(memory) == "user: " + "a" * 47 + "..."


def test_abstract_tokenizer_requires_implementation(memory_class):
    from lumis.core.django.models.chat_memory import ChatMemory

    assert ChatMemory.get_token_count_from_content("") == 0
    with pytest.raises(NotImplementedError):
        ChatMemory.get_token_count_from_content("text")
