from .base_memory import BaseMemory

from openai.types.chat import ChatCompletionMessageParam


class SimpleMemory(BaseMemory):
    def __init__(self):
        self.__messages: list[ChatCompletionMessageParam] = []

    def add(self, message: ChatCompletionMessageParam):
        self.__messages.append(message)

    def get(self, *args, **kwargs) -> list[ChatCompletionMessageParam]:
        return self.__messages

    def clear(self):
        self.__messages = []

    def remove(self, index: int):
        self.__messages.pop(index)

    def update(self, index: int, message: ChatCompletionMessageParam):
        self.__messages[index] = message

    def insert(self, index: int, message: ChatCompletionMessageParam):
        self.__messages.insert(index, message)

    @property
    def length(self) -> int:
        return len(self.__messages)
