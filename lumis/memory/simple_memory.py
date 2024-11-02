from .base_memory import BaseMemory

from openai.types.chat import ChatCompletionMessageParam


class SimpleMemory(BaseMemory):
    def __init__(self, max_memory_size: int = 2000):
        self.max_memory_size = max_memory_size
        self.__messages: list[ChatCompletionMessageParam] = []

    def add(self, message: ChatCompletionMessageParam):
        self.__messages.append(message)

    def get(self, *args, **kwargs) -> list[ChatCompletionMessageParam]:
        if len(self.__messages) <= self.max_memory_size:
            return self.__messages

        # Calculate the number of items to take from the start and the end
        half_max = self.max_memory_size // 2

        start = self.__messages[:half_max]
        end = self.__messages[-half_max:]

        return start + end

    def clear(self):
        self.__messages = []

    def remove(self, index: int):
        self.__messages.pop(index)

    def update(self, index: int, message: ChatCompletionMessageParam):
        self.__messages[index] = message

    def insert(self, index: int, message: ChatCompletionMessageParam):
        pass

    @property
    def length(self) -> int:
        return len(self.__messages)
