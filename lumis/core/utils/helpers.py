from __future__ import annotations

from typing import TypeVar

from deepmerge import always_merger
from pydantic import BaseModel
import tiktoken


def serialize(obj):
    """
    Recursively serialize an object or dictionary, converting custom classes to dictionaries.
    """
    if isinstance(obj, BaseModel):
        return {key: serialize(value) for key, value in obj.model_dump().items()}
    elif hasattr(obj, "dict"):
        # Serialize custom class by its __dict__ attribute
        return {key: serialize(value) for key, value in obj.dict().items()}

    elif hasattr(obj, "__dict__"):
        # Serialize custom class by its __dict__ attribute
        return {key: serialize(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, dict):
        # Serialize dictionaries
        return {serialize(key): serialize(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        # Serialize iterables
        return type(obj)(serialize(item) for item in obj)
    else:
        # Return the object itself if it's a primitive type
        return obj


_tokenizer_instances: dict[str, tiktoken.Encoding] = {}


def get_tokenizer(chat_model: str) -> tiktoken.Encoding:
    if chat_model not in _tokenizer_instances:
        _tokenizer_instances[chat_model] = tiktoken.encoding_for_model(chat_model)
    return _tokenizer_instances[chat_model]


T = TypeVar("T", bound=BaseModel)


def merge_pydantic_models(base: T, nxt: T) -> T:
    """Merge two Pydantic model instances.

    The attributes of 'base' and 'nxt' that weren't explicitly set are dumped into dicts
    using '.model_dump(exclude_unset=True)', which are then merged using 'deepmerge',
    and the merged result is turned into a model instance using '.model_validate'.

    For attributes set on both 'base' and 'nxt', the value from 'nxt' will be used in
    the output result.
    """
    base_dict = base.model_dump(exclude_unset=True)
    nxt_dict = nxt.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, nxt_dict)
    return base.model_validate(merged_dict)
