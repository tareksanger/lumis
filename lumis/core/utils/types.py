from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator
import logging
from typing import Any

from lumis.core.utils.string import snake_to_title
from pydantic.alias_generators import to_snake

logger = logging.getLogger(__name__)

def _metadata_to_dict(metadata: list[Any]) -> dict[str, Any]:
    result = {}
    for item in metadata:
        if isinstance(item, dict):
            for k, v in item.items():
                result[str(k)] = v
        elif isinstance(item, tuple):
            result[str(item[0])] = item[1]
    return result


class BaseSchema(BaseModel):
    """
    An Extension of BaseModel used to easily communicate between services.

    Converts incoming camelCase to snake_case.
    """

    def to_context_str(self, deliminator: str = "\n", depth: int = 0) -> str:  # noqa: C901
        parts = []
        indent = "\t" * depth

        for field_name, field_info in self.model_fields.items():
            context = _metadata_to_dict(field_info.metadata).get("context", False)
            if not context:
                logger.debug(f"{field_name} left out of context while generating context string.")
                continue

            value = getattr(self, field_name)
            field_title = snake_to_title(field_name)

            if isinstance(value, BaseSchema):
                # Recursively handle nested BaseSchema models
                nested_str = value.to_context_str(deliminator, depth + 1)
                parts.append(f"{indent}{field_title}:\n{nested_str}")

            elif isinstance(value, list) or isinstance(value, tuple):
                # Handle iterables by enumerating each item
                parts.append(f"{indent}{field_title}:")
                for item in value:
                    if isinstance(item, BaseSchema):
                        # Recursively handle nested models in lists
                        nested_str = item.to_context_str(deliminator, depth + 1)
                        # Prepend an indent and a dash
                        lines = nested_str.split(deliminator)

                        indented_lines = [("\t" * (depth + 1)) + "- " + lines[0]] + [("\t" * (depth + 1)) + line for line in lines[1:] if line.strip()]
                        parts.extend(indented_lines)
                    else:
                        indent_str = "\t" * (depth + 1)
                        # Scalar or non-model item
                        parts.append(f"{indent_str}- {item}")

            elif isinstance(value, dict):
                # Handle dictionaries by recursively processing their values
                parts.append(f"{indent}{field_title}:")
                for k, v in value.items():
                    key_title = snake_to_title(k)

                    indent_str = "\t" * (depth + 1)
                    if isinstance(v, BaseSchema):
                        nested_str = v.to_context_str(deliminator, depth + 1)
                        parts.append(f"{indent_str}{key_title}:\n{nested_str}")
                    else:
                        parts.append(f"{indent_str}{key_title}: {v}")

            else: 
                # Scalar values
                parts.append(f"{indent}{field_title}: {value}")

        return deliminator.join(parts)

    model_config = ConfigDict(
        alias_generator=to_snake,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        extra="ignore",
        # TODO: Properly type this and remove the ignore
        # TODO: move 'hidden' to Field MetaData
        json_schema_extra=lambda schema, _: schema.update(
            {
                "strict": False,
                "properties": {
                    # iterates through the properties of the schema and excludes any property that has a 'hidden' attribute set to True.
                    k: v
                    for k, v in schema.get("properties", {}).items()  # type: ignore #TODO: TEMP
                    if not v.get("hidden", False)  # type: ignore
                },
            }
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def convert_keys_to_snake_case(cls, values):
        if isinstance(values, dict):
            return {to_snake(k): v for k, v in values.items()}
        return values

