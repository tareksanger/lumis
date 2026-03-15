from __future__ import annotations

import warnings

from .base_agent import BaseAgent


def __getattr__(name: str):
    if name == "GraphBasedAgent":
        warnings.warn(
            "GraphBasedAgent is deprecated. Use lumis.pipeline.Pipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from lumis.pipeline.pipeline import Pipeline
        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseAgent"]
