from __future__ import annotations

import warnings

from .react_agent import ReactAgent, ReActThought

try:
    from .storm.agent import StormAgent
except (ImportError, ModuleNotFoundError):
    StormAgent = None  # type: ignore[assignment, misc]


def __getattr__(name: str):
    if name == "QAResearchAgent":
        warnings.warn(
            "QAResearchAgent is deprecated. Use lumis.pipeline.QAResearchPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from lumis.pipeline.qa_research_pipeline import QAResearchPipeline
        return QAResearchPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ReActThought", "ReactAgent", "StormAgent"]
