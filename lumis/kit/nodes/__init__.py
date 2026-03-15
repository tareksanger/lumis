from __future__ import annotations

import warnings

_REMAP = {
    "LLMChatNode": "LLMChatNode",
    "LLMStructuredNode": "LLMStructuredNode",
}


def __getattr__(name: str):
    if name in _REMAP:
        warnings.warn(
            f"Importing {name} from lumis.kit.nodes is deprecated. Use lumis.pipeline.nodes instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import lumis.pipeline.nodes as _nodes
        return getattr(_nodes, _REMAP[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_REMAP.keys())
