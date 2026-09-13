from __future__ import annotations

import warnings

_REMAP = {
    "Graph": "Graph",
    "Edge": "Edge",
    "RunnableLike": "RunnableLike",
    "RunnableCallableSync": "RunnableCallableSync",
    "RunnableCallableAsync": "RunnableCallableAsync",
}


def __getattr__(name: str):
    if name in _REMAP:
        warnings.warn(
            f"Importing {name} from lumis.kit is deprecated. Use lumis.pipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import lumis.pipeline.graph as _graph

        return getattr(_graph, _REMAP[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_REMAP.keys())
