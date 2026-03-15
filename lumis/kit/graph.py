"""Backward-compatibility stub. The graph module has moved to lumis.pipeline.graph."""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "lumis.kit.graph is deprecated. Use lumis.pipeline.graph instead.",
    DeprecationWarning,
    stacklevel=2,
)

from lumis.pipeline.graph import *  # noqa: F401, F403
from lumis.pipeline.graph import (  # noqa: F401 — explicit re-exports for type checkers
    Edge,
    Graph,
    NodeCallable,
    NodeLike,
    Runnable,
    RunnableCallableAsync,
    RunnableCallableSync,
    RunnableLike,
    S,
    StateProtocol,
    TERMINATE,
    Trace,
)
