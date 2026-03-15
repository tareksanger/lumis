"""Backward-compatibility stub. The utils module has moved to lumis.pipeline.utils."""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "lumis.kit.utils is deprecated. Use lumis.pipeline.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from lumis.pipeline.utils import *  # noqa: F401, F403
