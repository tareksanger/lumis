from __future__ import annotations

import logging
import traceback
from typing import Optional


class LoggerMixin:
    """
    A mixin class that sets up a logger for the inheriting class.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the logger.

        Args:
            verbose (bool): If True, set logging level to DEBUG. Otherwise, INFO.
            logger (Optional[logging.Logger]): An existing logger to use. If None, a new logger is created.
        """
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def log_exception(self, e: Exception, level: int = logging.ERROR):
        """
        Log an exception with an optional stack trace.

        Args:
            e (Exception): The exception to log.
            level (int): The logging level (e.g., logging.ERROR, logging.WARNING).
        """
        message = f"{e}\n{traceback.format_exc()}"
        self.logger.log(level, message)
