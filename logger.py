"""Centralized logging configuration for the application."""

import logging
import sys

# Default format for all loggers
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LEVEL = logging.INFO

# Get module logger (inherits config from setup_logging)
logger = logging.getLogger(__name__)

def setup_logging(level: int = DEFAULT_LEVEL) -> None:
    """
    Configure the root logger for the application.

    Call this once at application startup (e.g., in main.py or inference.py).
    All modules using logging.getLogger(__name__) will inherit this configuration.

    Args:
        level: The logging level (e.g., logging.DEBUG, logging.INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    This is a convenience wrapper around logging.getLogger().

    Args:
        name: Usually __name__ of the calling module

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
