"""
Logging utility functions for Qwen3-VL RAG Retrieval System.

Requirements: 8.2, 8.5
- Provide consistent logging across all modules
- Support different log levels and formats
"""

import logging
import sys
from typing import Optional, Literal
from pathlib import Path


# Global logger registry
_loggers: dict = {}

# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_file: Optional[str] = None,
    format_style: Literal["default", "simple", "detailed"] = "default",
    name: str = "qwen3_vl_retrieval",
) -> logging.Logger:
    """
    Set up logging configuration for the package.
    
    Args:
        level: Logging level.
        log_file: Optional path to log file. If None, logs to stdout only.
        format_style: Format style for log messages.
        name: Logger name.
    
    Returns:
        Configured logger instance.
    """
    # Get format string
    format_map = {
        "default": DEFAULT_FORMAT,
        "simple": SIMPLE_FORMAT,
        "detailed": DETAILED_FORMAT,
    }
    log_format = format_map.get(format_style, DEFAULT_FORMAT)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(getattr(logging, level))
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # Add file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Store in registry
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "qwen3_vl_retrieval") -> logging.Logger:
    """
    Get a logger instance by name.
    
    If the logger hasn't been set up, creates a default one.
    
    Args:
        name: Logger name. Use dot notation for hierarchy,
              e.g., "qwen3_vl_retrieval.models"
    
    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create child logger if parent exists
    parent_name = "qwen3_vl_retrieval"
    if name.startswith(parent_name) and parent_name in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
        return logger
    
    # Set up default logger
    return setup_logging(name=name)


def set_verbosity(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    name: str = "qwen3_vl_retrieval",
) -> None:
    """
    Set the verbosity level for a logger.
    
    Args:
        level: New logging level.
        name: Logger name.
    """
    logger = get_logger(name)
    logger.setLevel(getattr(logging, level))
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level))


def disable_logging(name: str = "qwen3_vl_retrieval") -> None:
    """
    Disable logging for a specific logger.
    
    Args:
        name: Logger name to disable.
    """
    logger = get_logger(name)
    logger.setLevel(logging.CRITICAL + 1)


def enable_logging(
    name: str = "qwen3_vl_retrieval",
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    """
    Re-enable logging for a specific logger.
    
    Args:
        name: Logger name to enable.
        level: Logging level to set.
    """
    set_verbosity(level, name)


class LoggingContext:
    """
    Context manager for temporarily changing log level.
    
    Example:
        with LoggingContext("DEBUG"):
            # Debug logging enabled here
            pass
        # Original level restored
    """
    
    def __init__(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        name: str = "qwen3_vl_retrieval",
    ):
        self.level = level
        self.name = name
        self.original_level: Optional[int] = None
    
    def __enter__(self):
        logger = get_logger(self.name)
        self.original_level = logger.level
        set_verbosity(self.level, self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            logger = get_logger(self.name)
            logger.setLevel(self.original_level)
            for handler in logger.handlers:
                handler.setLevel(self.original_level)
        return False


def log_gpu_memory(logger: Optional[logging.Logger] = None) -> None:
    """
    Log current GPU memory usage.
    
    Args:
        logger: Logger to use. If None, uses default logger.
    """
    from .torch_utils import get_gpu_memory_info
    
    if logger is None:
        logger = get_logger()
    
    mem_info = get_gpu_memory_info()
    
    if mem_info["total"] > 0:
        logger.info(
            f"GPU Memory: {mem_info['allocated'] / 1e9:.2f}GB / "
            f"{mem_info['total'] / 1e9:.2f}GB "
            f"({mem_info['utilization']:.1f}% used)"
        )
    else:
        logger.info("No GPU available")
