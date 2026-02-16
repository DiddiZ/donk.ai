from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from types import TracebackType
from typing import Self, override

import numpy as np


class DataLogger(ABC):
    """Base class for data loggers."""

    @abstractmethod
    def log(self, key: list[str], data: object) -> None:
        """Log one data.

        Args:
            key: Path-like, hierarchical name associated with the data
            data: Data to log, should be python primitives
        """

    def as_default(self) -> Self:
        """Make the the active data logger."""
        global _data_logger
        _data_logger = self
        return self

    def flush(self) -> None:
        """Flush data, if applicable."""

    def __enter__(self) -> Self:
        global _data_logger

        # Store previous logger
        self.stored_logger = _data_logger

        return self.as_default()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.flush()

        # Restore stored logger
        self.stored_logger.as_default()


class DummyDataLogger(DataLogger):
    """Logs directly to the void."""

    @override
    def log(self, key: list[str], data: object) -> None:
        """Ignore one data.

        Args:
            key: Path-like, hierarchical name associated with the data
            data: Data to log, should be python primitives
        """
        # Do nothing


class Context:
    """Context manager for hierarchical data logging."""

    def __init__(self, name: str) -> None:
        """Initialize this `Context`."""
        self.name = name

    def __enter__(self) -> Self:
        global _data_logger_context

        # Store previous context
        self.stored_context = _data_logger_context

        # Append new context
        _data_logger_context = [*_data_logger_context, self.name]

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        global _data_logger_context

        # Restore context
        _data_logger_context = self.stored_context


# Global state for the active logger
_data_logger: DataLogger = DummyDataLogger()
_data_logger_context = []


def log(**kwargs: object) -> None:
    """Log one or multiple pieces of data.

    Args:
        kwargs: Name and data pairs to log
    """
    for key, data in kwargs.items():
        _data_logger.log([*_data_logger_context, key], data)


def concat_iterations(data: dict, key_pattern: str, iterations: Iterable[int]) -> np.ndarray:
    """Concatenate log entries along an iteration axis.

    Args:
        data: Open shelve
        key_pattern: Format string for entries to concat
        iterations: Iterations to concat
    """
    return np.array([data[key_pattern.format(itr=itr)] for itr in iterations])
