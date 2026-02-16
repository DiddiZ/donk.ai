from __future__ import annotations

import dbm.dumb
import pickle
from pathlib import Path
from types import TracebackType
from typing import Self, override

from donk.datalogging.datalogging import DataLogger


class ShelveDataLogger(DataLogger):
    """Log data as single shelve file."""

    def __init__(self, file: str | Path) -> None:
        """Initialize this `ShelveDataLogger`."""
        super().__init__()

        self.file = file

    @override
    def log(self, key: list[str], data: object) -> None:
        """Log one data.

        Args:
            key: Path-like, hierarchical name associated with the data
            data: Data to log, should be python primitives
        """
        # Store data
        self.__data["/".join(key)] = pickle.dumps(data)

    @override
    def flush(self) -> None:
        """Write data to the file."""
        self.__data.sync()

    @override
    def __enter__(self) -> Self:
        # Open database
        self.__data = dbm.dumb.open(self.file, "c")

        return super().__enter__()

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)

        # Close shelve
        self.__data.close()
