from __future__ import annotations

import dbm.dumb
import pickle
from typing import List

from donk.datalogging.datalogging import DataLogger


class ShelveDataLogger(DataLogger):
    """Log data as single shelve file."""

    def __init__(self, file) -> None:
        super().__init__()

        self.file = file

    def log(self, key: List[str], data) -> None:
        """Log one data.

        Args:
            key: Path-like, hierarchical name associated with the data
            data: Data to log, should be python primitives
        """
        # Store data
        self.__data["/".join(key)] = pickle.dumps(data)

    def flush(self) -> None:
        """Write data to the file."""
        self.__data.sync()

    def __enter__(self) -> ShelveDataLogger:
        # Open database
        self.__data = dbm.dumb.open(self.file, "c")

        return super().__enter__()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)

        # Close shelve
        self.__data.close()
