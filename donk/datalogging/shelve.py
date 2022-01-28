import shelve
from typing import List

from donk.datalogging.datalogging import DataLogger


class ShelveDataLogger(DataLogger):
    """Log data as single shelve file."""

    def __init__(self, file) -> None:
        super().__init__()

        self.shelve = shelve.open(file)

    def log(self, key: List[str], data):
        """Log one data.

        Args:
            key: Path-like, hierarchical name associated with the data
            data: Data to log, should be python primitives
        """
        # Store data
        self.shelve["/".join(key)] = data

    def flush(self):
        """Write data to the file."""
        self.shelve.sync()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)

        # Close shelve
        self.shelve.close()
