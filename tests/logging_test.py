import shelve
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


def test_shelve_data_logger(tmp_path: Path) -> None:
    import donk.datalogging as datalogging

    def inner_fun(a: int, b: int) -> int:
        c = a + b

        datalogging.log(c=c)

        return c**2

    temp_file = tmp_path / "data"
    with datalogging.ShelveDataLogger(temp_file):
        a = 1
        b = 2

        datalogging.log(a=a, b=b)

        with datalogging.Context("inner_fun"):
            d = inner_fun(a, b)
        datalogging.log(d=d)

    with shelve.open(temp_file) as data:
        assert data["a"] == 1
        assert data["b"] == 2
        assert data["inner_fun/c"] == 3
        assert data["d"] == 9


def test_concat_iterations(tmp_path: Path) -> None:
    import donk.datalogging as datalogging

    temp_file = tmp_path / "data"
    with datalogging.ShelveDataLogger(temp_file):
        for itr in range(10):
            with datalogging.Context(f"itr_{itr:02d}"):
                datalogging.log(a=itr**2)

    with shelve.open(temp_file) as data:
        assert_allclose(
            datalogging.concat_iterations(dict(data), "itr_{itr:02d}/a", range(10)),  # noqa: RUF027
            np.arange(10) ** 2,
        )
