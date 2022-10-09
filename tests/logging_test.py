import shelve
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


class Test_ShelveDataLogger(unittest.TestCase):
    def test_logging(self):
        """Make sure TF dependenciesy are loaded properly."""
        import donk.datalogging as datalogging

        with tempfile.TemporaryDirectory() as tmpdirname:

            def inner_fun(a, b):
                c = a + b

                datalogging.log(c=c)

                return c ** 2

            temp_file = str(Path(tmpdirname) / "data")
            with datalogging.ShelveDataLogger(temp_file):
                a = 1
                b = 2

                datalogging.log(a=a, b=b)

                with datalogging.Context("inner_fun"):
                    d = inner_fun(a, b)
                datalogging.log(d=d)

            with shelve.open(temp_file) as data:
                self.assertEqual(data["a"], 1)
                self.assertEqual(data["b"], 2)
                self.assertEqual(data["inner_fun/c"], 3)
                self.assertEqual(data["d"], 9)

    def test_concat_iterations(self):
        import donk.datalogging as datalogging

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_file = str(Path(tmpdirname) / "data")
            with datalogging.ShelveDataLogger(temp_file):
                for itr in range(10):
                    with datalogging.Context(f"itr_{itr:02d}"):
                        datalogging.log(a=itr ** 2)

            with shelve.open(temp_file) as data:
                assert_allclose(datalogging.concat_iterations(data, "itr_{itr:02d}/a", range(10)), np.arange(10) ** 2)
