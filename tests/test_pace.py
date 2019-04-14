import numpy as np
import pandas as pd
import pytest

import pandas_implants.pace
from pandas_implants.pace import PaceArray
from pandas.tests.extension import base


@pytest.fixture
def data():
    return PaceArray(["1:00", "2:00"] + 98 * ["3:00"])


# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#testing-extension-arrays
#class TestConstructors(base.BaseConstructorsTests):
#    pass


class TestPaceArrayConstructor():
    def test_with_strings(self):
        array = PaceArray(["4:00", "4:20"])
        assert np.allclose(array, [240, 260])

    def test_with_ints(self):
        array = PaceArray([240, 260])
        assert np.allclose(array, [240, 260])

    def test_with_floats(self):
        array = PaceArray([240.1, 260.5])
        assert np.allclose(array, [240.1, 260.5])


def test_create_series_by_name():
    series = pd.Series(["4:00", "4:20"], dtype="pace")
    assert np.allclose(np.array(series), [240, 260])
