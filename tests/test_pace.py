import operator

import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension import base

from pandas_implants.pace import PaceArray, PaceDtype


# Fixtures as defined in https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/conftest.py
@pytest.fixture
def data():
    return PaceArray(["1:00", "2:00"] + 98 * ["3:00"])


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return np.nan


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    return operator.is_


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return PaceArray([np.nan, "1:00"])


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def dtype():
    return PaceDtype()


# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#testing-extension-arrays
class TestConstructors(base.BaseConstructorsTests): pass


class TestCasting(base.BaseCastingTests): pass


class TestDtype(base.BaseDtypeTests): pass


class TestGetitem(base.BaseGetitemTests): pass


class TestPaceArrayConstructor():
    def test_with_strings(self):
        array = PaceArray(["4:00", "4:20"])
        assert np.allclose(array, [240, 260])

    def test_with_invalid_strings(self):
        with pytest.raises(ValueError):
            _ = PaceArray(["4:a00", "4:20x"])

    def test_with_ints(self):
        array = PaceArray([240, 260])
        assert np.allclose(array, [240, 260])

    def test_with_floats(self):
        array = PaceArray([240.1, 260.5])
        assert np.allclose(array, [240.1, 260.5])


class TestSeriesConstructor():
    def test_with_cls(self):
        array = PaceArray(["4:00", "4:20"])
        series = pd.Series(array)
        assert np.allclose(np.array(series), [240, 260])

    def test_with_name(self):
        series = pd.Series(["4:00", "4:20"], dtype="pace")
        assert np.allclose(np.array(series), [240, 260])
