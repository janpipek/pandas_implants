import pytest

import numpy as np
import pandas as pd
from astropy.units import Quantity, Unit, m
from pandas.tests.extension import base

from pandas_implants.units import UnitsDtype, UnitsExtensionArray


@pytest.fixture
def data():
    return UnitsExtensionArray([1, 2] + 98 * [3], m)


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return UnitsExtensionArray([np.nan * m, 1 * m])


@pytest.fixture
def simple_data():
    return UnitsExtensionArray([1, 2, 3], m)


@pytest.fixture
def incoercible_data():
    return [Quantity(1, "kg"), Quantity(1, "m")]


@pytest.fixture
def coercible_data():
    return [Quantity(1, "kg"), Quantity(1, "g")]


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture(params=["", "mm", "kg s"])
def dtype(request):
    return UnitsDtype(request.param)


@pytest.fixture
def na_cmp():
    return np.isnan


@pytest.fixture
def na_value():
    # Must be the same unit as others
    return np.nan * m


class TestConstructors(base.BaseConstructorsTests): pass


class TestCasting(base.BaseCastingTests): pass


class TestDtype(base.BaseDtypeTests): pass


class TestGetitem(base.BaseGetitemTests): pass
    

class TestRepr:
    def test_repr(self, simple_data):
        assert "<UnitsExtensionArray>\n[1.0 m, 2.0 m, 3.0 m]\nLength: 3, dtype: unit[m]" == repr(simple_data)

    def test_series_repr(self, simple_data):
        assert "0   1.0 m\n1   2.0 m\n2   3.0 m\ndtype: unit[m]" == repr(pd.Series(simple_data))