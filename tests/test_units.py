import pytest

import numpy as np
import pandas as pd
from astropy.units import Quantity, Unit, m
from pandas.tests.extension import base

from pandas_implants.units import UnitsDtype, UnitsExtensionArray

try:
    from pandas.conftest import (
        all_arithmetic_operators,
        all_compare_operators,
    )
except:
    _all_arithmetic_operators = ['__add__', '__radd__',
        '__sub__', '__rsub__',
        '__mul__', '__rmul__',
        '__floordiv__', '__rfloordiv__',
        '__truediv__', '__rtruediv__',
        '__pow__', '__rpow__',
        '__mod__', '__rmod__']

    @pytest.fixture(params=_all_arithmetic_operators)
    def all_arithmetic_operators(request):
        return request.param

    @pytest.fixture(params=['__eq__', '__ne__', '__le__',
                        '__lt__', '__ge__', '__gt__'])
    def all_compare_operators(request):
        return request.param


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
    # Note: np.nan != np.nan
    def cmp(x, y):
        if np.isnan(x.value):
            return np.isnan(y.value)
        else:
            return x == y
    return cmp


@pytest.fixture
def na_value():
    # Must be the same unit as others
    return np.nan * m


class TestConstructors(base.BaseConstructorsTests): pass


class TestCasting(base.BaseCastingTests): pass


class TestDtype(base.BaseDtypeTests): pass


class TestGetitem(base.BaseGetitemTests):
    def test_unitless(self):
        series = pd.Series([0, 1, 2], dtype="unit[]")
        new_index = [2, 4]
        result = series.reindex(new_index)
        expected = pd.Series([2, np.nan], dtype="unit[]", index=new_index)
        self.assert_series_equal(result, expected)


class TestPrinting(base.BasePrintingTests):
    pass


class TestArithmeticsOps(base.BaseArithmeticOpsTests):
    pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    pass


class TestRepr:
    def test_repr(self, simple_data):
        assert "<UnitsExtensionArray>\n[1.0 m, 2.0 m, 3.0 m]\nLength: 3, dtype: unit[m]" == repr(simple_data)

    def test_series_repr(self, simple_data):
        assert "0   1.0 m\n1   2.0 m\n2   3.0 m\ndtype: unit[m]" == repr(pd.Series(simple_data))