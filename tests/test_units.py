import pytest

import numpy as np
import pandas as pd
from astropy.units import Quantity, Unit, m
from pandas.tests.extension import base
from pandas.tests.extension.base import BaseOpsUtil

from pandas_implants.units import UnitsDtype, UnitsExtensionArray, UnitsSeriesAccessor

try:
    from pandas.conftest import (
        all_arithmetic_operators,
        all_compare_operators,
    )
except:
    _all_arithmetic_operators = ['__add__', # '__radd__',
        '__sub__', # '__rsub__',
        '__mul__', # '__rmul__',
        '__floordiv__', #'__rfloordiv__',
        '__truediv__', #'__rtruediv__',
        # '__pow__', # '__rpow__',
        '__mod__', # '__rmod__'
    ]

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


@pytest.fixture(params=[" ", "mm", "kg s"])
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
    # divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def test_arith_series_with_scalar_pow(self, data):
        s = pd.Series(data)
        result = s ** 2
        expected = pd.Series([1, 4] + 98 * [9], dtype="unit[m^2]")
        self.assert_series_equal(result, expected)

    def test_error(self, data, all_arithmetic_operators):
        pass

    @pytest.mark.skip("Not implemented yet")
    def test_divmod(self, data):
        raise NotImplementedError


class TestComparisonOps(base.BaseComparisonOpsTests):
    def teeeest_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)

        result_2m = getattr(s, op_name)(2 * m)
        expected_2m = getattr(data.data, op_name)(2)
        self.assert_series_equal(result_2m, expected_2m)


class TestReduce:
    def test_sum(self, data, data_missing):
        assert pd.Series(data).sum() == 297 * m
        assert np.isnan(pd.Series(data_missing).sum(skipna=False))
        assert pd.Series(data_missing).sum() == 1 * m

    def test_mean(self, data):
        assert np.allclose(pd.Series(data).mean() / m, 2.97)

    def test_min(self, data):
        assert pd.Series(data).min() == 1 * m

    def test_max(self, data):
        assert pd.Series(data).max() == 3 * m

    def test_median(self, data):
        assert pd.Series(data).median() == 3 * m

    def test_std(self, data):
        assert np.allclose(pd.Series(data).std() / m, 0.2227015)

    def test_sem(self, data):
        assert np.allclose(pd.Series(data).sem() / m, 0.02227015033536137)

    def test_var(self, data):
        assert np.allclose(pd.Series(data).var() / (m **2),  0.0495959595959596)

    def test_unsupported(self, data):
        for method in ["any", "all", "prod"]:
            with pytest.raises(TypeError):
                getattr(pd.Series(data), method)()


class TestRepr:
    def test_repr(self, simple_data):
        assert "<UnitsExtensionArray>\n[1.0 m, 2.0 m, 3.0 m]\nLength: 3, dtype: unit[m]" == repr(simple_data)

    def test_series_repr(self, simple_data):
        assert "0   1.0 m\n1   2.0 m\n2   3.0 m\ndtype: unit[m]" == repr(pd.Series(simple_data))


class TestUnitsSeriesAccessor(BaseOpsUtil):
    def test_init(self, simple_data):
        s = pd.Series(simple_data)
        assert isinstance(s.units, UnitsSeriesAccessor)

    def test_invalid_type(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError):
            _ = s.units

    def test_to(self, simple_data):
        s = pd.Series(simple_data)
        result = s.units.to("mm")
        expected = pd.Series([1000, 2000, 3000], dtype="unit[mm]")
        self.assert_series_equal(result, expected)

    def test_unit(self, simple_data):
        s = pd.Series(simple_data)
        assert s.units.unit == Unit("m")

    def test_to_si(self):
        s = pd.Series([1, 2, 3], dtype="unit[km]")
        result = s.units.to_si()
        expected = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        self.assert_series_equal(result, expected)

    def test_temperature(self):
        s = pd.Series([0, 100], dtype="unit[deg_C]")

        s_f = s.units.to("deg_F")
        s_f_expected = pd.Series([32, 212], dtype="unit[deg_F]")
        self.assert_series_equal(s_f, s_f_expected)


class TestUnitsDataFrameAccessor(BaseOpsUtil):
    def test_df_to_si(self):
        df = pd.DataFrame({
            "a": pd.Series([1, 2, 3], dtype="unit[km]"),
            "b": pd.Series([2, 3, 4], dtype="unit[hour]")
        })
        result = df.units.to_si()
        expected = pd.DataFrame({
            "a": pd.Series([1000, 2000, 3000], dtype="unit[m]"),
            "b": pd.Series([7200, 10800, 14400], dtype="unit[s]")
        })
        self.assert_frame_equal(result, expected)