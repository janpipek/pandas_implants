import pytest

import numpy as np
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


class TestConstructors(base.BaseConstructorsTests): pass


class TestCasting(base.BaseCastingTests): pass


class TestDtype(base.BaseDtypeTests): pass


class TestGetitem(base.BaseGetitemTests): pass