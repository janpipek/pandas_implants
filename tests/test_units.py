import pytest

import numpy as np
from astropy.units import Quantity, Unit, m
from pandas.tests.extension import base

from pandas_implants.units import UnitsDtype, UnitsExtensionArray


@pytest.fixture
def data():
    return UnitsExtensionArray([1, 2] + 98 * [3], m)




@pytest.fixture(params=["", "mm", "kg s"])
def dtype(request):
    return UnitsDtype(request.param)

class TestConstructors(base.BaseConstructorsTests): pass

class TestDtype(base.BaseDtypeTests): pass


