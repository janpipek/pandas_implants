import pytest

import numpy as np
from astropy.units import Quantity, Unit, m
from pandas.tests.extension import base

from pandas_implants.units import UnitsDtype, UnitsExtensionArray


@pytest.fixture
def data():
    return UnitsExtensionArray([1, 2] + 98 * [3], m)


class TestConstructors(base.BaseConstructorsTests): pass