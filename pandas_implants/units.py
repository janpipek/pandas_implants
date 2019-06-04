import builtins
import re
from typing import Union

import numpy as np
from astropy.units import Quantity, Unit
from numpy.core._multiarray_umath import ndarray
from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
    register_extension_dtype,
)


@register_extension_dtype
class UnitsDtype(ExtensionDtype):
    BASE_NAME = "unit"

    type = Quantity
    kind = "f"

    _is_numeric = True

    def __init__(self, unit):
        if isinstance(unit, Unit):
            self.unit = unit
        else:
            self.unit = Unit(unit)


    @classmethod
    def construct_from_string(cls, string) -> "UnitsDtype":
        match = re.match(f"{cls.BASE_NAME}\\[(?P<name>\\w+)\\]", string)
        if not match:
            raise TypeError(f"Invalid UnitsDtype string: {string}")
        return cls(match["name"])

    @classmethod
    def construct_array_type(cls) -> builtins.type:
        return UnitsExtensionArray

    @property
    def name(self) -> str:
        return f"{self.BASE_NAME}[{self.unit.name}]"

    def __repr__(self):
        return f"{self.__class__.__name__}(\"{self.unit.name}\")"


class UnitsExtensionArray(ExtensionArray):
    def __init__(self, array, unit: Union[None, str, Unit] = None, *, copy: bool = True):
        if isinstance(array, Quantity):
            if copy:
                array = array.copy()
            self._dtype = UnitsDtype(array.unit)
            self.data = array.value.astype(float)
        else:
            if not unit:
                raise ValueError("You have to provide a unit!")
            array = np.array(array, dtype=float) if copy else np.asarray(array, dtype=float)
            self.data = array.astype(float)
            self._dtype = UnitsDtype(unit)

    @property
    def dtype(self) -> UnitsDtype:
        return self._dtype

    @property
    def unit(self) -> Unit:
        return self.dtype.unit

    def __len__(self):
        return len(self.data)

    @property
    def nbytes(self):
        return self.data.nbytes

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False) -> "UnitsExtensionArray":
        if dtype:
            result = cls(scalars, unit=dtype.unit, copy=copy)
        else:
            result = cls(scalars, copy=copy)
        return result

    def _formatter(self, boxed=False):
        return lambda x: f"{x} {self.unit.name}"

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.data[item] #, unit=self.unit)
        else:
            return self.__class__(self.data[item], unit=self.unit)

    def isna(self):
        return np.isnan(self.data)

    # * _from_sequence
    #* _from_factorized
    #* __getitem__
    #* __len__
    #* dtype
    #* nbytes
    #* isna
    #* take
    #* copy
    #* _concat_same_type