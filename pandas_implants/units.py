import builtins
import operator
import re
from typing import Union

import numpy as np
from astropy.units import Quantity, Unit
from numpy.core._multiarray_umath import ndarray
from pandas.api.extensions import (
    ExtensionDtype,
    ExtensionArray,
    ExtensionScalarOpsMixin,
    register_extension_dtype,
)
from pandas.compat import set_function_name
from pandas.core.dtypes.inference import is_list_like
from pandas.core import ops



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
        match = re.match(f"{cls.BASE_NAME}\\[(?P<name>.+)\\]", string)
        if not match:
            raise TypeError(f"Invalid UnitsDtype string: {string}")
        return cls(match["name"])

    @classmethod
    def construct_array_type(cls) -> builtins.type:
        return UnitsExtensionArray

    @property
    def name(self) -> str:
        return f"{self.BASE_NAME}[{self.unit.to_string()}]"

    def __repr__(self):
        return f"{self.__class__.__name__}(\"{self.unit.name}\")"


class UnitsExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
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

    @classmethod
    def _to_quantity(cls, obj) -> Quantity:
        if isinstance(obj, Quantity):
            return obj
        elif isinstance(obj, cls):
            return Quantity(obj.data, obj.unit)
        else:
            return Quantity(obj)

    def _formatter(self, boxed=False):
        return lambda x: str(x)

    def __getitem__(self, item):
        if np.isscalar(item):
            return Quantity(self.data[item], unit=self.unit)
        else:
            return self.__class__(self.data[item], unit=self.unit)

    def isna(self):
        return np.isnan(self.data)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        def _binop(self, other):
            self_q = cls._to_quantity(self)
            other_q = cls._to_quantity(other)
            result_q = op(self_q, other_q)
            if coerce_to_dtype:
                return cls(result_q)
            else:
                return result_q

        op_name = ops._get_op_name(op, True)
        return set_function_name(_binop, op_name, cls)
    
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


UnitsExtensionArray._add_arithmetic_ops()
UnitsExtensionArray._add_comparison_ops()