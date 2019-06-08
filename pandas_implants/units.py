import builtins
import operator
import re
from typing import Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.units import Quantity, Unit
from astropy.units import imperial
from astropy.units.format.generic import Generic
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    ExtensionScalarOpsMixin,
    register_extension_dtype,
    register_series_accessor,
    register_dataframe_accessor,
)
from pandas.api.types import is_list_like, is_array_like, is_scalar
from pandas.compat import set_function_name
from pandas.core import nanops
from pandas.core import ops
from pandas.core.algorithms import take
from pandas.core.dtypes.generic import ABCIndexClass, ABCSeries

imperial.enable()


@register_extension_dtype
class UnitsDtype(ExtensionDtype):
    BASE_NAME = "unit"

    type = Quantity
    kind = "f"

    _is_numeric = True
    _metadata = ("unit",)

    def __init__(self, unit):
        if isinstance(unit, Unit):
            self.unit = unit
        else:
            self.unit = Unit(unit)

    @classmethod
    def construct_from_string(cls, string) -> "UnitsDtype":
        match = re.match(f"{cls.BASE_NAME}\\[(?P<name>.*)\\]$", string)
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
        return f'{self.__class__.__name__}("{self.unit.to_string()}")'


class UnitsExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    def __init__(
        self, array, unit: Union[None, str, Unit] = None, *, copy: bool = True
    ):
        if isinstance(array, Quantity):
            if copy:
                array = array.copy()
            self._dtype = UnitsDtype(array.unit)
            self.data = array.value.astype(float)
        else:
            q = Quantity(array)
            if q.unit.is_unity():
                if unit:
                    q = q * unit
            else:
                if unit and q.unit != unit:
                    raise ValueError("Dtypes are not equivalent")

            self._dtype = UnitsDtype(q.unit)
            self.data = q.value.astype(float)

    @property
    def dtype(self) -> UnitsDtype:
        return self._dtype

    @property
    def unit(self) -> Unit:
        return self.dtype.unit

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype else self.data

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
        elif is_list_like(obj):
            return Quantity(list(obj))
        else:
            return Quantity(obj)

    def to_quantity(self) -> Quantity:
        return self._to_quantity(self)

    def to(
        self, new_unit: Union[str, Unit], equivalencies=None
    ) -> "UnitsExtensionArray":
        q = self.to_quantity()
        try:
            new_data = q.to(new_unit, equivalencies or [])
        except u.UnitConversionError:
            if q.unit.physical_type == "temperature":
                new_data = q.to(new_unit, u.temperature())
            else:
                raise
        return UnitsExtensionArray(new_data)

    def astype(self, dtype, copy=True):
        def _as_units_dtype(unit):
            return self.to(unit)

        if isinstance(dtype, UnitsDtype):
            return _as_units_dtype(dtype.unit)
        elif isinstance(dtype, str):
            try:
                dtype = UnitsDtype(dtype)
                return _as_units_dtype(dtype.unit)
            except:
                pass

        # Fall-back to default variant
        return ExtensionArray.astype(self, dtype, copy=copy)

    def _formatter(self, boxed=False):
        return lambda x: (str(x) if isinstance(x, Quantity) else f"{x} {self.unit}")

    def __getitem__(self, item):
        if np.isscalar(item):
            return Quantity(self.data[item], unit=self.unit)
        else:
            return self.__class__(self.data[item], unit=self.unit)

    def take(self, indices, allow_fill=False, fill_value=None) -> "UnitsExtensionArray":
        if allow_fill:
            if fill_value is None or np.isnan(fill_value):
                fill_value = np.nan
            else:
                fill_value = fill_value.value
        values = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        return UnitsExtensionArray(values, self.unit)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # to_concat = list(to_concat)
        if len(to_concat) == 0:
            return cls([])
        elif len(to_concat) == 1:
            return to_concat[0]
        elif len(set(item.unit for item in to_concat)) != 1:
            raise ValueError("Not all concatenated arrays have the same units.")
        else:
            return cls(
                np.concatenate([item.data for item in to_concat]), to_concat[0].unit
            )

    def isna(self):
        return np.isnan(self.data)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        # Overloaded from the default variant
        # to by-pass conversion to numpy arrays.
        def _binop(self, other):
            if isinstance(other, (ABCSeries, ABCIndexClass)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            elif is_scalar(other):
                if op_name in [
                    "__eq__",
                    "__ne__",
                    "__lt__",
                    "__gt__",
                    "__le__",
                    "__ge__",
                ]:
                    return NotImplemented

            elif is_array_like(other):
                if self.dtype != other.dtype:
                    if op_name in ["__eq__", "__ne__"]:
                        return NotImplemented
                    elif op_name in ["__lt__", "__gt__", "__le__", "__ge__"]:
                        raise TypeError

            self_q = cls._to_quantity(self)
            other_q = cls._to_quantity(other)
            result_q = op(self_q, other_q)
            if coerce_to_dtype:
                return cls(result_q)
            else:
                return result_q

        op_name = ops._get_op_name(op, True)
        return set_function_name(_binop, op_name, cls)

    def copy(self, deep=False):
        return self.__class__(self.data, self.unit, copy=True)

    def _reduce(self, name, skipna=True, **kwargs):
        # Borrowed from IntegerArray

        to_proxy = ["min", "max", "sum", "mean", "std", "var"]
        to_nanops = ["median", "sem"]
        to_error = ["any", "all", "prod"]

        # TODO: Check the dimension of this
        to_implement_yet = ["kurt", "skew"]

        if name in to_proxy:
            q = self.to_quantity()
            if name in ["std", "var"]:
                kwargs = {"ddof": kwargs.pop("ddof", 1)}
            else:
                kwargs = {}
            if skipna:
                q = q[~np.isnan(q)]
            return getattr(q, name)(**kwargs)

        elif name in to_nanops:
            data = self.data
            method = getattr(nanops, "nan" + name)
            result_without_dim = method(data, skipna=skipna)
            return Quantity(result_without_dim, self.unit)

        elif name in to_error:
            raise TypeError(f"Cannot perform {name} with type {self.dtype}")

        elif name in to_implement_yet:
            raise NotImplementedError

    def _from_factorized(self):
        # TODO: Will we actually implement this?
        raise NotImplementedError


@register_series_accessor("units")
class UnitsSeriesAccessor:
    def __init__(self, obj):
        # Inspired by fletcher
        if not isinstance(obj.values, UnitsExtensionArray):
            raise AttributeError("Only UnitsExtensionArray has units accessor")
        self.obj = obj

    @property
    def unit(self):
        return self.obj.values.unit

    def to(self, unit, equivalencies=None):
        """Convert series to another unit."""
        new_array = self.obj.values.to(unit, equivalencies)
        return self.obj.__class__(new_array)

    def to_si(self):
        """Convert series to another unit."""
        unit = self.obj.values.unit
        formatter = Generic()
        formatter._show_scale = False
        new_unit = Unit(formatter.to_string(unit.si))
        return self.to(new_unit)


@register_dataframe_accessor("units")
class UnitsDataFrameAccessor:
    def __init__(self, obj: pd.DataFrame):
        self.obj = obj

    def to_si(self):
        def _f(col):
            try:
                return col.units.to_si()
            except AttributeError:
                return col

        return self.obj.apply(_f)


UnitsExtensionArray._add_arithmetic_ops()
UnitsExtensionArray._add_comparison_ops()

UnitsExtensionArray.__pow__ = UnitsExtensionArray._create_method(operator.pow)
