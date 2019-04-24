"""Pandas extension types representing running pace expressed as minutes per km."""
import builtins
from datetime import timedelta

import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype
from pandas.core.algorithms import take


class Pace:
    @classmethod
    def parse(cls, s: str) -> float:
        if not s:
            return np.nan
        elif s in ["-", "nan", np.nan]:
            return np.nan
        else:
            try:
                frags_str = s.split(":")
                frags = [float(frag) for frag in frags_str]
                if len(frags) == 1:
                    interval = timedelta(seconds=float(frags[0]))
                else:
                    interval = timedelta(hours=frags[-3] if len(frags) == 3 else 0, minutes=frags[-2], seconds=frags[-1])
                return interval.total_seconds()
            except ValueError:
                raise ValueError("Cannot parse as time: {0}".format(s))

    @classmethod
    def format(cls, f: float) -> str:
        if np.isnan(f):
            return "NaN"
        m, s = divmod(f, 60)
        return f"{int(m)}:{'0' if s < 10 else ''}{s}"

    def __init__(self, value):
        if isinstance(value, str):
            self.value = self.parse(value)
        else:
            self.value = float(value)


@register_extension_dtype
class PaceDtype(ExtensionDtype):
    name = "pace"
    type = float
    kind = "f"
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string: str) -> 'PaceDtype':
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

    @classmethod
    def construct_array_type(cls) -> builtins.type:
        return PaceArray

    def convert_array(self, array) -> np.ndarray:
        array = np.asarray(array)
        if array.dtype.kind in ['f', 'i']:
            return array.astype(self.type)
        elif array.dtype.kind in ['U', 'S']:
            return np.array([Pace.parse(val) for val in array], dtype=self.type)
        elif array.dtype.kind == "m":
            return array / np.timedelta64(1, 's')
        else:
            raise ValueError(f"Cannot create PaceDtype from values of type '{array.dtype}'.")

    def convert_scale(self, scalar) -> float:
        return Pace(scalar).value


class PaceArray(ExtensionArray):
    _can_hold_na = True

    def __init__(self, array, dtype=None, copy: bool = True):
        array = np.array(array) if copy else np.asarray(array)
        dtype = dtype if dtype else PaceDtype()
        if not isinstance(dtype, PaceDtype):
            raise ValueError(f"Invalid dtype for PaceArray: {dtype}")

        self._dtype = dtype
        self.data = dtype.convert_array(array)

    @property
    def dtype(self) -> PaceDtype:
        return self._dtype

    def astype(self, dtype, copy=True) -> np.ndarray:
        typename = dtype if isinstance(dtype, str) else getattr(dtype, "name", dtype.__class__.__name__)
        if self.dtype == dtype:
            # TODO: This looks strange
            return self.copy() if copy else self
        elif typename.startswith("timedelta64"):
            return np.asarray(self.data * 1e9, dtype="timedelta64[ns]").astype(dtype, copy=copy)
        else:
            return np.asarray(self).astype(dtype, copy=copy)

    @property
    def nbytes(self):
        return self.data.nbytes

    def isna(self):
        return np.isnan(self.data)

    def __array__(self, dtype=None) -> np.ndarray:
        return self.data.astype(dtype)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.data[item]
        else:
            return self.__class__(self.data[item])

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            self.data[key] = self.dtype.convert_scalar(value)
        else:
            self.data[key] = self.dtype.convert_array(value)

    def take(self, indices, allow_fill=False, fill_value=None):
        data = self.astype(object)

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(self.data, indices, fill_value=fill_value,
                      allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def copy(self, deep=False) -> "PaceArray":
        return PaceArray(self.data.copy())

    def _formatter(self, boxed=False):
        return lambda x: Pace.format(x)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False) -> "PaceArray":
        result = cls(scalars)
        return result


