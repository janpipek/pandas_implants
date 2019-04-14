"""Pandas extension types representing running pace expressed as minutes per km."""
from datetime import timedelta

import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype


class Pace:
    @classmethod
    def parse(cls, s: str) -> float:
        if not s:
            return np.nan
        elif s in ["-", "nan", np.nan]:
            return np.nan
        else:
            try:
                frags = s.split(":")
                frags = [float(frag) for frag in frags]
                if len(frags) == 1:
                    interval = timedelta(seconds=float(frags[0]))
                else:
                    interval = timedelta(hours=frags[-3] if len(frags) == 3 else 0, minutes=frags[-2], seconds=frags[-1])
                return interval.total_seconds()
            except ValueError:
                raise ValueError("Cannot parse as time: {0}".format(s))

    @classmethod
    def format(cls, f: float) -> str:
        return "X:XX"

    def __init__(self, value):
        if isinstance(value, str):
            self.value = self.parse(value)
        else:
            self.value = float(value)


@register_extension_dtype
class PaceType(ExtensionDtype):
    name = "pace"
    type = float
    kind = "f"
    na_value = np.nan

    @classmethod
    def construct_from_string(cls, string) -> 'PaceType':
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

    @classmethod
    def construct_array_type(cls) -> 'type':
        return PaceArray


class PaceArray(ExtensionArray):
    _can_hold_na = True

    def __init__(self, array, dtype=None, copy: bool = True):
        array = np.array(array) if copy else np.asarray(array)
        dtype = dtype if dtype else PaceType()

        if array.dtype.kind in ['f', 'i']:
            array = array.astype(dtype.type)
        elif array.dtype.kind in ['U', 'S']:
            array = np.array([Pace.parse(val) for val in array], dtype=dtype.type)
        elif array.dtype.kind == "m":
            array = array / np.timedelta64(1, 's')
        else:
            raise ValueError(f"Cannot create PaceArray from values of type '{dtype}'.")

        self._dtype = dtype
        self.data = array

    @property
    def dtype(self) -> ExtensionDtype:
        return self._dtype

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

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False) -> "PaceArray":
        result = cls(scalars)
        return result


