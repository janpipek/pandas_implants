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
    type = np.float
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
    def construct_array_type(cls) -> type:
        return PaceArray


class PaceArray(ExtensionArray):
    _dtype = PaceType

    # def __init__(self, values):
    #     pass
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self._data)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        result = PaceArray()
        data = np.array(
            [cls.parse_single(scalar) for scalar in scalars],
            dtype=np.float64
        )
        result._dtype = dtype
        result._data = data
        return result

    @classmethod
    def parse_single(cls, s) -> float:
        if not s:
            return None
        elif s in ["-", "nan", np.nan]:
            return None
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
