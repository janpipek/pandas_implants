import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype


# Example:
# >>> favourite_colours = np.asarray([(255, 255, 0), (255, 0, 0)], dtype=color.dtypes["rgb"])
dtypes = {
    'rgb': np.dtype([('r', np.uint8), ('g', np.uint8), ('b', np.uint8)]),
    'rgba': np.dtype([('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('a', np.float32)])
}


@register_extension_dtype
class RgbType(ExtensionDtype):
    name = 'rgb'
    na_value = np.nan
    type = str
    kind = 'S7'

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls}' from '{string}'")


class RgbArray(ExtensionArray):
    dtype = ...
    