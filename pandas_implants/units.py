import builtins
import re

from astropy.units import Quantity, Unit
import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype


@register_extension_dtype
class UnitsDtype(ExtensionDtype):
    BASE_NAME = "unit"

    type = Quantity

    def __init__(self, *args, **kwargs):
        self.unit = Unit(*args, **kwargs)

    @classmethod
    def construct_from_string(cls, string):
        # TODO: extract unit name

    @classmethod
    def construct_array_type(cls) -> builtins.type:
        return UnitsExtensionArray

    @property
    def name(self):
        return f"{self.BASE_NAME}[{self.unit}]"


class UnitsExtensionArray(ExtensionArray):
    pass

