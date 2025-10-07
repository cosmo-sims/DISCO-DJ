from ._discodj_native import *
from ._discodj_native import __doc__

import sys


# set docstrings of this python module of the _discodj_native pybind11 module
_module = sys.modules[__name__]
for name in dir(_module):
    obj = getattr(_module, name)
    try:
        obj.__module__ = __name__
    except (AttributeError, TypeError):
        # not all objects allow __module__ to be set
        pass
