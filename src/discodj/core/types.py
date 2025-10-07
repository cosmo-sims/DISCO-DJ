from typing import Union
import jax
import numpy as onp

__all__ = ['AnyArray']

# Define array type (Jax or Numpy)
AnyArray = Union[jax.Array, onp.ndarray]
