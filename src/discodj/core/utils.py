import numpy as onp
import jax
import jaxlib
import jax.numpy as jnp
from collections.abc import Callable
from ..core.types import AnyArray
from jax import Array

__all__ = ['is_jax_object', 'scan_fct_np', 'set_indices_to_val', 'add_to_indices', 'set_0_to_val',
           'set_row_or_col_to_val', 'get_indices_for_pad_and_crop', 'crop', 'pad', 'conv2_fourier',
           'preprocess_fixed_inds']


def is_jax_object(t: object) -> bool:
    """Helper function that checks if t is a Jax object.

    :param t: object to check
    :return: boolean indicating whether t is a Jax object
    """
    # from version 0.4.1 on: jax.Array type https://jax.readthedocs.io/en/latest/jax_array_migration.html
    if int(jax.__version__[0]) > 0 or int(jax.__version__[2]) >= 4:
        return isinstance(t, jax.Array)
    else:
        # Legacy
        return isinstance(t, (jax.xla.DeviceArray, jax.interpreters.partial_eval.DynamicJaxprTracer,
                              jax.interpreters.batching.BatchTracer, jaxlib.xla_extension.ArrayImpl,
                              jax.interpreters.ad.JVPTracer))


def scan_fct_np(f: Callable, init: object, xs: AnyArray | list | tuple, length: int | None = None,
                multiple_xs: bool = False) -> tuple[object, onp.ndarray]:
    """Helper function that mimics jax.lax.scan for numpy,
    see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html.

    :param f: function to apply
    :param init: initial value for the carry
    :param xs: the values to scan over.
        This can also be a list/tuple of arrays, then the scan runs over zipped values.
    :param length: optional integer specifying the number of loop iterations
    :param multiple_xs: if True: xs is a list/tuple of arrays, otherwise xs is a single array
    :return: final carry, stacked array of ys
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    iter_vals = zip(*xs) if multiple_xs else xs
    for x in iter_vals:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, onp.stack(ys)


def set_indices_to_val(t: AnyArray, inds: AnyArray | tuple | slice | int, v: AnyArray | float) -> AnyArray:
    """Helper function that sets t[inds] = v  for numpy / jax.numpy.

    :param t: tensor
    :param inds: indices to set
    :param v: value to set
    :return: output tensor
    """
    if isinstance(t, onp.ndarray):
        t[inds] = v
    elif is_jax_object(t):
        t = t.at[inds].set(v)
    else:
        print(f"Type {type(t)} not supported!")
        raise NotImplementedError
    return t


def add_to_indices(t: AnyArray, inds: AnyArray | tuple | slice | int, v: AnyArray | float) -> AnyArray:
    """Helper function that adds v to t[inds] for numpy / jax.numpy.

    :param t: tensor
    :param inds: indices where to add
    :param v: value to add
    :return: output tensor
    """
    if isinstance(t, onp.ndarray):
        t[inds] += v
    elif is_jax_object(t):
        t = t.at[inds].add(v)
    else:
        print(f"Type {type(t)} not supported!")
        raise NotImplementedError
    return t


def set_0_to_val(dim: int, t: AnyArray, v: float) -> AnyArray:
    """Helper function that sets t[(0,) * dim] = v.

    :param dim: dimension
    :param t: tensor
    :param v: value to set
    :return: output tensor
    """
    inds = (0,) * dim
    return set_indices_to_val(t, inds, v)


def set_row_or_col_to_val(dim: int, t: AnyArray, ind: int, v: float, axis: int = 0) -> AnyArray:
    """Helper function that sets t[..., ind, ...] = v

    :param dim: dimension
    :param t: tensor
    :param ind: index of row, column, etc.
    :param v: value to set
    :param axis: axis for ind
    :return: output tensor
    """
    inds = tuple([slice(None)] * axis + [ind] + [slice(None)] * (dim - axis - 1))
    if isinstance(t, onp.ndarray):
        t[inds] = v
    elif is_jax_object(t):
        t = t.at[inds].set(v)
    else:
        print(f"Type {type(t)} not supported!")
        raise NotImplementedError
    return t


def get_indices_for_pad_and_crop(orig_res: int, ext_res: int) -> tuple:
    """Get indices for cropping and padding.
    First 4 indices (ip0, ip1, ip2, ip3): start and end of first and second block w.r.t. the extended resolution, i.e.
    ip0:ip1, ip2:ip3 should be read in the extended field
    Last 4 indices (i0, i1, i2, i3): start and end of first and second block w.r.t. the original resolution, i.e.
    i0:i1, i2:i3 should be set in the original field.

    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :return: ip0, ip1, ip2, ip3, i0, i1, i2, i3
    """
    return (0, orig_res // 2, ext_res - orig_res // 2 + 1, ext_res,
            0, orig_res // 2, orig_res // 2 + 1, orig_res)


def crop(dim: int, orig_res: int, ext_res: int, ff: AnyArray, dtype_c: type, full: bool = False, with_jax: bool = True)\
        -> AnyArray:
    """Crop a field in Fourier space.

    :param dim: number of dimensions
    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :param ff: field in Fourier space
    :param dtype_c: complex dtype to use
    :param full: if True: full Fourier layout, else: real Fourier layout
    :param with_jax: use Jax?
    :return: cropped field in Fourier space
    """
    np = jnp if with_jax else onp

    if ext_res == orig_res:
        return ff

    ip0, ip1, ip2, ip3, i0, i1, i2, i3 = get_indices_for_pad_and_crop(orig_res, ext_res)

    if dim == 1:
        new_shape = [orig_res] if full else [orig_res // 2 + 1]   # TODO!
        if full:
            ff_cropped = np.zeros(new_shape, dtype=dtype_c)

            ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1),), ff[ip0:ip1])
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3),), ff[ip2:ip3])
        else:
            ff_cropped = ff[ip0:ip1]

    elif dim == 2:
        new_shape = [orig_res] * 2 if full else [orig_res, orig_res // 2 + 1]
        ff_cropped = np.zeros(new_shape, dtype=dtype_c)

        ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1), slice(i0, i1)), ff[ip0:ip1, ip0:ip1])
        ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3), slice(i0, i1)), ff[ip2:ip3, ip0:ip1])

        if full:
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1), slice(i2, i3)), ff[ip0:ip1, ip2:ip3])
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3), slice(i2, i3)), ff[ip2:ip3, ip2:ip3])

    elif dim == 3:
        new_shape = [orig_res] * 3 if full else [orig_res] * 2 + [orig_res // 2 + 1]
        ff_cropped = np.zeros(new_shape, dtype=dtype_c)

        ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1), slice(i0, i1), slice(i0, i1)),
                                        ff[ip0:ip1, ip0:ip1, ip0:ip1])
        ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1), slice(i2, i3), slice(i0, i1)),
                                        ff[ip0:ip1, ip2:ip3, ip0:ip1])
        ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3), slice(i0, i1), slice(i0, i1)),
                                        ff[ip2:ip3, ip0:ip1, ip0:ip1])
        ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3), slice(i2, i3), slice(i0, i1)),
                                        ff[ip2:ip3, ip2:ip3, ip0:ip1])

        if full:
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1), slice(i0, i1), slice(i2, i3)),
                                            ff[ip0:ip1, ip0:ip1, ip2:ip3])
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i0, i1), slice(i2, i3), slice(i2, i3)),
                                            ff[ip0:ip1, ip2:ip3, ip2:ip3])
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3), slice(i0, i1), slice(i2, i3)),
                                            ff[ip2:ip3, ip0:ip1, ip2:ip3])
            ff_cropped = set_indices_to_val(ff_cropped, (slice(i2, i3), slice(i2, i3), slice(i2, i3)),
                                            ff[ip2:ip3, ip2:ip3, ip2:ip3])

    else:
        raise NotImplementedError

    return ff_cropped * (orig_res / ext_res) ** dim


def pad(dim: int, orig_res: int, ext_res: int, ff: AnyArray, dtype_c: type, full: bool = False, with_jax: bool = True) \
        -> AnyArray:
    """Pad a field in Fourier space, setting higher modes to zero.

    :param dim: number of dimensions
    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :param ff: field in Fourier space
    :param dtype_c: complex dtype to use
    :param full: if True: full Fourier layout, else: real Fourier layout
    :param with_jax: use Jax?
    :return: padded field in Fourier space
    """
    np = jnp if with_jax else onp

    # ip: indices into original tensor, i: indices into extended tensor
    ip0, ip1, ip2, ip3, i0, i1, i2, i3 = get_indices_for_pad_and_crop(orig_res, ext_res)

    if dim == 1:
        new_shape = [ext_res] if full else [ext_res // 2 + 1]
        if full:
            padff = np.zeros(new_shape, dtype=dtype_c)

            padff = set_indices_to_val(padff, (slice(ip0, ip1),), ff[i0:i1])
            padff = set_indices_to_val(padff, (slice(ip2, ip3),), ff[i2:i3])
        else:
            padff = np.zeros(new_shape, dtype=dtype_c)
            padff = set_indices_to_val(padff, (slice(ip0, ip1),), ff[i0:i1])

    elif dim == 2:
        new_shape = [ext_res, ext_res // 2 + 1]
        padff = np.zeros(new_shape, dtype=dtype_c)

        padff = set_indices_to_val(padff, (slice(ip0, ip1), slice(ip0, ip1)), ff[i0:i1, i0:i1])
        padff = set_indices_to_val(padff, (slice(ip2, ip3), slice(ip0, ip1)), ff[i2:i3, i0:i1])

        if full:
            padff = set_indices_to_val(padff, (slice(ip0, ip1), slice(i2, i3)), ff[i0:i1, i2:i3])
            padff = set_indices_to_val(padff, (slice(ip2, ip3), slice(i2, i3)), ff[i2:i3, i2:i3])

        padff = set_0_to_val(2, padff, 0)

    elif dim == 3:
        new_shape = [ext_res] * 3 if full else [ext_res] * 2 + [ext_res // 2 + 1]
        padff = np.zeros(new_shape, dtype=dtype_c)

        padff = set_indices_to_val(padff, (slice(ip0, ip1), slice(ip0, ip1), slice(ip0, ip1)), ff[i0:i1, i0:i1, i0:i1])
        padff = set_indices_to_val(padff, (slice(ip0, ip1), slice(ip2, ip3), slice(ip0, ip1)), ff[i0:i1, i2:i3, i0:i1])
        padff = set_indices_to_val(padff, (slice(ip2, ip3), slice(ip0, ip1), slice(ip0, ip1)), ff[i2:i3, i0:i1, i0:i1])
        padff = set_indices_to_val(padff, (slice(ip2, ip3), slice(ip2, ip3), slice(ip0, ip1)), ff[i2:i3, i2:i3, i0:i1])

        if full:
            padff = set_indices_to_val(padff, (slice(ip0, ip1), slice(ip0, ip1), slice(i2, i3)),
                                       ff[i0:i1, i0:i1, i2:i3])
            padff = set_indices_to_val(padff, (slice(ip0, ip1), slice(ip2, ip3), slice(i2, i3)),
                                       ff[i0:i1, i2:i3, i2:i3])
            padff = set_indices_to_val(padff, (slice(ip2, ip3), slice(ip0, ip1), slice(i2, i3)),
                                       ff[i2:i3, i0:i1, i2:i3])
            padff = set_indices_to_val(padff, (slice(ip2, ip3), slice(ip2, ip3), slice(i2, i3)),
                                       ff[i2:i3, i2:i3, i2:i3])

        padff = set_0_to_val(3, padff, 0)

    else:
        raise NotImplementedError

    return padff * (ext_res / orig_res) ** dim


def conv2_fourier(dim: int, ff1: AnyArray, ff2: AnyArray, orig_res: int | None, ext_res: int | None, dtype_c: type,
                  full: bool = False, with_jax: bool = True, do_crop: bool = True) -> AnyArray:
    """Convolution of two fields in Fourier space.

    :param dim: number of dimensions
    :param ff1: field1 in Fourier space
    :param ff2: field2 in Fourier space
    :param dtype_c: complex dtype to use
    :param orig_res: original resolution per dimension, only needed if do_crop is True
    :param ext_res: extended resolution per dimension, only needed if do_crop is True
    :param full: if True: full Fourier layout, else: real Fourier layout
    :param with_jax: use Jax?
    :param do_crop: if True: crop the convolution to the original resolution
    :return: convolution of ff1 and ff2 in Fourier space
    """
    np = jnp if with_jax else onp
    fft = np.fft.fftn if full else np.fft.rfftn
    ifft = np.fft.ifftn if full else np.fft.irfftn
    if do_crop:
        return crop(dim, orig_res, ext_res, fft(ifft(ff1) * ifft(ff2)), dtype_c=dtype_c, full=full, with_jax=with_jax)
    else:
        return fft(ifft(ff1) * ifft(ff2))


def preprocess_fixed_inds(dim: int, f: tuple | list | None) -> tuple:
    """Utility function to preprocess indices when slice or skewer is requested.
    Returns the indices that shall be extracted in each dimension and a list
    indicating whether the input was an integer before or not (those dimensions can then be
    squeezed by the calling function).    
    
    :param dim: number of dimensions
    :param f: list of indices, one per dimension. If the index is None, all indices along the dimension are
        used. If the index is an integer, it indicates an infinitely thin slice along the dimension. 
        If the index is a 2-tuple, the first element is the starting index and the second element is
        the ending index of the thick slice.
    :return: fixed indices and list indicating whether the input was an integer before
    """
    was_integer_before_loc = []
    if f is None:
        return None, None
    else:
        f = list(f)  # convert to list so it's mutable
        assert len(f) == dim, "fixed_inds must be a list of length dim or None."
        for d, f_d in enumerate(f):
            if isinstance(f_d, int):
                f[d] = (f_d, f_d + 1)
                was_integer_before_loc.append(True)
            elif isinstance(f_d, tuple):
                assert len(f_d) == 2, "Each element of fixed_inds must be either None, an integer or a 2-tuple."
                was_integer_before_loc.append(False)
            else:
                assert f_d is None, "Each element of fixed_inds must be either None, an integer or a 2-tuple."
                was_integer_before_loc.append(False)
        return tuple(f), was_integer_before_loc

# 20th order Gauss-Legendre quadrature weights and abscissas
def get_gaussian_quadrature_weights_and_abscissas() -> tuple:
    abscissas = (0.0765265211334973, 0.2277858511416451, 0.3737060887154195, 0.5108670019508271, 0.6360536807265150,
                 0.7463319064601508, 0.8391169718222188, 0.9122344282513259, 0.9639719272779138, 0.9931285991850949)
    weights = (0.1527533871307258, 0.1491729864726037, 0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
               0.1019301198172404, 0.0832767415767048, 0.0626720483341091, 0.0406014298003869, 0.0176140071391521)
    abscissas = onp.asarray(abscissas)
    weights = onp.asarray(weights)
    abscissas = onp.concatenate((-abscissas[::-1], abscissas))
    weights = onp.concatenate((weights[::-1], weights))
    return abscissas, weights


def rk4_integrate(func, y0, t_array) -> Array:
    """Integrates using RK4 method with jax.lax.scan for variable time steps.
    
    :param func: Function representing the ODE (dy/dt = func(t, y)).
    :param y0: Initial value of y.
    :param t_array: Array of time steps where integration is evaluated.
    :return: Array of y values corresponding to t_array.
    """

    def rk4_step(carry, t_next):
        """Performs a single RK4 step."""
        y, t = carry
        dt = t_next - t
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt / 2 * k1)
        k3 = func(t + dt / 2, y + dt / 2 * k2)
        k4 = func(t + dt, y + dt * k3)
        y_out = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y_out, t_next), y_out

    # Perform scan
    _, y_values = jax.lax.scan(rk4_step, (y0, t_array[0]), t_array[1:])
    return jnp.concatenate([y0[None, :], y_values])  # Include initial condition
