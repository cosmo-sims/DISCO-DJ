import jax.numpy as jnp
import numpy as onp
from itertools import product
from ..core.utils import set_indices_to_val
from ..core.types import AnyArray

__all__ = ['get_lagrangian_grid_vectors', 'get_fourier_grid', 'enforce_hermitian_symmetry']


def get_lagrangian_grid_vectors(dim: int, res: int, boxsize: float, with_jax: bool = True) -> list:
    """Get the Lagrangian grid vectors (qx, qy, qz, ...) in a list.

    :param dim: dimension
    :param res: linear resolution, i.e. resolution in each dimension
    :param boxsize: boxsize in Mpc/h
    :param with_jax: use Jax?
    :return: list of Lagrangian grid vectors [qx, qy, qz, ...]
    """
    np = jnp if with_jax else onp
    dx = boxsize / res
    q_vec = np.arange(res) * dx
    qs = []
    for i in range(dim):
        exp_list = tuple([None] * i + [slice(None)] + [None] * (dim - i - 1))  # expand with other dims (length 1)
        qs.append(q_vec[exp_list])
    return qs


def get_fourier_grid(shape: tuple | list, boxsize: float = 1.0, sparse_k_vecs: bool = True, full: bool = False,
                     relative: bool = False, dtype_num: int = 64, with_jax: bool = True) -> dict:
    """Get a dictionary with the Fourier mesh k, the mesh vectors kx, ky, kz, ..., and kx, ky, kz at each mesh point.

    :param shape: shape of the grid, e.g. (256, 256, 256)
    :param boxsize: boxsize in Mpc/h
    :param sparse_k_vecs: instead of returning a mesh, return a tensor whose length is 1 along dimensions
        where k is const.
    :param full: use full complex FFT layout (np.fft.fft) instead of real FFT layout (np.fft.rfft)
    :param relative: if True, return k in units of the Nyquist mode instead of in units of h/Mpc
    :param dtype_num: 32 or 64 (float32 or float64)
    :param with_jax: use Jax?
    :return: dictionary with abs(k) (modulus at each mesh point), k_vecs (kx, ky, kz, ...) at each mesh point,
        potentially sparse i.e. with length 1 along dimensions where k is const.
    """
    np = jnp if with_jax else onp
    dtype = np.float64 if dtype_num == 64 else np.float32

    dim = len(shape)
    dk = 2 * np.pi
    k_vecs = []

    for i in range(dim):
        scale_fac_i = shape[i] if relative else boxsize
        if i < dim - 1 or full:
            k_vec = np.fft.fftshift(np.arange(-shape[i] // 2, shape[i] // 2, dtype=dtype) * dk / scale_fac_i)
            k_vecs.append(k_vec.astype(dtype))
        # last dimension: only half the modes are included for real FFT
        elif i == dim - 1 and not full:
            k_vecs.append(np.arange(0, shape[i] // 2 + 1, dtype=dtype) * dk / scale_fac_i)

    # Make a meshgrid
    k_vec_mesh = np.meshgrid(*k_vecs, indexing='ij')

    # Concatenate along (new) 0th dimension and sum to get k square
    k_sq = np.concatenate(tuple([kk[None] ** 2 for kk in k_vec_mesh]), 0).sum(0)
    k = np.sqrt(k_sq)

    # If desired: return sparse layout for k_vecs
    if sparse_k_vecs:
        k_vec_mesh = []
        for i in range(dim):
            exp_list = tuple([None] * i + [slice(None)] + [None] * (dim - i - 1))  # expand with other dims (length 1)
            k_vec_mesh.append(k_vecs[i][exp_list])

    # Return all the ks in each dimension, as well as k_vec, and |k|
    k_dict = {"|k|": k, "k_vecs": k_vec_mesh}
    return k_dict


def _enforce_hermitian_symmetry_1d(x: AnyArray, assert_eps: float | None = None, with_jax: bool = True) -> AnyArray:
    """Enforce Hermitian symmetry on a 1D array.
    """
    np = jnp if with_jax else onp
    assert len(x.shape) == 1
    res = (x.shape[0] - 1) * 2
    make_real = lambda x_: np.real(x_)

    x = set_indices_to_val(x, 0, make_real(x[0]))
    x = set_indices_to_val(x, res // 2, make_real(x[res // 2]))

    if assert_eps is not None:
        assert np.all(np.abs(np.fft.rfft(np.fft.irfft(x)) - x) < assert_eps), \
            'The array x does not have the required Hermitian symmetry!'
    return x


def _enforce_hermitian_symmetry_2d(x: AnyArray, assert_eps: float | None = None, with_jax: bool = True) -> AnyArray:
    """Enforce Hermitian symmetry on a 2D array.
    """
    np = jnp if with_jax else onp
    assert len(x.shape) == 2
    assert x.shape[1] == x.shape[0] // 2 + 1, "2D Fourier layout seems to be incorrect!"
    res = x.shape[0]
    make_real = lambda x_: np.real(x_)

    x = set_indices_to_val(x, (0, 0), make_real(x[0, 0]))
    x = set_indices_to_val(x, (0, res // 2), make_real(x[0, res // 2]))
    x = set_indices_to_val(x, (res // 2, 0), make_real(x[res // 2, 0]))
    x = set_indices_to_val(x, (res // 2, res // 2), make_real(x[res // 2, res // 2]))

    x = set_indices_to_val(x, (slice(res // 2 + 1, None), 0), x[1:res // 2, 0].conj()[::-1])
    x = set_indices_to_val(x, (slice(res // 2 + 1, None), res // 2), x[1:res // 2, res // 2].conj()[::-1])

    if assert_eps is not None:
        assert np.all(np.abs(np.fft.rfft2(np.fft.irfft2(x)) - x) < assert_eps), \
            'The array x does not have the required Hermitian symmetry!'
    return x


def _enforce_hermitian_symmetry_3d(x: AnyArray, assert_eps: float | None = None, with_jax: bool = True) -> AnyArray:
    """Enforce Hermitian symmetry on a 3D array.
    """
    np = jnp if with_jax else onp
    assert len(x.shape) == 3
    assert (x.shape[1] == x.shape[0]) and (x.shape[2] == x.shape[0] // 2 + 1), \
        "3D Fourier layout seems to be incorrect!"
    res = x.shape[0]
    make_real = lambda x_: np.real(x_)
    real_combs = list(product(*list(zip([0] * 3, [res // 2] * 3))))
    real_comb_inds = tuple([*list(zip(*real_combs))])  # reorder: first x, then y, ...

    x = set_indices_to_val(x, real_comb_inds, make_real(x[real_comb_inds]).astype(np.complex64))

    x = set_indices_to_val(x, (slice(res // 2 + 1, None), slice(1, None), 0),
                           jnp.conj(jnp.fliplr(jnp.flipud(x[1:res // 2, 1:, 0]))))
    x = set_indices_to_val(x, (slice(res // 2 + 1, None), 0, 0), jnp.conj(x[res // 2 - 1:0:-1, 0, 0]))
    x = set_indices_to_val(x, (0, slice(res // 2 + 1, None), 0), jnp.conj(x[0, res // 2 - 1:0:-1, 0]))
    x = set_indices_to_val(x, (res // 2, slice(res // 2 + 1, None), 0), jnp.conj(x[res // 2, res // 2 - 1:0:-1, 0]))

    x = set_indices_to_val(x, (slice(res // 2 + 1, None), slice(1, None), res // 2),
                           jnp.conj(jnp.fliplr(jnp.flipud(x[1:res // 2, 1:, res // 2]))))
    x = set_indices_to_val(x, (slice(res // 2 + 1, None), 0, res // 2), jnp.conj(x[res // 2 - 1:0:-1, 0, res // 2]))
    x = set_indices_to_val(x, (0, slice(res // 2 + 1, None), res // 2), jnp.conj(x[0, res // 2 - 1:0:-1, res // 2]))
    x = set_indices_to_val(x, (res // 2, slice(res // 2 + 1, None), res // 2),
                           jnp.conj(x[res // 2, res // 2 - 1:0:-1, res // 2]))

    if assert_eps is not None:
        assert np.all(np.abs(np.fft.rfftn(np.fft.irfftn(x)) - x) < assert_eps), \
            'The array x does not have the required Hermitian symmetry!'
    return x


def enforce_hermitian_symmetry(x: AnyArray, assert_eps: float | None = None, with_jax: bool = True) -> AnyArray:
    """Enforce Hermitian symmetry on a 1D, 2D, or 3D array.

    :param x: input array
    :param assert_eps: if not None: assert that the array has the required symmetry up to this tolerance
    :param with_jax: use Jax?
    :return: x with enforced symmetry
    """
    dim = len(x.shape)
    if dim == 1:
        return _enforce_hermitian_symmetry_1d(x, assert_eps=assert_eps, with_jax=with_jax)
    elif dim == 2:
        return _enforce_hermitian_symmetry_2d(x, assert_eps=assert_eps, with_jax=with_jax)
    elif dim == 3:
        return _enforce_hermitian_symmetry_3d(x, assert_eps=assert_eps, with_jax=with_jax)
    else:
        raise NotImplementedError
