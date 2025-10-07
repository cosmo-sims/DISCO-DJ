import numpy as onp
import jax.numpy as jnp
from jax import Array
from functools import reduce
from ..core.utils import set_indices_to_val


__all__ = ['gradient_kernel', 'inv_laplace_kernel', 'inv_mak_kernel']


def gradient_kernel(k_vecs: list, axis: int, order: int = 0, with_jax: bool = True) -> Array:
    """Get the gradient kernel along the specified axis.

    :param k_vecs: list of k-space arrays
    :param axis: direction of the gradient (0, 1, or 2)
    :param order: order of the kernel (0: ik, 2/4/6 for finite difference kernels)
    :param with_jax: whether to use Jax or Numpy
    :return: gradient kernel
    """
    np = jnp if with_jax else onp
    dim = len(k_vecs)

    # order 0: exact Fourier kernel
    if order == 0:
        kernel = (1j * k_vecs[axis]).squeeze()
        # Zero Nyquist frequency in the direction of the gradient
        if axis == dim - 1:  # half the length, Nyquist sits at the end
            kernel = set_indices_to_val(kernel, -1, 0)
        else:
            kernel = set_indices_to_val(kernel, len(kernel) // 2, 0)  # full length, Nyquist sits in the middle
        kernel = kernel.reshape(k_vecs[axis].shape)

    # finite-difference kernels
    else:
        k = k_vecs[axis]
        max_k_over_pi = np.abs(k).max() / np.pi  # rescale: note that input k_vecs has units!
        if order == 2:
            kernel = 1.0 * (np.sin(k / max_k_over_pi))
        elif order == 4:
            kernel = 1.0 / 6.0 * (8 * np.sin(k / max_k_over_pi) - np.sin(2 * k / max_k_over_pi))
        elif order == 6:
            kernel = (1.0 / 30.0 * (45.0 * np.sin(k / max_k_over_pi)
                                        - 9.0 * np.sin(2 * k / max_k_over_pi) + np.sin(3 * k / max_k_over_pi)))
        else:
            raise NotImplementedError
        kernel *= 1j * max_k_over_pi

    return kernel


def inv_laplace_kernel(k_vecs: list, order: int = 0, with_jax: bool = True) -> Array:
    """Get the inverse Laplace kernel from a given list of k vectors.

    :param k_vecs: list of k-space arrays
    :param order: order of the kernel (0: -1/k^2, 2/4/6 for finite difference kernels)
    :param with_jax: whether to use Jax or Numpy
    :return: inverse Laplace kernel
    """
    np = jnp if with_jax else onp

    # order 0: exact Fourier kernel
    if order == 0:
        ksquare = sum(ki ** 2 for ki in k_vecs)
        ksquare = set_indices_to_val(ksquare, (0,) * len(k_vecs), 1.0)  # DC mode
        kernel = - 1. / ksquare
        mask = (~(ksquare == 0)).astype(int)
        kernel *= mask

    # finite-difference kernels
    else:
        max_k_over_pi_all = np.asarray([np.abs(k_1d).max() / np.pi for k_1d in k_vecs])

        if order == 2:
            kernel_inv = sum([-2.0 * (-np.cos(k_1d / max_k_over_pi_1d) + 1.0)
                              * max_k_over_pi_1d ** 2
                              for k_1d, max_k_over_pi_1d in zip(k_vecs, max_k_over_pi_all)])
        elif order == 4:
            kernel_inv = sum([-1.0 / 6.0 * (np.cos(2 * k_1d / max_k_over_pi_1d)
                                            - 16.0 * np.cos(k_1d / max_k_over_pi_1d) + 15.0)
                              * max_k_over_pi_1d ** 2
                              for k_1d, max_k_over_pi_1d in zip(k_vecs, max_k_over_pi_all)])
        elif order == 6:
            kernel_inv = sum([-1.0 / 90.0 * (-2.0 * np.cos(3 * k_1d / max_k_over_pi_1d)
                                             + 27.0 * np.cos(2 * k_1d / max_k_over_pi_1d)
                                             - 270.0 * np.cos(k_1d / max_k_over_pi_1d) + 245.0)
                              * max_k_over_pi_1d ** 2
                              for k_1d, max_k_over_pi_1d in zip(k_vecs, max_k_over_pi_all)])
        else:
            raise NotImplementedError

        kernel_inv = set_indices_to_val(kernel_inv, (0,) * len(k_vecs), -1.)
        kernel = 1.0 / kernel_inv
    return kernel


def inv_mak_kernel(k_vecs: list, account_for_shotnoise: bool = False, double_exponent: bool = False,
                   worder: int = 2, with_jax: bool = True) -> Array:
    """Get inverse mass assignment kernel.
    Based on Jing et al 2005 (https://arxiv.org/abs/astro-ph/0409240).

    :param k_vecs: list of k-space arrays
    :param account_for_shotnoise: account for shotnoise?
    :param double_exponent: use double exponent?
        This should be used when scattering onto mesh and gathering field at particle positions afterwards.
    :param worder: order of the mass assignment kernel (2: CIC, 3: TSC, 4: PCS)
    :param with_jax: whether to use Jax or Numpy
    :return: inverse mass assignment kernel
    """
    np = jnp if with_jax else onp

    # Rescale: note that input k_vecs has units!
    max_k_over_pi_all = [np.abs(kk).max() / np.pi for kk in k_vecs]

    if account_for_shotnoise:
        sin_terms = [np.sin(k_vecs[i] / (2.0 * max_k_over_pi_all[i])) ** 2 for i in range(len(k_vecs))]
        if worder == 2:
            kernel_per_dim = [(1 - 2.0 / 3 * s) ** 0.5 for s in sin_terms]
        elif worder == 3:
            kernel_per_dim = [(1 - s + 2.0 / 15 * s ** 2) ** 0.5 for s in sin_terms]
        elif worder == 4:
            kernel_per_dim = [(1 - 4.0 / 3 * s + 2.0 / 5 * s ** 2 - 4.0 / 315 * s ** 3) ** 0.5 for s in sin_terms]
            # see https://github.com/bccp/nbodykit/blob/a387cf429d8cb4a07bb19e3b4325ffdf279a131e/nbodykit/source/mesh/catalog.py#L547
        else:
            raise NotImplementedError
        exponent = -2 if double_exponent else -1

    else:
        kernel_per_dim = [np.sinc(k_vecs[i] / (2 * np.pi) / max_k_over_pi_all[i]) for i in range(len(k_vecs))]
        exponent = -1 * worder
        if double_exponent:
            exponent *= 2

    kernel = reduce(lambda x, y: x * y, kernel_per_dim) ** exponent
    return kernel
