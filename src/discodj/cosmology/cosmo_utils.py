from jax import Array
import jax.numpy as jnp
import numpy as onp
from ..core.types import AnyArray

__all__ = ['get_sigma8_squared_from_Pk']


def get_sigma8_squared_from_Pk(k: Array, Pk: Array, R: float = 8.0, with_jax: bool = True) -> AnyArray:
    """Compute sigma8^2 from the power spectrum.
    NOTE: make sure that the k-range is large enough to reasonably cover the whole power spectrum!

    :param k: wavenumber in h / Mpc
    :param Pk: power spectrum in Mpc^3 / h^3 at those wavenumbers
    :param R: radius in Mpc / h. Default is 8.0
    :param with_jax: use Jax?
    :return: sigma8^2
    """
    np = jnp if with_jax else onp

    # Dimensionless argument of the window function
    x = k * R

    # Compute the tophat window function W(kR)
    W = np.where(x != 0, 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3, 0.0)

    # Compute integrand: k^2 * P(k) * W(kR)^2 dk = k^3 * P(k) * W(kR)^2 d(log(k))
    integrand = k ** 3 * Pk * W ** 2

    # Perform integration over log(k)
    log_k = np.log(k)
    sigma8_squared = np.trapezoid(integrand, log_k)

    # Normalize by the prefactor
    normalization = 1.0 / (2.0 * np.pi ** 2)
    return sigma8_squared * normalization
