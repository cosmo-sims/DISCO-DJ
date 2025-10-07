import jax.numpy as jnp
from jax import Array
from ..cosmology.cosmology import Cosmology
from ..core.types import AnyArray

__all__ = ["generate_ics_plane_wave"]


def generate_ics_plane_wave(a: float | Array, q: AnyArray, cosmo: Cosmology, boxsize: float,
                            a_cross: float = 1.0, A_dec: float = 0.0, n_mode: int = 1, from_potential: bool = False) \
        -> dict:
    """Generate plane wave initial conditions.

    :param a: scale factor at which to evaluate the plane wave
    :param q: Lagrangian positions
    :param cosmo: cosmology object
    :param boxsize: box size in Mpc / h
    :param a_cross: shell-crossing scale factor, defaults to 1.0
    :param A_dec: decaying mode amplitude, defaults to 0.0
    :param n_mode: mode to use for the plane wave, defaults to 1
    :param from_potential: whether to generate the plane wave from the potential, defaults to False.
        Specifically, if this is True, this function only sets the potential, and the caller must
        then compute the initial displacement field from the potential with (1)LPT.
    :return: dictionary with the plane wave initial conditions
    """
    dim = q.shape[-1]
    dtype = q.dtype


    if from_potential:
        assert A_dec == 0, "'from_potential' requires A_dec to be 0!"

    D_cross = cosmo.Dplus(jnp.array(a_cross))

    def phi_plus():
        return (cosmo.Dplus(a)
                * jnp.cos(n_mode * 2.0 * jnp.pi * q[..., 0] / boxsize)
                / (n_mode * 2 * jnp.pi) ** 2 * boxsize ** 2 / D_cross)

    def phi_minus():
        return (-cosmo.E(a)  # Dminus is proportional to E(a)
                * jnp.sin(n_mode * 2.0 * jnp.pi * q[..., 0] / boxsize)
                / (n_mode * 2 * jnp.pi) ** 2 * boxsize ** 2 * A_dec)

    def delta_plus():
        return -cosmo.Dplus(a) \
            * jnp.cos(n_mode * 2.0 * jnp.pi * q[..., 0] / boxsize) / D_cross

    def delta_minus():
        return cosmo.E(a) \
            * jnp.sin(n_mode * 2.0 * jnp.pi * q[..., 0] / boxsize) * A_dec

    def psi_plus_no_prefac():
        psi_raw = (jnp.sin(n_mode * 2.0 * jnp.pi * q[..., 0] / boxsize)
                   / (n_mode * 2 * jnp.pi) * boxsize / D_cross)[..., None]
        return jnp.concatenate([psi_raw] + [jnp.zeros_like(psi_raw)] * (dim - 1), -1)

    def psi_minus_no_prefac():
        psi_raw = (jnp.cos(n_mode * 2.0 * jnp.pi * q[..., 0] / boxsize)
                   / (n_mode * 2 * jnp.pi) * boxsize * A_dec)[..., None]
        return jnp.concatenate([psi_raw] + [jnp.zeros_like(psi_raw)] * (dim - 1), -1)

    if from_potential:
        return {"fphi": jnp.fft.rfftn((phi_plus() + phi_minus()).astype(dtype))}
    else:
        X = ((q + cosmo.Dplus(a) * psi_plus_no_prefac()
              + cosmo.E(a) * psi_minus_no_prefac()).astype(dtype))
        P = (a ** 2 * (cosmo.Dplusda(a) * cosmo.E(a) * a * psi_plus_no_prefac()
                       + cosmo.Eda(a) * cosmo.E(a) * a * psi_minus_no_prefac())).astype(dtype)
        return {"pos": X, "vel": P}
