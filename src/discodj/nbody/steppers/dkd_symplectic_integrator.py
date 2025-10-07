import numpy as onp
import jax
import jax.numpy as jnp
from .dkd_leapfrog import DriftKickDrift

__all__ = ["DKDSymplecticIntegrator"]


class DKDSymplecticIntegrator(DriftKickDrift):
    """Drift-kick-drift symplectic integrator.
    """
    integrator_name: str = "symplectic"

    def get_integrator_args(self) -> dict:
        integrator_dict = {"symplectic": self.symplectic}
        return integrator_dict[self.integrator_name]()

    # Second-order symplectic DKD integrator
    def symplectic(self) -> dict:
        dtype = getattr(jnp, f"float{self.dtype_num}")

        # Get the time steps in terms of internal time and scale factor a
        all_a, all_internal = self.all_a_and_internal
        internal_mid = (all_internal[:-1] + all_internal[1:]) / 2.0  # midpoint w.r.t. internal time is used
        a_mid = self.internal_to_a(internal_mid)

        # alphas: 1.0
        alphas = jnp.ones_like(a_mid)

        # betas: 1.5 Omega_m \int da / (a^2 E(a))
        integrand = lambda a: 1.0 / (a ** 2 * self.cosmo.E(a))
        quad_points, weights = onp.polynomial.legendre.leggauss(5)  # use Gauss-Legendre quadrature
        quad_points = quad_points.astype(dtype)
        weights = weights.astype(dtype)
        a_begin = all_a[:-1]
        a_end = all_a[1:]
        integral = lambda a_b, a_e: (
                0.5 * (a_e - a_b) * jnp.sum(weights * integrand(0.5 * (a_e - a_b) * quad_points + 0.5 * (a_b + a_e))))
        betas = 1.5 * self.cosmo.Omega_m * jax.vmap(integral)(a_begin, a_end)

        # Drift length: in terms of superconformal time
        all_superconft = self.cosmo.a_to_superconft(all_a)
        superconft_mid = self.cosmo.a_to_superconft(a_mid)
        dsuperconft1 = superconft_mid - all_superconft[:-1]
        dsuperconft2 = all_superconft[1:] - superconft_mid

        return {"alpha": alphas, "beta": betas, "ddrift1": dsuperconft1, "ddrift2": dsuperconft2}

    def Pi_to_velocity(self, pi, a):
        return pi * self.cosmo.Fplus(a)

    def velocity_to_Pi(self, vel, a):
        return vel / self.cosmo.Fplus(a)
