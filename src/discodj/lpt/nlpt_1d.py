import jax
import jax.numpy as jnp
import numpy as onp
from functools import partial
from inspect import signature
from .nlpt import NLPT
from ..core.utils import set_0_to_val
from ..core.types import AnyArray
from ..cosmology.cosmology import Cosmology

__all__ = ["NLPT1"]


class NLPT1(NLPT):
    def __init__(self, dim: int, res: int, cosmo: Cosmology, boxsize: float, n_order: int, grad_kernel_order: int = 0,
                 with_jax: bool = True, try_to_jit: bool = True, dtype_num: int = 32, dtype_c_num: int = 64,
                 batched: bool = False, exact_growth: bool = False, no_mode_left_behind: bool = False,
                 no_transverse: bool = False, no_factors: bool = False):
        """NLPT class for 1D LPT (only Zel'dovich approximation).

        :param dim: dimensionality of the simulation
        :param res: resolution of the simulation per dimension
        :param cosmo: cosmology object
        :param boxsize: boxsize of the simulation in Mpc/h
        :param n_order: order of the LPT (NOTE: in 1D, all orders >1 vanish)
        :param grad_kernel_order: order of the gradient kernel
        :param with_jax: use Jax?
        :param try_to_jit: if True, try to jit the compute_core function
        :param dtype_num: 32 or 64 (float32 or float64)
        :param dtype_c_num: 64 or 128 (complex64 or complex128)
        :param batched: if True: use batched computation, so fields have a leading batch dimension
        :param exact_growth: if True: use exact growth factor
        :param no_mode_left_behind: ignored in 1D
        :param no_transverse: ignored in 1D
        :param no_factors: if True: factor for each term is set to 1
        """
        superclass_params = signature(NLPT.__init__).parameters
        args = {k: v for k, v in locals().items() if k in superclass_params and k != 'self'}
        super(NLPT1, self).__init__(**args)
        assert n_order == 1, "Only Zel'dovich approximation is supported in 1D"

    def compute(self, fphi_ini: AnyArray) -> dict:
        """Compute the Zel'dovich displacement field.

        :param fphi_ini: initial potential field in Fourier space
        :return: dictionary with the displacement fields psi
        """

        d_dx = self.grad_kernel(self.k_vecs, 0)

        compute_fun = partial(jax.jit, static_argnames=("with_jax", "dtype_num"))(self.compute_core) \
            if self.with_jax and self.try_to_jit else self.compute_core

        if self.batched:
            assert self.try_to_jit, "Batched computation is only supported with jit"
            compute_fun_final = jax.vmap(lambda phi: compute_fun(phi, d_dx, self.with_jax, self.dtype_num),
                                         in_axes=0, out_axes=0)
        else:
            compute_fun_final = lambda phi: compute_fun(phi, d_dx, self.with_jax, self.dtype_num)
        return compute_fun_final(fphi_ini)

    @staticmethod
    def compute_core(fphi_ini: AnyArray, d_dx: AnyArray, with_jax: bool, dtype_num: int) -> dict:
        """Core of the computation, can be jitted or not.
        """
        np = jnp if with_jax else onp
        dtype = getattr(np, f"float{dtype_num}")
        out_dict = {}

        # Zero out DC mode
        fphi = set_0_to_val(dim=1, t=fphi_ini, v=0.0)

        # Set up first order (Zel'dovich), in Fourier space
        out_dict["psi_1"] = -np.fft.irfft(d_dx * fphi)[:, np.newaxis].astype(dtype)

        return out_dict
