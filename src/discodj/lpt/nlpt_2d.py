import jax
import jax.numpy as jnp
import numpy as onp
from inspect import signature
from functools import partial
from .nlpt import NLPT, fmu2_1, fmu2_C_2
from ..core.grids import get_fourier_grid
from ..core.utils import set_0_to_val, crop
from ..core.types import AnyArray
from ..cosmology.cosmology import Cosmology

__all__ = ["NLPT2"]


# TODO: Rewrite this in the 3D style
class NLPT2(NLPT):
    def __init__(self, dim: int, res: int, cosmo: Cosmology, boxsize: float, n_order: int, grad_kernel_order: int = 0,
                 with_jax: bool = True, try_to_jit: bool = True, dtype_num: int = 32, dtype_c_num: int = 64,
                 batched: bool = False, exact_growth: bool = False, no_mode_left_behind: bool = False,
                 no_transverse: bool = False, no_factors: bool = False):
        """NLPT class for 2D LPT (at arbitrary order).

        :param dim: dimensionality of the simulation
        :param res: resolution of the simulation per dimension
        :param cosmo: cosmology object
        :param boxsize: boxsize of the simulation in Mpc/h
        :param n_order: order of the LPT        
        :param grad_kernel_order: order of the gradient kernel
        :param with_jax: use Jax?
        :param dtype_num: 32 or 64 (float32 or float64)
        :param dtype_c_num: 64 or 128 (complex64 or complex128)
        :param batched: if True: use batched computation, so fields have a leading batch dimension
        :param exact_growth: if True: use exact growth factor
        :param no_mode_left_behind: if True: do not leave any mode behind
        :param no_transverse: if True: do not compute transverse modes (has no effect in 2D!)
        :param no_factors: if True: factor for each term is set to 1
        """
        superclass_params = signature(NLPT.__init__).parameters
        args = {k: v for k, v in locals().items() if k in superclass_params and k != 'self'}
        super(NLPT2, self).__init__(**args)

        # self.kx, self.ky = self.kks
        # self.d_dx, self.d_dy = self.grad_kernel(self.kks, 0), self.grad_kernel(self.kks, 1)
        # self.inv_lap = self.inv_laplace_kernel(self.kks)
        # self.d_dx_ext, self.d_dy_ext = self.grad_kernel(self.kks_ext, 0), self.grad_kernel(self.kks_ext, 1)
        # self.inv_lap_ext = self.inv_laplace_kernel(self.kks_ext)
        # self.cellsize = res / boxsize
        # self._np = jnp if with_jax else onp


    def compute(self, fphi_ini: AnyArray) -> dict:
        """Compute the LPT displacement fields psi at each order and store them in self.psi

        :param fphi_ini: initial potential field in Fourier space
        :return: dictionary with the displacement fields psi
        """
        d_dx, d_dy = self.grad_kernel(self.k_vecs, 0), self.grad_kernel(self.k_vecs, 1)
        d_dx_ext, d_dy_ext = self.grad_kernel(self.k_vecs_ext, 0), self.grad_kernel(self.k_vecs_ext, 1)
        inv_lap = self.inv_laplace_kernel(self.k_vecs)

        kwargs = {"derivs": (d_dx, d_dy, inv_lap),
                  "derivs_ext": (d_dx_ext, d_dy_ext),
                  "n_order": self.n_order,
                  "res": self.res,
                  "dtype_num": self.dtype_num,
                  "dtype_c_num": self.dtype_c_num,
                  "with_jax": self.with_jax,
                  "no_transverse": self.no_transverse,
                  "no_factors": self.no_factors,
                  "no_mode_left_behind": self.no_mode_left_behind}
        if self.no_mode_left_behind:
            kwargs["grad_kernel"] = self.grad_kernel
            kwargs["inv_laplace_kernel"] = self.inv_laplace_kernel
            kwargs["boxsize"] = self.boxsize
        else:
            kwargs["grad_kernel"] = None
            kwargs["inv_laplace_kernel"] = None
            kwargs["boxsize"] = None

        compute_fun = partial(jax.jit, static_argnames=("n_order", "res", "with_jax", "dtype_num", "dtype_c_num",
                                                        "no_transverse", "no_factors", "no_mode_left_behind",
                                                        "grad_kernel", "inv_laplace_kernel", "boxsize"))(
            self.compute_core) if self.with_jax and self.try_to_jit else self.compute_core

        if self.batched:
            compute_fun_final = jax.vmap(lambda phi: compute_fun(phi, **kwargs), in_axes=0, out_axes=0)
        else:
            compute_fun_final = lambda phi: compute_fun(phi, **kwargs)
        return compute_fun_final(fphi_ini)

    @staticmethod
    def compute_core(fphi_ini, derivs, derivs_ext, n_order, res, with_jax, dtype_num, dtype_c_num, no_transverse,
                     no_factors, no_mode_left_behind: bool, grad_kernel=None, inv_laplace_kernel=None,
                     boxsize=None) -> dict:
        """Core of the computation, can be jitted or not."""
        np = jnp if with_jax else onp
        out_dict = {}

        dtype = getattr(np, f"float{dtype_num}")
        dtype_c = getattr(np, f"complex{dtype_c_num}")

        # Extract derivatives
        d_dx, d_dy, inv_lap = derivs
        grad_list = [d_dx, d_dy]

        # Compute fphi
        fphi = set_0_to_val(dim=2, t=fphi_ini, v=0.0)

        # Set up first order (Zel'dovich), in Fourier space
        out_dict["psi_1"] = -np.stack([(deriv * fphi) for deriv in grad_list], -1)

        res_current = res
        do_crop = not no_mode_left_behind

        # Higher orders
        for i in range(2, n_order + 1):
            ext_res = 3 * res_current // 2
            next_res = res_current if do_crop else ext_res
            fL = np.zeros((next_res, next_res // 2 + 1), dtype=dtype_c)
            fT = np.zeros((next_res, next_res // 2 + 1), dtype=dtype_c)

            # For nomodeleftbehind: get derivative kernels for current resolution
            if no_mode_left_behind:
                k_dict_curr = get_fourier_grid((next_res, next_res), boxsize=boxsize, full=False,
                                               relative=False, dtype_num=dtype_num, with_jax=False)  # always use Numpy
                k_vecs_current = k_dict_curr["k_vecs"]
                inv_lap_current = inv_laplace_kernel(k_vecs_current)
                d_dx_current, d_dy_current = grad_kernel(k_vecs_current, 0), grad_kernel(k_vecs_current, 1)

            # Iterate over j
            for j in range(1, i // 2 + 1):
                fac = 1.0 if no_factors else ((3 - i) / 2 - j ** 2 - (i - j) ** 2) / ((i + 3 / 2) * (i - 1))

                if j == i - j:
                    if no_mode_left_behind:
                        res_current_1 = int(1.5 ** (j - 1) * res)
                        mu2 = fmu2_1(2, out_dict[f"psi_{j}"], res_current_1, next_res,
                                     (d_dx_current, d_dy_current), dtype_c=dtype_c,
                                     with_jax=with_jax, do_crop=do_crop)
                    else:
                        mu2 = fmu2_1(2, out_dict[f"psi_{j}"], res_current, ext_res, derivs_ext, dtype_c=dtype_c,
                                     with_jax=with_jax,
                                     do_crop=do_crop)
                    fL += fac * mu2

                else:
                    if no_mode_left_behind:
                        res_current_1 = int(1.5 ** (j - 1) * res)
                        res_current_2 = int(1.5 ** (i - j - 1) * res)
                        res_current_list = (res_current_1, res_current_2)

                        mu2, C = fmu2_C_2(2, out_dict[f"psi_{j}"], out_dict[f"psi_{i - j}"], res_current_list,
                                          next_res,(d_dx_current, d_dy_current),
                                          dtype_c=dtype_c, with_jax=with_jax, do_crop=do_crop)
                    else:
                        mu2, C = fmu2_C_2(2, out_dict[f"psi_{j}"], out_dict[f"psi_{i - j}"], (res, res),
                                          ext_res, derivs_ext, dtype_c=dtype_c, with_jax=with_jax, do_crop=do_crop)

                    if no_transverse:
                        C = 0.0

                    fac_T = 1.0 if no_factors else (1.0 - 2 * j / i)

                    fL += fac * mu2
                    fT += fac_T * C

                if no_mode_left_behind:
                    psi_i_x = inv_lap_current * (d_dx_current * fL - d_dy_current * fT)
                    psi_i_y = inv_lap_current * (d_dy_current * fL + d_dx_current * fT)

                else:
                    psi_i_x = inv_lap * (d_dx * fL - d_dy * fT)
                    psi_i_y = inv_lap * (d_dy * fL + d_dx * fT)
                out_dict[f"psi_{i}"] = np.stack([psi_i_x, psi_i_y], -1)

            if no_mode_left_behind:
                res_current = next_res

        # Convert to real space
        for i in range(1, n_order + 1):

            # For no_mode_left_behind: crop the psi's now
            if no_mode_left_behind:
                res_current = int(res * 1.5 ** (i - 1))
                out_dict[f"psi_{i}"] = np.stack([crop(2, res, res_current, out_dict[f"psi_{i}"][:, :, 0],
                                                      dtype_c, full=False, with_jax=with_jax),
                                                 crop(2, res, res_current, out_dict[f"psi_{i}"][:, :, 1], dtype_c,
                                                      full=False, with_jax=with_jax)], -1)

            f0 = np.fft.irfftn(out_dict[f"psi_{i}"][:, :, 0])
            f1 = np.fft.irfftn(out_dict[f"psi_{i}"][:, :, 1])

            out_dict[f"psi_{i}"] = np.stack([f0, f1], -1).astype(dtype)

        return out_dict
