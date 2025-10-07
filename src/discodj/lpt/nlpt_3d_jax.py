import jax
import jax.numpy as jnp
from jax import Array
import numpy as onp
from inspect import signature
from functools import partial
from .nlpt import NLPT
from .nlpt import conv2_fourier, pad
from ..core.utils import set_0_to_val
from ..cosmology.cosmology import Cosmology

__all__ = ["NLPT3"]


# Derivative helper functions
def z_deriv_fun(X: Array, derivs_rolled_padded: Array, m: int, orig_res: int, ext_res: int, dtype_c: type,
                with_jax: bool = True) -> Array:
    np = jnp if with_jax else onp
    return np.moveaxis(derivs_rolled_padded[:, :, :, 2], source=0, destination=2)[:, :, :ext_res // 2 + 1] \
        * pad(3, orig_res, ext_res, X[..., m], dtype_c=dtype_c, with_jax=with_jax)  # need to crop padding later

def y_deriv_fun(X: Array, derivs_rolled_padded: Array, m: int, orig_res: int, ext_res: int, dtype_c: type,
                with_jax: bool = True) -> Array:
    np = jnp if with_jax else onp
    return np.moveaxis(derivs_rolled_padded[:, :, :, 1], source=0, destination=1) \
        * pad(3, orig_res, ext_res, X[..., m], dtype_c=dtype_c, with_jax=with_jax)

def x_deriv_fun(X: Array, derivs_rolled_padded: Array, m: int, orig_res: int, ext_res: int, dtype_c: type,
                with_jax: bool = True) -> Array:
    return derivs_rolled_padded[:, :, :, 0] \
        * pad(3, orig_res, ext_res, X[..., m], dtype_c=dtype_c, with_jax=with_jax)

def get_derivs_rolled_padded(derivs_ext: Array, ext_res: int, dtype_c: type, with_jax: bool = True):
    np = jnp if with_jax else onp
    derivs_rolled_padded = [np.moveaxis(d, source=i, destination=0) for i, d in enumerate(derivs_ext)]
    derivs_rolled_padded[-1] = np.concatenate(
        [derivs_rolled_padded[-1], np.zeros((ext_res // 2 - 1, 1, 1), dtype=dtype_c)], 0)  # zero-pad last dim.
    return np.stack(derivs_rolled_padded, -1)


# Symmetric mu2 term for j == i - j
def fmu2_sym(f1: Array, orig_res: int, ext_res: int, derivs_ext: Array, dtype_c: type, with_jax: bool = True) -> Array:
    """Compute mu2 in Fourier space in the symmetric case where j == i - j.
    (see Algorithm 1 in arXiv:2010.12584)

    :param f1: field in Fourier space
    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :param derivs_ext: derivative kernels at extended resolution
    :param dtype_c: complex dtype to use
    :param with_jax: use Jax?
    """
    np = jnp if with_jax else onp
    args = {"orig_res": orig_res, "ext_res": ext_res, "dtype_c": dtype_c, "with_jax": with_jax}
    derivs_rolled_padded = get_derivs_rolled_padded(derivs_ext, ext_res, dtype_c, with_jax=with_jax)

    @jax.jit
    def update(A, i, j, k, l, s):
        # indices:
        # i: coordinate of A in first term
        # j: coordinate of A in second term
        # k: direction of derivative in first term
        # l: direction of derivative in second term
        # s: sign

        term1 = jax.lax.cond(k == 2,
                             lambda: z_deriv_fun(A, derivs_rolled_padded, i, **args),
                             lambda: jax.lax.cond(k == 1,
                                                  lambda: y_deriv_fun(A, derivs_rolled_padded, i, **args),
                                                  lambda: x_deriv_fun(A, derivs_rolled_padded, i, **args)))
        term2 = jax.lax.cond(l == 2,
                             lambda: z_deriv_fun(A, derivs_rolled_padded, j, **args),
                             lambda: jax.lax.cond(l == 1,
                                                  lambda: y_deriv_fun(A, derivs_rolled_padded, j, **args),
                                                  lambda: x_deriv_fun(A, derivs_rolled_padded, j, **args)))

        return s * conv2_fourier(3, term1, term2, **args)

    # Define the lists
    i_list = np.array([0, 0, 1, 0, 0, 1])
    j_list = np.array([1, 2, 2, 1, 2, 2])
    k_list = np.array([0, 0, 1, 1, 2, 2])
    l_list = np.array([1, 2, 2, 0, 0, 1])
    s_list = np.array([1, 1, 1, -1, -1, -1], dtype=dtype_c)

    def scan_body(carry, x):
        i, j, k, l, s = x
        carry += update(f1, i, j, k, l, s)
        return carry, None

    return jax.lax.scan(scan_body, init=np.zeros((orig_res, orig_res, orig_res // 2 + 1), dtype=dtype_c),
                        xs=(i_list, j_list, k_list, l_list, s_list))[0]


# Remaining mu2 terms for j != i - j
def fmu2_and_C(f1: Array, f2: Array, orig_res: int, ext_res: int, derivs_ext: Array, dtype_c: type,
               with_jax: bool = True) -> tuple[Array, Array]:
    """Compute mu2 and C in Fourier space in the asymmetric case where j != i - j
    (see Algorithm 1 & 3 in arXiv:2010.12584)

    :param f1: field 1 in Fourier space
    :param f2: field 2 in Fourier space
    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :param derivs_ext: derivative kernels at extended resolution
    :param dtype_c: complex dtype to use
    :param with_jax: use Jax?
    """
    np = jnp if with_jax else onp
    args = {"orig_res": orig_res, "ext_res": ext_res, "dtype_c": dtype_c, "with_jax": with_jax}
    derivs_rolled_padded = get_derivs_rolled_padded(derivs_ext, ext_res, dtype_c, with_jax=with_jax)

    @jax.jit
    def update(A, B, i, j, k, l, s):
        # indices:
        # i: coordinate of A in first term
        # j: coordinate of B in second term
        # k: direction of derivative in first term
        # l: direction of derivative in second term
        # s: sign

        term1 = jax.lax.cond(k == 2,
                             lambda: z_deriv_fun(A, derivs_rolled_padded, i, **args),
                             lambda: jax.lax.cond(k == 1,
                                                  lambda: y_deriv_fun(A, derivs_rolled_padded, i, **args),
                                                  lambda: x_deriv_fun(A, derivs_rolled_padded, i, **args)))
        term2 = jax.lax.cond(l == 2,
                             lambda: z_deriv_fun(B, derivs_rolled_padded, j, **args),
                             lambda: jax.lax.cond(l == 1,
                                                  lambda: y_deriv_fun(B, derivs_rolled_padded, j, **args),
                                                  lambda: x_deriv_fun(B, derivs_rolled_padded, j, **args)))

        return s * conv2_fourier(3, term1, term2, **args)

    def scan_body(carry, x):
        i, j, k, l, s = x
        carry += update(f1, f2, i, j, k, l, s)
        return carry, None

    # Define the lists for mu2
    i_list = np.array([0, 0, 1, 2, 1, 1, 2, 0, 2, 2, 0, 1])
    j_list = np.array([1, 2, 0, 0, 2, 0, 1, 1, 0, 1, 2, 2])
    k_list = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    l_list = np.array([1, 2, 1, 2, 2, 0, 2, 0, 0, 1, 0, 1])
    s_list = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1], dtype=dtype_c)

    # Compute mu2
    mu2 = jax.lax.scan(scan_body, init=np.zeros((orig_res, orig_res, orig_res // 2 + 1), dtype=dtype_c),
                        xs=(i_list, j_list, k_list, l_list, s_list))[0]

    # Define the lists for Cx (Cy, Cz are shifted)
    i_list_C = np.array([0, 0, 1, 1, 2, 2])
    j_list_C = np.array([0, 0, 1, 1, 2, 2])
    k_list_C = np.array([1, 2, 1, 2, 1, 2])
    l_list_C = np.array([2, 1, 2, 1, 2, 1])
    s_list_C = np.array([1, -1, 1, -1, 1, -1], dtype=dtype_c)

    # Compute C
    Cx = jax.lax.scan(scan_body, init=np.zeros((orig_res, orig_res, orig_res // 2 + 1), dtype=dtype_c),
                        xs=(i_list_C, j_list_C, k_list_C, l_list_C, s_list_C))[0]
    Cy = jax.lax.scan(scan_body, init=np.zeros((orig_res, orig_res, orig_res // 2 + 1), dtype=dtype_c),
                      xs=(np.mod(i_list_C + 1, 3), np.mod(j_list_C + 1, 3),
                          np.mod(k_list_C + 1, 3), np.mod(l_list_C + 1, 3),
                          s_list_C))[0]
    Cz = jax.lax.scan(scan_body, init=np.zeros((orig_res, orig_res, orig_res // 2 + 1), dtype=dtype_c),
                      xs=(np.mod(i_list_C + 2, 3), np.mod(j_list_C + 2, 3),
                          np.mod(k_list_C + 2, 3), np.mod(l_list_C + 2, 3),
                          s_list_C))[0]
    C = np.stack([Cx, Cy, Cz], -1)

    return mu2, C


def fmu3(f1: Array, f2: Array, f3: Array, orig_res: int, ext_res: int, derivs_ext: Array | list, dtype_c: type,
         with_jax: bool = True) -> Array:
    """Compute mu3 in Fourier space
    (see Algorithm 2 in arXiv:2010.12584)

    :param f1: field 1 in Fourier space
    :param f2: field 2 in Fourier space
    :param f3: field 3 in Fourier space
    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :param derivs_ext: derivative kernels at extended resolution
    :param dtype_c: complex dtype to use
    :param with_jax: use Jax?
    :return: mu3 in Fourier space
    """
    np = jnp if with_jax else onp
    args = {"orig_res": orig_res, "ext_res": ext_res, "dtype_c": dtype_c, "with_jax": with_jax}
    derivs_rolled_padded = get_derivs_rolled_padded(derivs_ext, ext_res, dtype_c, with_jax=with_jax)

    @jax.jit
    def update(A, B, C, k, l, m, s):
        # indices:
        # k: direction of derivative for A
        # l: direction of derivative for B
        # m: direction of derivative for C
        # s: sign

        termA = jax.lax.cond(k == 2,
                             lambda: z_deriv_fun(A, derivs_rolled_padded, 0, **args),
                             lambda: jax.lax.cond(k == 1,
                                                  lambda: y_deriv_fun(A, derivs_rolled_padded, 0, **args),
                                                  lambda: x_deriv_fun(A, derivs_rolled_padded, 0, **args)))
        term1 = jax.lax.cond(l == 2,
                             lambda: z_deriv_fun(B, derivs_rolled_padded, 1, **args),
                             lambda: jax.lax.cond(l == 1,
                                                  lambda: y_deriv_fun(B, derivs_rolled_padded, 1, **args),
                                                  lambda: x_deriv_fun(B, derivs_rolled_padded, 1, **args)))
        term2 = jax.lax.cond(m == 2,
                             lambda: z_deriv_fun(C, derivs_rolled_padded, 2, **args),
                             lambda: jax.lax.cond(m == 1,
                                                  lambda: y_deriv_fun(C, derivs_rolled_padded, 2, **args),
                                                  lambda: x_deriv_fun(C, derivs_rolled_padded, 2, **args)))

        return s * conv2_fourier(3, termA, conv2_fourier(3, term1, term2, do_crop=False, **args), **args)

    # Define the lists
    k_list = np.array([0, 0, 1, 1, 2, 2])
    l_list = np.array([1, 2, 2, 0, 0, 1])
    m_list = np.array([2, 1, 0, 2, 1, 0])
    s_list = np.array([1, -1, 1, -1, 1, -1], dtype=dtype_c)

    def scan_body(carry, x):
        k, l, m, s = x
        carry += update(f1, f2, f3, k, l, m, s)
        return carry, None

    return jax.lax.scan(scan_body, init=np.zeros((orig_res, orig_res, orig_res // 2 + 1), dtype=dtype_c),
                        xs=(k_list, l_list, m_list, s_list))[0]


class NLPT3(NLPT):
    def __init__(self, dim: int, res: int, cosmo: Cosmology, boxsize: float, n_order: int, grad_kernel_order: int = 0,
                 with_jax: bool = True, try_to_jit: bool = True, dtype_num: int = 32, dtype_c_num: int = 64,
                 batched: bool = False, exact_growth: bool = False, no_mode_left_behind: bool = False,
                 no_transverse: bool = False, no_factors: bool = False):
        """NLPT class for 3D LPT (at arbitrary order) for Jax.

        :param dim: dimensionality of the simulation
        :param res: resolution of the simulation per dimension
        :param cosmo: cosmology object
        :param boxsize: boxsize of the simulation in Mpc/h
        :param n_order: order of the LPT
        :param grad_kernel_order: order of the gradient kernel
        :param with_jax: use Jax (needs to be true)
        :param try_to_jit: if True, try to jit the compute_core function
        :param dtype_num: 32 or 64 (float32 or float64)
        :param dtype_c_num: 64 or 128 (complex64 or complex128)
        :param batched: if True: use batched computation, so fields have a leading batch dimension
        :param exact_growth: if True: use exact growth factor
        :param no_mode_left_behind: if True: do not leave any mode behind: not implemented in 3D!
        :param no_transverse: if True: do not compute transverse modes
        :param no_factors: if True: factor for each term is set to 1
        """
        superclass_params = signature(NLPT.__init__).parameters
        args = {k: v for k, v in locals().items() if k in superclass_params and k != 'self'}
        super(NLPT3, self).__init__(**args)
        assert self.with_jax, "Jax must be used for NLPT3"

    def compute(self, fphi_ini: Array) -> dict:
        """Compute the LPT displacement fields psi at each order and store them in self.psi.

        :param fphi_ini: initial potential field in Fourier space
        :return: dictionary with the displacement fields psi
        """
        d_dx, d_dy, d_dz = (
            self.grad_kernel(self.k_vecs, 0), self.grad_kernel(self.k_vecs, 1), self.grad_kernel(self.k_vecs, 2))
        d_dx_ext, d_dy_ext, d_dz_ext = (self.grad_kernel(self.k_vecs_ext, 0),
                                        self.grad_kernel(self.k_vecs_ext, 1),
                                        self.grad_kernel(self.k_vecs_ext, 2))
        inv_lap = self.inv_laplace_kernel(self.k_vecs)

        kwargs = {"derivs": (d_dx, d_dy, d_dz, inv_lap),
                  "derivs_ext": (d_dx_ext, d_dy_ext, d_dz_ext),
                  "n_order": self.n_order,
                  "res": self.res,
                  "dtype_num": self.dtype_num,
                  "dtype_c_num": self.dtype_c_num,
                  "with_jax": self.with_jax,
                  "no_transverse": self.no_transverse,
                  "no_factors": self.no_factors}

        compute_core = self.compute_core_exact if self.exact_growth else self.compute_core
        compute_fun = partial(jax.jit, static_argnames=("n_order", "res", "with_jax", "dtype_num", "dtype_c_num",
                                                        "no_transverse", "no_factors"))(
            compute_core) if self.with_jax and self.try_to_jit else compute_core

        if self.batched:
            compute_fun_final = jax.vmap(lambda phi: compute_fun(phi, **kwargs), in_axes=0, out_axes=0)
        else:
            compute_fun_final = lambda phi: compute_fun(phi, **kwargs)

        out = compute_fun_final(fphi_ini)

        # Delete big arrays that are no longer needed
        del d_dx, d_dy, d_dz, inv_lap, d_dx_ext, d_dy_ext, d_dz_ext
        del self.k_dict, self.k, self.k_vecs, self.k_dict_ext, self.k_ext, self.k_vecs_ext

        return out

    @staticmethod
    def compute_core(fphi_ini, derivs, derivs_ext, n_order, res, with_jax, dtype_num, dtype_c_num, no_transverse,
                     no_factors) -> dict:
        """Core of the computation."""
        np = jnp if with_jax else onp
        dtype = getattr(np, f"float{dtype_num}")
        dtype_c = getattr(np, f"complex{dtype_c_num}")

        # Extract derivatives
        d_dx, d_dy, d_dz, inv_lap = derivs
        grad_list = [d_dx, d_dy, d_dz]

        # Compute fphi
        fphi = set_0_to_val(dim=3, t=fphi_ini, v=0.0)

        # Set up first order (Zel'dovich), in Fourier space
        psi = -np.stack([(deriv * fphi) for deriv in grad_list], -1)[None, ...]

        # Resolution for de-aliases convolutions
        ext_res = 3 * res // 2

        def scan_body_mu2_and_C(carry, x):
            i, fL, fT, C_multiplier = carry
            j, psi_j, psi_i_minus_j = x
            if no_factors:
                fac_mu2 = np.array(1.0, dtype=fL.dtype)
                fac_C = np.array(1.0, dtype=fT.dtype)
            else:
                fac_mu2 = (((3 - i) / 2 - j ** 2 - (i - j) ** 2) / ((i + 3 / 2) * (i - 1))).astype(fL.dtype)
                fac_C = (1.0 - 2 * j / i).astype(fT.dtype)
            mu2, C = fmu2_and_C(psi_j, psi_i_minus_j, res, ext_res, derivs_ext, dtype_c=dtype_c, with_jax=with_jax)
            fL += fac_mu2 * mu2
            fT += fac_C * C * C_multiplier
            return (i, fL, fT, C_multiplier), None

        def scan_body_mu3(carry, x):
            i, fL = carry
            k, l, psi_k, psi_l, psi_i_minus_k_minus_l = x
            if no_factors:
                fac = np.array(1.0, dtype=fL.dtype)
            else:
                fac = (((3 - i) / 2 - k ** 2 - l ** 2 - (i - k - l) ** 2) / ((i + 3 / 2) * (i - 1))).astype(fL.dtype)
            fL += fac * fmu3(psi_k, psi_l, psi_i_minus_k_minus_l, res, ext_res, derivs_ext,
                             dtype_c=dtype_c, with_jax=with_jax)
            return (i, fL), None

        # Iterate over higher orders
        for i in range(2, n_order + 1):

            fL = np.zeros((res, res, res // 2 + 1), dtype=dtype_c)
            fT = np.zeros((res, res, res // 2 + 1, 3), dtype=dtype_c)

            # first: mu2 part for even orders where j == i - j -> j = i / 2
            if i % 2 == 0:
                fac_sym = 1.0 if no_factors else ((3 - i) / 2 - (i // 2) ** 2 - (i // 2) ** 2) / ((i + 3 / 2) * (i - 1))
                fL += fac_sym * fmu2_sym(psi[i // 2 - 1], res, ext_res, derivs_ext, dtype_c=dtype_c, with_jax=with_jax)

            # now: other terms
            j_vec = np.arange(1, (i + 1) // 2)  # even case: don't include i // 2 here as it's been taken care of above
            i_minus_j_vec = np.arange(i - 1, i // 2, -1)

            if i > 2:
                # mu2 and C for j != i - j
                C_multiplier = 0.0 if no_transverse else 1.0

                fL, fT = jax.lax.scan(f=scan_body_mu2_and_C, init=(i, fL, fT, C_multiplier),
                                      xs=(j_vec,  # j
                                          psi[j_vec - 1, ...],  # psi[j]
                                          psi[i_minus_j_vec - 1, ...])  # psi[i - j]
                                      )[0][1:3]  # [0]: carry [1]: fL [2]: fT

                # mu3
                for k in range(1, i - 1):
                    l_vec = np.arange(1, i - k)
                    n_l = l_vec.shape[0]
                    fL = jax.lax.scan(f=scan_body_mu3, init=(i, fL),
                                      xs=(np.array([k] * n_l),  # k
                                          l_vec,  # l
                                          np.tile(psi[k-1:k, ...], [n_l, 1, 1, 1, 1]),  # psi[k]
                                          psi[0:i-k-1, ...],  # psi[l]
                                          psi[i-k-2::-1],  # psi[i - k - l]
                                          )
                                      )[0][1]  # [0]: carry [1]: fL

            psi_i_x = inv_lap * (d_dx * fL - (d_dy * fT[..., 2] - d_dz * fT[..., 1]))
            psi_i_y = inv_lap * (d_dy * fL - (d_dz * fT[..., 0] - d_dx * fT[..., 2]))
            psi_i_z = inv_lap * (d_dz * fL - (d_dx * fT[..., 1] - d_dy * fT[..., 0]))
            psi = np.concatenate([psi, jnp.stack([psi_i_x, psi_i_y, psi_i_z], -1)[None, ...]], 0)
            del psi_i_x, psi_i_y, psi_i_z

        # Convert to real space
        psi = jax.vmap(jax.vmap(np.fft.irfftn, in_axes=0, out_axes=0), in_axes=-1, out_axes=-1)(psi)

        # Make a dictionary out of it
        return {f"psi_{i}": psi[i - 1] for i in range(1, n_order + 1)}  # psi_1 = psi[0], ...

    @staticmethod
    def compute_core_exact(fphi_ini, derivs, derivs_ext, n_order, res, with_jax, dtype_num, dtype_c_num, no_transverse,
                           no_factors) -> dict:
        """Core of the computation for the case with exact growth.
        The growth factors themselves are not included in the spatial fields computed here.
        """
        np = jnp if with_jax else onp
        out_dict = {}

        dtype = getattr(np, f"float{dtype_num}")
        dtype_c = getattr(np, f"complex{dtype_c_num}")

        # Extract derivatives
        d_dx, d_dy, d_dz, inv_lap = derivs
        grad_list = [d_dx, d_dy, d_dz]

        # Compute fphi
        fphi = set_0_to_val(dim=3, t=fphi_ini, v=0.0)

        # Set up first order (Zel'dovich), in Fourier space
        out_dict["psi_1"] = -np.stack([(deriv * fphi) for deriv in grad_list], -1)

        if n_order >= 2:
            # Resolution for de-aliases convolutions
            ext_res = 3 * res // 2

            def scan_body_mu2_and_C(carry, x):
                i, fL, fT, C_multiplier = carry
                j, psi_j, psi_i_minus_j = x
                mu2, C = fmu2_and_C(psi_j, psi_i_minus_j, res, ext_res, derivs_ext, dtype_c=dtype_c, with_jax=with_jax)
                fL -= 0.5 * mu2
                fT -= C * C_multiplier
                return (i, fL, fT, C_multiplier), None

            def scan_body_mu3(carry, x):
                i, fL = carry
                k, l, psi_k, psi_l, psi_i_minus_k_minus_l = x
                fac = -1.0
                fL += fac * fmu3(psi_k, psi_l, psi_i_minus_k_minus_l, res, ext_res, derivs_ext,
                                 dtype_c=dtype_c, with_jax=with_jax)
                return (i, fL), None

            # 2LPT term
            fL_2lpt = fmu2_sym(out_dict["psi_1"], res, ext_res, derivs_ext, dtype_c=dtype_c, with_jax=with_jax)
            psi_i_x, psi_i_y, psi_i_z = inv_lap * (d_dx * fL_2lpt), inv_lap * (d_dy * fL_2lpt), inv_lap * (d_dz * fL_2lpt)
            out_dict["psi_2_ex"] = np.stack([psi_i_x, psi_i_y, psi_i_z], -1)

            if n_order >= 3:
                # 3LPT terms
                fL_3a_init = np.zeros((res, res, res // 2 + 1), dtype=dtype_c)
                fL_3b_init = np.zeros((res, res, res // 2 + 1), dtype=dtype_c)
                fT_3c_init = np.zeros((res, res, res // 2 + 1, 3), dtype=dtype_c)

                l_vec = np.asarray([1])
                n_l = l_vec.shape[0]
                fL_3a = jax.lax.scan(f=scan_body_mu3, init=(3, fL_3a_init),
                                     xs=(np.array([1] * n_l),  # k
                                         l_vec,  # l
                                         out_dict["psi_1"][None, ...],  # psi[k]
                                         out_dict["psi_1"][None, ...],  # psi[l]
                                         out_dict["psi_1"][None, ...],  # psi[i - k - l]
                                         )
                                     )[0][1]  # [0]: carry [1]: fL

                fL_3b, fT_3c = jax.lax.scan(f=scan_body_mu2_and_C, init=(3, fL_3b_init, fT_3c_init, 1.0),
                                            xs=(np.asarray([1]),  # j
                                                out_dict["psi_1"][None, ...],  # psi[j]
                                                out_dict["psi_2_ex"][None, ...])  # psi[i - j]
                                            )[0][1:3]  # [0]: carry [1]: fL [2]: fT

                psi_3a_x, psi_3a_y, psi_3a_z = inv_lap * (d_dx * fL_3a), inv_lap * (d_dy * fL_3a), inv_lap * (d_dz * fL_3a)
                psi_3b_x, psi_3b_y, psi_3b_z = inv_lap * (d_dx * fL_3b), inv_lap * (d_dy * fL_3b), inv_lap * (d_dz * fL_3b)
                psi_3c_x = inv_lap * (- (d_dy * fT_3c[..., 2] - d_dz * fT_3c[..., 1]))
                psi_3c_y = inv_lap * (- (d_dz * fT_3c[..., 0] - d_dx * fT_3c[..., 2]))
                psi_3c_z = inv_lap * (- (d_dx * fT_3c[..., 1] - d_dy * fT_3c[..., 0]))

                out_dict["psi_3a_ex"] = np.stack([psi_3a_x, psi_3a_y, psi_3a_z], -1)
                out_dict["psi_3b_ex"] = np.stack([psi_3b_x, psi_3b_y, psi_3b_z], -1)
                out_dict["psi_3c_ex"] = np.stack([psi_3c_x, psi_3c_y, psi_3c_z], -1)

        # Convert to real space
        vectorial_irfftn = lambda v: jax.vmap(np.fft.irfftn, in_axes=3, out_axes=3)(v)
        out_dict = {k: (vectorial_irfftn(v)) if k != "psi_0" else v for k, v in out_dict.items()}

        return out_dict
