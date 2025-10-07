import warnings
import jax
import jax.numpy as jnp
import numpy as onp
from functools import partial
from einops import rearrange
from ..core.grids import get_fourier_grid
from ..core.kernels import gradient_kernel, inv_laplace_kernel
from ..core.utils import set_0_to_val, pad, conv2_fourier
from ..core.types import AnyArray
from ..cosmology.cosmology import Cosmology

__all__ = ["NLPT", "fmu2_1", "fmu2_C_2"]


def fmu2_1(dim: int, f1: AnyArray, orig_res: int, ext_res: int, derivs_ext: AnyArray | list | tuple, dtype_c: type,
           with_jax: bool = True, do_crop: bool = True) -> AnyArray:
    """Compute mu2 in Fourier space in the symmetric case where j == i - j
    (see Algorithm 1 in arXiv:2010.12584).

    :param dim: number of dimensions
    :param f1: field in Fourier space
    :param orig_res: original resolution per dimension
    :param ext_res: extended resolution per dimension
    :param derivs_ext: derivative kernels at extended resolution
    :param dtype_c: complex dtype to use
    :param with_jax: use Jax?
    :param do_crop: crop mu2 to the original resolution
    :return: mu2 in Fourier space
    """
    a = {"orig_res": orig_res, "ext_res": ext_res, "dtype_c": dtype_c, "with_jax": with_jax, "do_crop": do_crop}

    if dim == 2:
        ff1x = pad(2, orig_res, ext_res, f1[..., 0], dtype_c=dtype_c, with_jax=with_jax)
        ff1y = pad(2, orig_res, ext_res, f1[..., 1], dtype_c=dtype_c, with_jax=with_jax)
        d_dx_ext, d_dy_ext = derivs_ext

        fdxf1x = d_dx_ext * ff1x
        fdyf1x = d_dy_ext * ff1x
        fdxf1y = d_dx_ext * ff1y
        fdyf1y = d_dy_ext * ff1y

        return conv2_fourier(2, fdxf1x, fdyf1y, **a) - conv2_fourier(2, fdyf1x, fdxf1y, **a)

    elif dim == 3:
        ff1x, ff1y, ff1z = [pad(3, orig_res, ext_res, f1[:, :, :, i], dtype_c=dtype_c, with_jax=with_jax)
                            for i in range(3)]
        fdxf1x, fdyf1x, fdzf1x = derivs_ext[0] * ff1x, derivs_ext[1] * ff1x, derivs_ext[2] * ff1x
        fdxf1y, fdyf1y, fdzf1y = derivs_ext[0] * ff1y, derivs_ext[1] * ff1y, derivs_ext[2] * ff1y
        fdxf1z, fdyf1z, fdzf1z = derivs_ext[0] * ff1z, derivs_ext[1] * ff1z, derivs_ext[2] * ff1z

        return (conv2_fourier(3, fdxf1x, fdyf1y, **a) + conv2_fourier(3, fdxf1x, fdzf1z, **a)
                + conv2_fourier(3, fdyf1y, fdzf1z, **a)
                - conv2_fourier(3, fdyf1x, fdxf1y, **a) - conv2_fourier(3, fdzf1x, fdxf1z, **a)
                - conv2_fourier(3, fdzf1y, fdyf1z, **a))

    else:
        raise NotImplementedError


def fmu2_C_2(dim: int, f1: AnyArray, f2: AnyArray, orig_res_list: list | tuple, ext_res: int,
             derivs_ext: AnyArray | list | tuple, dtype_c: type, with_jax: bool = True,
             do_crop: bool = True) -> tuple[AnyArray, AnyArray]:
    """Compute mu2 and C in Fourier space in the asymmetric case where j != i - j
    (see Algorithms 1 & 3 in arXiv:2010.12584).

    :param dim: number of dimensions
    :param f1: field in Fourier space
    :param f2: field in Fourier space
    :param orig_res_list: original resolution per dimension, for f1 and f2 (might be different for no_mode_left_behind)
    :param ext_res: extended resolution per dimension
    :param derivs_ext: derivative kernels at extended resolution
    :param dtype_c: complex dtype to use
    :param with_jax: use Jax?
    :param do_crop: crop mu2 to the original resolution
    :return: mu2 and C in Fourier space
    """
    np = jnp if with_jax else onp
    if do_crop:
        assert onp.all([rr == orig_res_list[0] for rr in orig_res_list[1:]])
        orig_res = orig_res_list[0]
    else:
        orig_res = None

    a = {"orig_res": orig_res, "ext_res": ext_res, "dtype_c": dtype_c, "with_jax": with_jax, "do_crop": do_crop}

    if dim == 2:
        ff1x = pad(2, orig_res_list[0], ext_res, f1[..., 0], dtype_c=dtype_c, with_jax=with_jax)
        ff1y = pad(2, orig_res_list[0], ext_res, f1[..., 1], dtype_c=dtype_c, with_jax=with_jax)
        ff2x = pad(2, orig_res_list[1], ext_res, f2[..., 0], dtype_c=dtype_c, with_jax=with_jax)
        ff2y = pad(2, orig_res_list[1], ext_res, f2[..., 1], dtype_c=dtype_c, with_jax=with_jax)
        d_dx_ext, d_dy_ext = derivs_ext

        fdxf1x = d_dx_ext * ff1x
        fdyf1x = d_dy_ext * ff1x
        fdxf1y = d_dx_ext * ff1y
        fdyf1y = d_dy_ext * ff1y
        fdxf2x = d_dx_ext * ff2x
        fdyf2x = d_dy_ext * ff2x
        fdxf2y = d_dx_ext * ff2y
        fdyf2y = d_dy_ext * ff2y

        return conv2_fourier(2, fdxf1x, fdyf2y, **a) + conv2_fourier(2, fdyf1y, fdxf2x, **a) \
               - conv2_fourier(2, fdyf1x, fdxf2y, **a) - conv2_fourier(2, fdxf1y, fdyf2x, **a), \
               conv2_fourier(2, fdxf1x, fdyf2x, **a) + conv2_fourier(2, fdxf1y, fdyf2y, **a) \
               - conv2_fourier(2, fdyf1x, fdxf2x, **a) - conv2_fourier(2, fdyf1y, fdxf2y, **a)

    elif dim == 3:
        assert orig_res_list[0] == orig_res_list[1] == orig_res_list[2], "In 3D, all resolutions must be equal for now!"
        orig_res = orig_res_list[0]
        shape = [orig_res, orig_res, orig_res // 2 + 1, 3]

        f1_p = [pad(3, orig_res, ext_res, f1[:, :, :, i], dtype_c=dtype_c, with_jax=with_jax) for i in range(3)]
        f2_p = [pad(3, orig_res, ext_res, f2[:, :, :, i], dtype_c=dtype_c, with_jax=with_jax) for i in range(3)]

        mu2 = np.zeros(shape[:-1], dtype=dtype_c)  # scalar
        Cx, Cy, Cz = np.zeros(shape[:-1], dtype=dtype_c), \
            np.zeros(shape[:-1], dtype=dtype_c), np.zeros(shape[:-1], dtype=dtype_c)  # vectorial

        a = {"orig_res": orig_res, "ext_res": ext_res, "dtype_c": dtype_c, "with_jax": with_jax}

        # TODO: improve this double for-loop (but it's anyway not used in the 3D implementation)
        for i in range(3):
            # Compute mu2
            j = (i + 1) % 3
            k = (i + 2) % 3
            mu2 += conv2_fourier(3, derivs_ext[i] * f1_p[i], derivs_ext[j] * f2_p[j], **a) \
                   + conv2_fourier(3, derivs_ext[i] * f1_p[i], derivs_ext[k] * f2_p[k], **a) \
                   - conv2_fourier(3, derivs_ext[i] * f1_p[j], derivs_ext[j] * f2_p[i], **a) \
                   - conv2_fourier(3, derivs_ext[i] * f1_p[k], derivs_ext[k] * f2_p[i], **a)

            # Compute C
            for l in range(3):
                update = conv2_fourier(3, derivs_ext[j] * f1_p[l], derivs_ext[k] * f2_p[l], **a) \
                         - conv2_fourier(3, derivs_ext[k] * f1_p[l], derivs_ext[j] * f2_p[l], **a)
                if i == 0:
                    Cx += update
                elif i == 1:
                    Cy += update
                elif i == 2:
                    Cz += update

        C = np.stack([Cx, Cy, Cz], -1)

        # note: no division by 2 here, but neither a multiplication with 2 in the loop below
        return mu2, C


# @jax.tree_util.register_pytree_node_class (not needed for now)
class NLPT:
    def __init__(self, dim: int, res: int, cosmo: Cosmology, boxsize: float, n_order: int, grad_kernel_order: int = 0,
                 dtype_num: int = 32, dtype_c_num: int = 64, with_jax: bool = True, try_to_jit: bool = True,
                 batched: bool = False, exact_growth: bool = False, no_mode_left_behind: bool = False,
                 no_transverse: bool = False, no_factors: bool = False):
        """NLPT base class with common attributes for all orders.

        :param dim: dimensionality of the simulation
        :param res: resolution of the simulation per dimension
        :param boxsize: boxsize of the simulation in Mpc/h
        :param n_order: order of the LPT
        :param grad_kernel_order: order of the gradient kernel
        :param dtype_num: 32 or 64 (float32 or float64)
        :param dtype_c_num: 64 or 128 (complex64 or complex128)
        :param with_jax: use Jax?
        :param try_to_jit: if True, try to jit the compute method
        :param batched: if True: use batched computation, so fields have a batch dimension in front
        :param exact_growth: if True: use exact growth factor for each order (only in 3D)
        :param no_mode_left_behind: if True: new modes arising at each LPT order are traced through computation
        :param no_transverse: if True: transverse modes are not computed
        :param no_factors: if True: factor for each term is set to 1
        """
        self._np = jnp if with_jax else onp
        self.dim = dim
        self.res = res
        self.boxsize = boxsize
        self.n_order = n_order
        self.shape = dim * [res]
        self.with_jax = with_jax
        self.try_to_jit = try_to_jit
        self.dtype = self._np.float64 if dtype_num == 64 else self._np.float32
        self.dtype_c = self._np.complex128 if dtype_c_num == 128 else self._np.complex64
        self.dtype_num = dtype_num
        self.dtype_c_num = dtype_c_num
        self.grad_kernel_order = grad_kernel_order
        self.grad_kernel = partial(gradient_kernel, order=grad_kernel_order, with_jax=with_jax)
        self.inv_laplace_kernel = partial(inv_laplace_kernel, with_jax=with_jax)
        self.batched = batched
        self.exact_growth = exact_growth
        self.no_mode_left_behind = no_mode_left_behind
        self.no_transverse = no_transverse
        self.no_factors = no_factors
        self._cosmo = cosmo

        if exact_growth and dim != 3:
            raise NotImplementedError("`exact_growth` is currently only implemented in 3D!")

        if no_transverse and dim == 1:
            warnings.warn("`no_transverse' has no effect in one dimension!")

        if no_mode_left_behind and dim != 2:
            raise NotImplementedError("`no_mode_left_behind` is currently only implemented in 2D!")

        if batched:
            assert with_jax, "Batched LPT computation is only supported with JAX!"

        # Fourier grid
        self.k_dict = get_fourier_grid(self.shape, boxsize=boxsize, full=False, relative=False,
                                       dtype_num=dtype_num, with_jax=False)  # always use Numpy
        self.k_vecs, k_raw = self.k_dict["k_vecs"], self.k_dict["|k|"]
        self.k = set_0_to_val(dim, k_raw, 1.0)

        # Extended Fourier grid for dealiasing
        self.res_ext = (res * 3) // 2
        self.shape_ext = dim * [self.res_ext]
        self.k_dict_ext = get_fourier_grid(self.shape_ext, boxsize=boxsize, full=False, relative=False,
                                           dtype_num=dtype_num, with_jax=False)  # always use Numpy
        self.k_vecs_ext, k_raw_ext = self.k_dict_ext["k_vecs"], self.k_dict_ext["|k|"]
        self.k_ext = set_0_to_val(dim, k_raw_ext, 1.0)

        self.psi = {}

    def __str__(self):
        return (f"NLPT ({self.dim}D) \n"
                f"res: {self.res} \n"
                f"boxsize: {self.boxsize} \n"
                f"n_order: {self.n_order} \n"
                f"grad_kernel_order: {self.grad_kernel_order} \n"
                f"with_jax: {self.with_jax} \n"
                f"try_to_jit: {self.try_to_jit} \n"
                f"dtype_num: {self.dtype_num} \n"
                f"dtype_c_num: {self.dtype_c_num} \n"
                f"batched: {self.batched} \n"
                f"no_mode_left_behind: {self.no_mode_left_behind} \n"
                f"no_transverse: {self.no_transverse} \n"
                f"no_factors: {self.no_factors}")

    def __repr__(self):
        return (f"NLPT(dim={self.dim}, res={self.res}, boxsize={self.boxsize}, n_order={self.n_order}, "
                f"grad_kernel_order={self.grad_kernel_order}, with_jax={self.with_jax}, "
                f"try_to_jit={self.try_to_jit}, dtype_num={self.dtype_num}, dtype_c_num={self.dtype_c_num}, "
                f"batched={self.batched}, no_mode_left_behind={self.no_mode_left_behind}, "
                f"no_transverse={self.no_transverse}, no_factors={self.no_factors})")

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten the PyTree."""
        children = (self.psi,)
        aux_data = {
            "dim": self.dim,
            "res": self.res,
            "boxsize": self.boxsize,
            "n_order": self.n_order,
            "grad_kernel_order": self.grad_kernel_order,
            "dtype_num": self.dtype_num,
            "dtype_c_num": self.dtype_c_num,
            "with_jax": self.with_jax,
            "try_to_jit": self.try_to_jit,
            "batched": self.batched,
            "no_mode_left_behind": self.no_mode_left_behind,
            "no_transverse": self.no_transverse,
            "no_factors": self.no_factors,
            "cosmo": self._cosmo,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "NLPT":
        """Reconstruct the PyTree from its flattened representation."""
        instance = cls(**aux_data)
        instance.psi = children[0]
        return instance

    def compute(self, phi_ini: AnyArray):
        """Compute method, will be overwritten by each child class.

        :param phi_ini: initial potential
        """
        pass

    def evaluate(self, a: AnyArray | float, n_order: int | None = None, q: AnyArray | None = None,
                 include_psi_0: bool = False, time_derivative: bool = False, D_derivative: bool = False,
                 eulerian_acc: bool = False, exact_growth: bool = False) -> AnyArray:
        r"""Evaluate the LPT displacement / velocity / position field at the given scale factor and LPT order.

        :param a: the scale factor at which to evaluate the LPT displacement field
        :param n_order: the order of the LPT displacement field to evaluate
        :param q: the Lagrangian positions (only required if include_psi_0 is True)
        :param include_psi_0: whether to include the zeroth order displacement field (i.e. the initial positions)
            if True -> positions, if False -> displacements
        :param time_derivative: whether to evaluate the time derivative of the LPT displacement field
            (w.r.t. superconformal time)
        :param D_derivative: whether to evaluate the derivative of the LPT displacement field w.r.t. the growth factor
        :param eulerian_acc: whether to evaluate the Eulerian acceleration field instead of the displacement field,
                given by (2/3 a^2 \partial_a^2 + a \partial_a) psi (only for EdS).
                NOTE: it is NOT checked whether the cosmology is EdS, so this is up to the user!
        :param exact_growth: whether to use the exact 3LPT displacement field, i.e. accounting for the different
            growth factors for the different 3LPT contributions.
        :return: the LPT displacement / position field at the given scale factor and LPT order
        """

        q_shape = self.shape + [self.dim]

        # Compute growth factor at a
        a_1d = jnp.atleast_1d(a)

        D = self._cosmo.Dplus(a_1d).astype(self.dtype)

        if time_derivative:  # note: w.r.t. superconformal time!
            dD_over_dsuperconft_by_D = (self._cosmo.growth_rate(a_1d) * self._cosmo.E(a_1d) * a_1d ** 2
                                        ).astype(self.dtype)
            time_fac = self._cosmo.E(a_1d) * a_1d ** 3  # d/dsuperconft = H a^3 d/da
            suffix = "da"
        elif D_derivative:
            time_fac = 1.0 / self._cosmo.get_interpolated_property(a_1d, "a", "Dplusda")
            suffix = "da"
        else:
            time_fac = 1.0
            suffix = ""

        if n_order > self.n_order:
            warnings.warn(f"Selected n_order = {n_order}, but LPT has only been computed up to order {self.n_order}! "
                          f"Using n_order = {self.n_order} instead.")
            n_order = self.n_order

        # LPT with exact growth (up to 3rd order)
        if exact_growth:
            assert any("ex" in k for k in self.psi.keys()), "LPT has not been computed for exact growth factors!"
            assert not eulerian_acc, "Exact growth factors are not implemented for Eulerian accelerations!"

            out = q if include_psi_0 else jnp.zeros(q_shape, dtype=self.dtype)
            if len(a_1d) > 1:
                out = out[None, :]  # add time dimension

            if n_order >= 1:
                this_fac = time_fac * self._cosmo.get_interpolated_property(a_1d, "a", "Dplus" + suffix)
                out += self.psi["psi_1"] * this_fac
            if n_order >= 2:
                this_fac = time_fac * self._cosmo.get_interpolated_property(a_1d, "a", "D2plus" + suffix)
                out += self.psi["psi_2_ex"] * this_fac
            if n_order == 3:
                this_fac_a = time_fac * self._cosmo.get_interpolated_property(a_1d, "a", "D3plusa" + suffix)
                this_fac_b = time_fac * self._cosmo.get_interpolated_property(a_1d, "a", "D3plusb" + suffix)
                this_fac_c = time_fac * self._cosmo.get_interpolated_property(a_1d, "a", "D3plusc" + suffix)
                out += (self.psi["psi_3a_ex"] * this_fac_a
                        + self.psi["psi_3b_ex"] * this_fac_b
                        + self.psi["psi_3c_ex"] * this_fac_c)
            if n_order > 3:
                raise NotImplementedError

        # LPT with EdS approximation for growth
        else:
            def scan_body(carry, xs):
                n, psi = xs

                if time_derivative:
                    fac = n * D ** n * dD_over_dsuperconft_by_D
                elif D_derivative:
                    fac = n * D ** (n - 1)
                elif eulerian_acc:
                    fac = (n / 3) * (1 + 2 * n) * D ** n
                else:
                    fac = D ** n

                if len(a_1d) > 1:
                    rearrange_str_D = "() " * (self.dim + 1)
                    carry += rearrange(fac, "a -> a " + rearrange_str_D) * psi[None, :]
                else:
                    carry += fac * psi

                return carry, None

            # Support for multiple a's (along new dimension 0)
            if len(a_1d) > 1:
                out_shape = [len(a_1d)] + q_shape
                if include_psi_0:
                    q_reshaped = q[None, ...]
            else:
                out_shape = q_shape
                if include_psi_0:
                    q_reshaped = q

            out = jnp.zeros(out_shape, dtype=self.dtype)

            psi_keys = [f"psi_{n}" for n in range(1, n_order + 1)]
            assert all(key in self.psi.keys() for key in psi_keys), \
                ("Key not found in LPT psi dictionary! Make sure that 'compute_lpt()' has been called() with "
                 "'exact_growth=False!'")

            out, _ = jax.lax.scan(scan_body, out,(jnp.arange(1, n_order + 1),
                                                  jnp.asarray([self.psi[key] for key in psi_keys])))

        if include_psi_0:
            out += q_reshaped

        return out
