import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
import numpy as onp
from functools import partial
from itertools import product
from jax.scipy import ndimage
from scipy import ndimage as ndimage_sp
from einops import rearrange
from collections.abc import Callable
from .disco_nufft import disconufft3d1r
from .kernels import inv_mak_kernel
from .grids import get_fourier_grid, get_lagrangian_grid_vectors
from .utils import scan_fct_np, add_to_indices, set_indices_to_val, preprocess_fixed_inds
from .types import AnyArray
from ..cosmology.cosmology import Cosmology

__all__ = ['scatter', 'gather', 'deconvolve_mak',
           'interpolate_field', 'fourier_interpolate_field', 'linearly_interpolate_field',
           'spawn_interpolated_particles', 'get_shift_vecs_for_sheet_interpolation',
           'compute_field_quantity_from_particles']


def comoving_to_grid_space(pos, res, boxsize, with_jax=True):
    np = jnp if with_jax else onp
    return np.mod(pos, boxsize) / boxsize * res


def grid_to_comoving_space(pos, res, boxsize, with_jax=True):
    np = jnp if with_jax else onp
    return jnp.mod(pos, res) / res * boxsize


def get_stencil(dim: int, dtype: type, worder: int = 2) -> (AnyArray, int):
    baselist = {
        1: [0],
        2: [0, 1],
        3: [-1, 0, 1],
        4: [-1, 0, 1, 2]
    }.get(worder)
    if baselist is None:
        raise NotImplementedError("Supported worder values are 2 (CIC), 3 (TSC), and 4 (PCS).")
    stencil = onp.array(list(product(*((baselist,) * dim))), dtype=dtype)[None]
    return stencil, stencil.shape[1]


# Define kernel and gradient functions for each mass assignment scheme
def ngp_kernel(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 0.5, 0.0, 1.0)

def ngp_kernel_grad(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.zeros_like(abs_diff)  # NGP kernel gradient is zero everywhere

def cic_kernel(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 1.0, 0.0, 1 - abs_diff)

def cic_kernel_grad(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 1.0, 0.0, -1.0 * np.ones_like(abs_diff))

def tsc_kernel(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 1.5, 0.0, np.where(abs_diff > 0.5, 0.5 * (1.5 - abs_diff) ** 2, 0.75 - abs_diff ** 2))

def tsc_kernel_grad(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 1.5, 0.0, np.where(abs_diff > 0.5, -(1.5 - abs_diff), -2 * abs_diff))

def pcs_kernel(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 2.0, 0.0, np.where(abs_diff > 1.0, (1.0 / 6.0) * (2 - abs_diff) ** 3,
                                                  (1.0 / 6.0) * (4 - 6 * abs_diff ** 2 + 3 * abs_diff ** 3)))

def pcs_kernel_grad(abs_diff, with_jax=True):
    np = jnp if with_jax else onp
    return np.where(abs_diff > 2.0, 0.0, np.where(abs_diff > 1.0, -0.5 * (2 - abs_diff) ** 2,
                                                  (1.0 / 6.0) * abs_diff * (9 * abs_diff - 12)))


# Kernel and gradient selection based on worder
kernel_functions = {
    1: (ngp_kernel, ngp_kernel_grad),
    2: (cic_kernel, cic_kernel_grad),
    3: (tsc_kernel, tsc_kernel_grad),
    4: (pcs_kernel, pcs_kernel_grad)
}


# Main function for scattering and gathering
def scatter_or_gather(mesh: AnyArray, positions: AnyArray, do_gather: bool, n_part_tot: int, res: int, boxsize: float,
                      dtype_num: int, weight: AnyArray | None = None, index_lists: list | None = None, worder: int = 2,
                      chunk_size: int | None = None, with_jax: bool = True, use_custom_derivatives: bool = True,
                      requires_jacfwd: bool = False) -> AnyArray:
    """Scatter particle positions onto mesh or gather values from a mesh.

    :param mesh: mesh with shape (nx, ny, nz, ...)
    :param positions: particle positions with shape (n_part, dim)
    :param do_gather: if True: gather, else: scatter
    :param n_part_tot: total number of particles (needs to be provided as shape cannot be inferred at compile time)
    :param res: linear resolution (i.e. grid size per dimension)
    :param boxsize: boxsize per dimension in Mpc/h
    :param dtype_num: 32 (for float32) or 64 (for float64)
    :param weight: if not None: weight for each particle, e.g. a velocity component or another tracer
    :param index_lists: if scattering on a slice or a skewer: list of indices can be provided, where each list
        element is a list of indices for each dimension that maps global indices to local indices:
        for full dimensions: np.arange(res), for thin slice: only 0s (in slice) and 1s (outside)
    :param worder: order of the mass assignment (1: NGP (not diff'able!), 2: CIC, 3: TSC, 4: PCS)
    :param chunk_size: scatter/gather in chunks of this size. If None: all at once
    :param with_jax: use Jax? (default: True)
    :param use_custom_derivatives: use custom JVP / VJP instead of automatic ones (default: True)
    :param requires_jacfwd: if True: use custom JVP implementation rather than custom VJP implementation
    :return: data on mesh (shape: [nx, ny, nz, ...]) if gather=False
        data at particle positions (shape: (n_part_tot, dim)) if gather=True
    """
    params = [do_gather, n_part_tot, res, boxsize, dtype_num, index_lists, worder, chunk_size, with_jax]
    if with_jax and use_custom_derivatives:
        if requires_jacfwd:
            return scatter_or_gather_jax_for_fwd(*params, mesh, positions, weight)
        else:
            params = params[1:]  # split up gather / scatter for VJP because Diffrax doesn't like the varying out shape
            if do_gather:
                return gather_jax_for_bwd(params, mesh, positions, weight)
            else:
                return scatter_jax_for_bwd(params, mesh, positions, weight)
    else:
        return scatter_or_gather_core(*params, mesh, positions, weight)


##### Implementation with custom JVP #####
@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))
def scatter_or_gather_jax_for_fwd(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists, worder, chunk_size,
                                  with_jax, mesh, positions, weight):  # mesh, positions, weight as diff'able args last
    return scatter_or_gather_core(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists, worder,
                                  chunk_size, with_jax, mesh, positions, weight)


@scatter_or_gather_jax_for_fwd.defjvp
def scatter_or_gather_jvp(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists, worder, chunk_size, with_jax,
                          primals, tangents):
    mesh, positions, weight = primals

    # For gathering:
    # 0) primal gather
    # 1) JVP w.r.t. the mesh is a gather of the tangent mesh evaluated at the same indices as the primal mesh
    # 2) JVP w.r.t. the positions is a gather with sum_d (gradient kernel * tangents) as the kernel
    # 3) Forward-mode gathering does not support weights, so no contribution here

    # For scattering:
    # 0) primal gather
    # 1) JVP w.r.t. the positions is a scatter with sum_d (gradient kernel * tangents) as the kernel onto an empty grid
    # 2) JVP w.r.t. the mesh: simply pass along the tangents
    # 3) JVP w.r.t. the weights is a scatter onto an empty grid with tangents as weights

    # Tangents for scattering w.r.t. the particles / tangents for gathering w.r.t. the mesh will be computed in the
    # core function
    tangents_to_pass = tangents[0] if do_gather else tangents[1]

    # Call primary function for primal (base) outputs and tangents w.r.t. positions
    primals_out, tangents_out = scatter_or_gather_core(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists,
                                                       worder, chunk_size, with_jax, mesh, positions, weight,
                                                       tangents=tangents_to_pass)

    # Now: need to compute the JVP w.r.t. the other component that lives in the output domain and apply product rule!
    if do_gather:
        # weight: None for gathering
        tangents_out += scatter_or_gather_core(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists, worder,
                                               chunk_size, with_jax, mesh, positions, None,
                                               tangents_wrt_out=tangents[1])
    else:
        # For scattering: tangents w.r.t. mesh propagate through and need to be added
        tangents_out += tangents[0]

    # If weights are given: need to compute the JVP w.r.t. the weights and apply product rule!
    if weight is not None:
        # Gathering
        if do_gather:
            raise NotImplementedError("Gathering with weights is not supported.")

        # Scattering is linear in the weights, so the Jacobian is constant -> JVP is a scatter with tangents as weights
        else:
            tangents_out += scatter_or_gather_core(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists,
                                                   worder, chunk_size, with_jax, jnp.zeros_like(mesh), positions,
                                                   weight=tangents[2])
    return primals_out, tangents_out


##### Implementation with custom VJP #####
@partial(jax.custom_vjp, nondiff_argnums=(0,))
def scatter_jax_for_bwd(params, mesh, positions, weight):  # diff'able arguments go last
    return scatter_or_gather_core(False, *params, mesh, positions, weight)


def scatter_fwd(params, mesh, positions, weight):
    return scatter_or_gather_core(False, *params, mesh, positions, weight), (mesh, positions, weight)


def scatter_bwd(params, primals, g):
    mesh, positions, weight = primals
    # VJPs w.r.t.: mesh, positions, weight
    vjp_mesh = g  # cotangent w.r.t. mesh propagates through
    # do_gather = True: VJP w.r.t. positions is a gather!
    vjp_pos = scatter_or_gather_core(True, *params, mesh=g, positions=positions, weight=weight, do_vjp=True)
    if weight is None:
        vjp_weight = None
    else:
        # VJP is linear in the weights, so the VJP is given by a gather of the cotangent mesh
        vjp_weight = scatter_or_gather_core(True, *params, mesh=g, positions=positions, weight=None)[:, None]
    return vjp_mesh, vjp_pos, vjp_weight


scatter_jax_for_bwd.defvjp(scatter_fwd, scatter_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def gather_jax_for_bwd(params, mesh, positions, weight):  # diff'able arguments go last
    return scatter_or_gather_core(True, *params, mesh, positions, weight)


def gather_fwd(params, mesh, positions, weight):
    return scatter_or_gather_core(True, *params, mesh, positions, weight), (mesh, positions, weight)


def gather_bwd(params, primals, g):
    mesh, positions, weight = primals
    # VJPs w.r.t.: mesh, positions, weight
    # NOTE: need to scatter the VJP onto empty mesh, not the primal mesh!
    weight_for_vjp_mesh = g[..., None]  # need to scatter cotangents onto mesh
    # do_gather = False: VJP w.r.t. mesh is a scatter
    vjp_mesh = scatter_or_gather_core(False, *params, mesh=jnp.zeros_like(mesh), positions=positions,
                                      weight=weight_for_vjp_mesh)
    # do_gather = True: VJP w.r.t. positions is a gather with the cotangents as weights
    vjp_pos = scatter_or_gather_core(True, *params, mesh=mesh, positions=positions, weight=weight_for_vjp_mesh,
                                     do_vjp=True)  # True: VJP w.r.t. positions is a gather
    vjp_weight = None  # weight must be None for gathering, so there is no VJP w.r.t. the weights
    return vjp_mesh, vjp_pos, vjp_weight


gather_jax_for_bwd.defvjp(gather_fwd, gather_bwd)


##### Core function #####
def scatter_or_gather_core(do_gather, n_part_tot, res, boxsize, dtype_num, index_lists, worder, chunk_size,
                           with_jax, mesh, positions, weight, tangents=None, tangents_wrt_out=None, unroll=1,
                           do_vjp=False):
    # JVP parameters (for forward-mode diff.)
    has_tangents = tangents is not None  # positions for scatter, mesh for gather
    has_tangents_wrt_out = tangents_wrt_out is not None  # mesh for scatter, positions for gather
    has_any_tangents = has_tangents or has_tangents_wrt_out

    if has_any_tangents:
        assert with_jax, "JVPs are only supported with Jax."

    np = jnp if with_jax else onp
    _, dim = positions.shape

    dtype = getattr(np, f"float{dtype_num}")
    kernel_eps = np.finfo(dtype).eps

    positions = comoving_to_grid_space(positions, res, boxsize, with_jax=with_jax)
    if chunk_size is None:
        chunk_size = n_part_tot
    n_chunks = onp.ceil(n_part_tot / chunk_size).astype(onp.int32)

    positions = np.expand_dims(positions, 1)
    worder_int = onp.ceil(worder).astype(onp.int8)  # this is here for possible future "fractional-order" MAKs
    stencil, n_ngbs = get_stencil(dim, dtype=dtype, worder=worder_int)
    stencil_dtype = np.int16 if int(n_part_tot ** (1 / dim)) <= 32767 else np.int32
    stencil = stencil.astype(stencil_dtype)
    weights_provided = weight is not None
    assert not (weights_provided and do_gather and not do_vjp), \
        "Weights are not supported for gather (except for VJP of scattering)."

    kernel_func, grad_func = kernel_functions[worder_int]

    def compute_kernel(pos, neighbor_coords):
        abs_diff = np.abs(pos - neighbor_coords)
        return np.prod(kernel_func(abs_diff), axis=-1)

    # Kernel for JVP: requires derivative of kernel and tangent in position space
    def compute_jvp_kernel(pos, neighbor_coords, t_):
        abs_diff = np.abs(pos - neighbor_coords)
        kernel = kernel_func(abs_diff)
        kernel_grad = grad_func(abs_diff)

        def dim_kernel(i):
            sign_term = np.sign(pos[..., i] - neighbor_coords[..., i]) * t_[:, i][:, None]
            reduction_terms = np.prod(kernel, axis=-1) / (kernel_eps + kernel[..., i])
            return sign_term * kernel_grad[..., i] * reduction_terms

        results = sum(dim_kernel(i) for i in range(pos.shape[-1]))  # this seems to be more memory-efficient than scan

        return results

    # Kernel for VJP: only requires derivative of kernel
    def compute_vjp_kernel(pos, neighbor_coords):
        abs_diff = np.abs(pos - neighbor_coords)
        kernel = kernel_func(abs_diff)
        kernel_grad = grad_func(abs_diff)

        def dim_kernel(i):
            sign_term = np.sign(pos[..., i] - neighbor_coords[..., i])
            reduction_terms = np.prod(kernel, axis=-1) / (kernel_eps + kernel[..., i])
            return sign_term * kernel_grad[..., i] * reduction_terms

        results = np.transpose(np.array([dim_kernel(i) for i in range(pos.shape[-1])]), (1, 2, 0))

        return results

    def loop_body(carry, x):
        # If JVP: always need two meshes: primal and tangent
        # for the gathering, the tangent mesh will store the output
        # for the scattering, the tangent mesh is the V of the JVP
        # for VJP of the scatter (implemented as a gather): only need the primal mesh
        if has_tangents:
            m, m_tangents = carry
        else:
            m = carry

        # t: tangents of particle positions are required for scattering / gathering JVPs w.r.t. particles
        # if this is not a JVP, t will contain zeros
        pos, w, i, t = x

        int_coord = np.floor(pos).astype(stencil_dtype) if worder_int % 2 == 0 else np.round(pos).astype(stencil_dtype)
        neighbor_coords = int_coord + stencil

        if do_gather and (do_vjp or has_tangents_wrt_out):
            kernel = compute_vjp_kernel(pos, neighbor_coords)  # for VJP of scattering, kernel gradient is needed
        else:
            kernel = compute_kernel(pos, neighbor_coords)

        if has_tangents and not do_gather:  # for scattering, tangent output will contain kernel gradient * tangent
            kernel_grad = compute_jvp_kernel(pos, neighbor_coords, t)

        if weights_provided:
            kernel *= w
            if has_tangents and not do_gather:
                kernel_grad *= w

        # Map neighbors inside the box
        neighbor_coords = np.mod(neighbor_coords, res)

        # GATHER
        if do_gather:
            index_tuple = tuple(np.split(neighbor_coords, indices_or_sections=dim, axis=-1))

            # Gather is called as the VJP of the scatter
            if do_vjp:
                tangents_out = np.sum(m[index_tuple] * kernel, axis=-2)  # sum over the neighbors
                return carry, tangents_out

            # Actual gather
            else:

                # If this is a JVP of gather w.r.t. the particle positions
                if has_tangents_wrt_out:
                    tangents_out = np.sum(np.sum(m[index_tuple] * kernel * t[:, None, :], axis=-1), axis=-1)
                    return carry, tangents_out
                else:
                    primals_out = np.sum(m[index_tuple][..., 0] * kernel, axis=-1)

                # If this is a JVP of gather w.r.t. the grid values
                if has_tangents:
                    # The gathering is linear in the grid-values, so the Jacobian is constant and given by the
                    # tangent grid evaluated at the same indices as the primal grid
                    tangents_out = np.sum(m_tangents[index_tuple][..., 0] * kernel, axis=-1)
                    return carry, (primals_out, tangents_out)
                else:
                    # pass on the mesh in carry, and store the interpolated values at the particles in this chunk
                    return carry, primals_out

        # SCATTER
        else:
            # If scattering on a slice or a skewer: convert from global to local indices
            if index_lists is not None:
                # elements of index_lists need to be Jax arrays in case neighbor_coords is a tracer
                if any(isinstance(i, onp.ndarray) for i in index_lists) and with_jax:
                    local_index_lists = jax.tree.map(jnp.asarray, index_lists)
                else:
                    local_index_lists = index_lists

                neighbor_coords = np.stack([i[neighbor_coords[:, :, d]] for d, i in zip(range(dim), local_index_lists)],
                                           axis=-1)

                # np.add.at does not support `mode="drop", so we need to drop the invalid indices manually without jax
                if not with_jax:
                    neighbor_coords = neighbor_coords.reshape([-1, dim])
                    valid_indices = np.ones(neighbor_coords.shape[0], dtype=np.bool_)
                    for d in range(dim):
                        valid_indices &= (neighbor_coords[:, d] < carry.shape[d])
                    neighbor_coords = neighbor_coords[valid_indices]
                    kernel = kernel.reshape(-1)[valid_indices]

            if with_jax:
                dnums = jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=tuple(range(dim)),
                    scatter_dims_to_operand_dims=tuple(range(dim)))

                # pass on the updated mesh in carry, no need to store anything
                m = lax.scatter_add(m, neighbor_coords, kernel.reshape([-1, n_ngbs]), dnums, mode="drop")
                # mode="drop" ignores everything outside slice/skewer

                if has_tangents:
                    # For scattering, tangent output will contain kernel gradient * tangent
                    m_tangents = lax.scatter_add(m_tangents, neighbor_coords, kernel_grad.reshape([-1, n_ngbs]),
                                                 dnums, mode="drop")
                    return (m, m_tangents), None
                else:
                    return m, None
            else:
                neighbor_coords_tuple = tuple([nc.squeeze(-1) for nc in np.split(neighbor_coords,
                                                                                 indices_or_sections=dim, axis=-1)])

                np.add.at(m, neighbor_coords_tuple, kernel)
                return m, None

    # Define scan function
    scan_fct = partial(jax.lax.scan, unroll=unroll) if with_jax else partial(scan_fct_np, multiple_xs=True)

    # GATHER
    if do_gather:
        out_raw = scan_fct(loop_body,
                           mesh if not has_tangents else (mesh, tangents) if not do_vjp else tangents,  # initial carry
                           (np.reshape(positions, (n_chunks, -1, 1, dim)),  # xs[0]: positions
                            np.ones(n_chunks, dtype=dtype),  # xs[1]: weights
                            np.arange(n_chunks),  # xs[2]: chunk indices
                            np.zeros(n_chunks, dtype=dtype) if not has_tangents_wrt_out
                            else np.reshape(tangents_wrt_out, (n_chunks, -1, dim))  # xs[3]: tangents
                            )
                           )[1]
        if has_tangents:
            return (np.reshape(out_raw[0], -1), np.reshape(out_raw[1], -1))
        elif do_vjp:
            rhomean = n_part_tot / res ** dim
            if weights_provided:
                return np.reshape(out_raw, (-1, dim)) * res / boxsize * weight
            else:
                return np.reshape(out_raw, (-1, dim)) * res / boxsize * 1 / rhomean
        else:
            if has_tangents_wrt_out:
                return np.reshape(out_raw, -1) * res / boxsize
            else:
                return np.reshape(out_raw, -1)  # return ys, reshaped from n_chunk x chunk_size to n_part_tot

    # SCATTER
    else:
        out = scan_fct(loop_body,
                       (mesh, np.zeros_like(mesh)) if has_tangents else mesh,  # initial carry
                       (np.reshape(positions, (n_chunks, -1, 1, dim)),  # xs[0]: positions
                        np.asarray(np.split(weight, n_chunks, axis=0)) \
                            if weights_provided else np.ones(n_chunks, dtype=dtype),  # xs[1]: weights
                        np.arange(n_chunks),  # xs[2]: chunk indices
                        np.zeros(n_chunks, dtype=dtype) if not has_tangents
                        else np.reshape(tangents, (n_chunks, -1, dim))  # xs[3]: tangents
                        )
                       )[0]  # return carry, which contains the mesh

        if weights_provided:  # in this case, the quantity is not necessarily the density, so we don't normalize
            if has_tangents:
                return out[0], out[1] * res / boxsize  # primal and tangent mesh
            else:
                return out
        else:
            rhomean = n_part_tot / res ** dim
            if has_tangents:
                return out[0] / rhomean - 1.0, out[1] / rhomean * res / boxsize
                # for the tangents: no - 1, and need to account for normalization
            else:
                return out / rhomean - 1.0


def scatter(mesh: AnyArray, positions: AnyArray, n_part_tot: int, res: int, boxsize: float, dtype_num: int,
            weight: AnyArray | None = None, worder: int = 2, index_lists: list | None = None, with_jax: bool = True,
            chunk_size: int | None = None, use_custom_derivatives: bool = True, requires_jacfwd: bool = False) \
        -> AnyArray:
    """Wrapper around scatter_or_gather for scattering.
    """
    return scatter_or_gather(do_gather=False, mesh=mesh, positions=positions, n_part_tot=n_part_tot, res=res,
                             boxsize=boxsize, dtype_num=dtype_num, weight=weight, worder=worder,
                             index_lists=index_lists, chunk_size=chunk_size, with_jax=with_jax,
                             use_custom_derivatives=use_custom_derivatives, requires_jacfwd=requires_jacfwd)


def gather(mesh: AnyArray, positions: AnyArray, n_part_tot: int, res: int, boxsize: float, dtype_num: int,
           worder: int = 2,
           with_jax: bool = True, chunk_size: int | None = None, use_custom_derivatives: bool = True,
           requires_jacfwd: bool = False) -> AnyArray:
    """Wrapper around scatter_or_gather for gathering.
    """
    return scatter_or_gather(do_gather=True, mesh=mesh, positions=positions, n_part_tot=n_part_tot, res=res,
                             boxsize=boxsize, dtype_num=dtype_num, weight=None, worder=worder, chunk_size=chunk_size,
                             with_jax=with_jax, use_custom_derivatives=use_custom_derivatives,
                             requires_jacfwd=requires_jacfwd)


def deconvolve_mak(field: AnyArray, double_exponent: bool = False, worder: int = 2, with_jax: bool = True) -> AnyArray:
    """Divide by mass assignment kernel (MAK) in Fourier space.

    :param field: input CIC/TSC/PCS-scattered field (in real space)
    :param double_exponent: if True: exponent is doubled, to account for back- and forth interpolation
    :param worder: 2: CIC, 3: TSC, 4: PCS
    :param with_jax: use Jax?
    :return: deconvolved field
    """
    np = jnp if with_jax else onp
    res_cic_shape = field.shape
    dtype = field.dtype
    dtype_num = np.finfo(dtype).bits
    k_vecs = get_fourier_grid(res_cic_shape, relative=False, full=False, sparse_k_vecs=True, with_jax=with_jax,
                              dtype_num=dtype_num)["k_vecs"]
    field_k = np.fft.rfftn(field)
    cic_comp = inv_mak_kernel(k_vecs, double_exponent=double_exponent, worder=worder, account_for_shotnoise=False,
                              with_jax=with_jax)
    field_k_deconv = cic_comp * field_k
    return jnp.fft.irfftn(field_k_deconv).astype(dtype)


def interpolate_field(dim: int, field: AnyArray, dshift: list | AnyArray, boxsize: float, dtype_num: int,
                      which: str = "fourier", with_jax: bool = True) -> AnyArray:
    """Interpolate a field to a new set of positions using Fourier or piecewise linear interpolation.

    :param dim: dimensionality of the field
    :param field: field to be interpolated, can be scalar field (res, res, ...) or vector field (res, res, ..., d)
    :param dshift: 1D vector containing the shift vector in units of grid cells
    :param boxsize: size of the box in Mpc/h
    :param dtype_num: number indicating the precision (32 or 64)
    :param which: "fourier" or "linear"
    :param with_jax: use Jax?
    :return: new interpolated particle positions indexed in Q space
    """
    np = jnp if with_jax else onp
    dtype = getattr(np, f"float{dtype_num}")

    # Scalar field
    if field.ndim == dim:
        field = field[..., None]
        is_scalar_field = True
    # Vector field
    elif field.ndim == dim + 1:
        is_scalar_field = False
    # Wrong dimensionality
    else:
        raise ValueError("Field has wrong dimensionality.")

    shape = field.shape[:-1]
    assert onp.all([s == shape[0] for s in shape]), "Field must be cubic."
    res = shape[0]
    field_dim = field.shape[-1]

    assert dim == len(dshift)

    if which == "fourier":
        k_vecs = get_fourier_grid(shape, boxsize=boxsize, sparse_k_vecs=False, full=False,
                                  relative=True, dtype_num=dtype_num, with_jax=False)["k_vecs"]
        k_vecs = [kk / (2.0 * onp.pi) for kk in k_vecs]
        kd_prod_sum = sum([kk * d for (kk, d) in zip(k_vecs, dshift)]).astype(dtype)
        interpolated_field = np.moveaxis(np.asarray(
            [np.fft.irfftn(np.fft.rfftn(field[..., d]) * np.exp(1j * 2 * np.pi * kd_prod_sum))
             for d in range(field_dim)]), 0, -1)

    elif which == "linear":
        q_vecs = get_lagrangian_grid_vectors(dim, res=res, boxsize=1.0, with_jax=False)  # relative units!
        q_shift = np.asarray([dshift[d] * boxsize / res for d in range(dim)])
        q_grid_shifted_normalized = np.moveaxis(np.asarray(np.meshgrid(
            *([(q_vecs[d].reshape(-1) + q_shift[d]) / boxsize * res for d in range(dim)]), indexing="ij")),
            0, -1).reshape(-1, dim)

        if with_jax:
            # mode 'grid-wrap' is not yet available in Jax -> need to manually pad and use mode 'wrap'
            field_padded = [np.pad(field[..., d], tuple([(0, 1) for _ in range(dim)]), mode='wrap')
                            for d in range(field_dim)]
            interpolated_field = np.moveaxis(np.asarray([ndimage.map_coordinates(field_padded[d],
                                                                                 q_grid_shifted_normalized.T, order=1,
                                                                                 mode='wrap')
                                                         for d in range(field_dim)]), 0, -1).reshape(field.shape)
        else:
            interpolated_field = np.moveaxis(np.asarray([ndimage_sp.map_coordinates(field[..., d],
                                                                                    q_grid_shifted_normalized.T,
                                                                                    order=1, mode='grid-wrap',
                                                                                    prefilter=False)
                                                         for d in range(dim)]), 0, -1).reshape(field.shape)
    else:
        raise ValueError("'which' must be either 'fourier' or 'linear'.")

    return interpolated_field[..., 0] if is_scalar_field else interpolated_field


def fourier_interpolate_field(dim: int, field: AnyArray, dshift: list | AnyArray, boxsize: float, dtype_num: int,
                              with_jax: bool = True) -> AnyArray:
    """Wrapper around interpolate_field for Fourier interpolation.
    """
    return interpolate_field(dim=dim, field=field, which="fourier", dshift=dshift, boxsize=boxsize,
                             dtype_num=dtype_num, with_jax=with_jax)


def linearly_interpolate_field(dim: int, field: AnyArray, dshift: list | AnyArray, boxsize: float, dtype_num: int,
                               with_jax: bool = True) -> AnyArray:
    """Wrapper around interpolate_field for piecewise linear interpolation.
    """
    return interpolate_field(dim=dim, field=field, which="linear", dshift=dshift, boxsize=boxsize,
                             dtype_num=dtype_num, with_jax=with_jax)


def spawn_interpolated_particles(positions: AnyArray, q: AnyArray, dshift: list | AnyArray, boxsize: float,
                                 dtype_num: int, linear: bool = False, with_jax: bool = True) -> AnyArray:
    """Generate a new set of particles by Fourier interpolation of the dark matter sheet to position q+dshift.

    :param positions: particle positions indexed in Q space (i.e. in mesh shape)
    :param q: Lagrangian positions
    :param dshift: 1D vector containing the shift vector in units of grid cells
    :param boxsize: size of the box in Mpc/h
    :param dtype_num: number indicating the precision (32 or 64)
    :param linear: use piecewise linear interpolation instead of Fourier interpolation
    :param with_jax: use Jax?
    :return: new interpolated particle positions indexed in Q space
    """
    np = jnp if with_jax else onp
    dtype = getattr(np, f"float{dtype_num}")
    dim = positions.shape[-1]
    psi = np.mod(positions - q + boxsize / 2, boxsize) - boxsize / 2  # wrap inside [-L/2, L/2)
    qshifts = np.asarray([boxsize / positions.shape[i] * dshift[i] for i in range(dim)], dtype=dtype)

    # Piecewise linear interpolation
    if linear:
        # Interpolate the displacement field psi from q to q+dshift
        new_psi = linearly_interpolate_field(dim, psi, dshift, boxsize, dtype_num, with_jax=with_jax)

    # Fourier interpolation
    else:
        new_psi = fourier_interpolate_field(dim, psi, dshift, boxsize, dtype_num, with_jax=with_jax)

    qshift_expand_inds = tuple([None] * dim + [slice(None)])
    return np.mod(q + new_psi + qshifts[qshift_expand_inds] + boxsize, boxsize).astype(dtype)  # wrap inside [0, L)


def get_shift_vecs_for_sheet_interpolation(dim: int, n_resample: int) -> onp.ndarray:
    """Get a NumPy array of shift vectors for sheet interpolation for a given resampling factor.

    :param dim: dimension
    :param n_resample: resampling factor (e.g. 2 means each particle is interpolated to 2^dim new particles)
    :return: array of shift vectors
    """
    d_vec = onp.linspace(0, 1, n_resample, endpoint=False)
    grid = onp.meshgrid(*((d_vec,) * dim), indexing="ij")
    return onp.asarray([g.flatten() for g in grid]).T  # (0, 0, ...)


def compute_field_quantity_from_particles(pos: Array, res: int, boxsize: float, cosmo: Cosmology, requires_jacfwd: bool,
                                          q: AnyArray, quantity: Array | None = None, method: str = "pm",
                                          worder: int = 2, deconvolve: bool = False, n_resample: int = 1,
                                          antialias: int = 0, chunk_size: int | None = None,
                                          eps_nufft: float | None = None, kernel_size_nufft: int | None = None,
                                          fixed_inds: list | tuple | None = None, try_to_jit: bool = True,
                                          normalize_by_density: bool = True, in_redshift_space: bool = False,
                                          vel: Array | None = None, a: float | Array | None = None,
                                          average_radial_velocity: bool = False, radial_dim: int = -1,
                                          apply_fun_before_mapping_to_redshift_space:
                                          Callable[[Array], Array] | None = None) -> Array:
    """This method computes an arbitrary field quantity given at the positions of the particles.
    It also supports resampling the field with sheet-based interpolation, antialiasing and deconvolution of the
    mass assignment kernel. Further, it can be used to compute the density (contrast) by setting quantity=None.
    If a quantity is provided, the field will be computed as (F * rho) / rho, where F is the quantity and rho the
    density field (note that this requires a sufficient number of particles per cell such that rho is not 
    zero anywhere). The field can be computed on a slice or skewer by setting fixed_inds.        
    
    :param pos: particle positions, flat indexing
    :param res: resolution of the density field
    :param boxsize: size of the box in Mpc/h
    :param cosmo: cosmology object
    :param requires_jacfwd: whether forward-mode autodiff is required
    :param q: Lagrangian positions
    :param quantity: field quantity to compute (if None, the density field is computed)
    :param method: method to use for computing the field ("pm" or "nufftpm")
    :param worder: order of the MAK (2: CIC, 3: TSC, 4: PCS)
    :param deconvolve: whether to deconvolve the MAK
    :param n_resample: number of times to resample the gravity-source particles with sheet-based interpolation
    :param antialias: antialiasing factor (0: no antialiasing, 1: 1xAA with interlaced grid, 2: 2xAA, 3: 3xAA)
    :param eps_nufft: accuracy of NUFFT (if None: 1e-10 for double, 2e-6 for single precision)
    :param kernel_size_nufft: size of the kernel, should be even for now. Specify EITHER kernel_size OR eps.    
    :param chunk_size: size of the chunks of particles for which the computation is performed at once
           (if None, the whole array is used)
    :param fixed_inds: list of indices, one per dimension. If the index is None, all indices along the dimension are
        used. If the index is an integer, the density field is sliced at that index w.r.t. to the grid of size
        res ** dim. If the index is a 2-tuple, the first element is the starting index and the second element is
        the ending index of the slice.
        NOTE: the combination of fixed_inds and n_resample > 1 currently requires a non-JIT context!
    :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to True.
    :param normalize_by_density: whether to normalize the field by the density field, i.e. compute (F * rho) / rho 
        instead of (F * rho), where F is the quantity and rho the density field (defaults to True). 
        NOTE: this only affects the case where quantity is not None!
    :param in_redshift_space: whether to compute the field in redshift space
    :param vel: particle velocities, flat indexing (only used if in_redshift_space=True)
    :param a: scale factor (only used if in_redshift_space=True)
    :param average_radial_velocity: when computing the field in redshift space, there are two options for the
        velocity: either the velocity of the particle is used, or the average velocity of the particles in the cell
        is used. This parameter controls which option is used.
    :param radial_dim: dimension along which the redshift space distortion is applied
    :param apply_fun_before_mapping_to_redshift_space: function to apply to the field before mapping to redshift
        space (only used if in_redshift_space=True and average_radial_velocity=True)!
    :return: array with the field quantity
    """
    dim = pos.shape[-1]
    dtype = pos.dtype
    dtype_num = jnp.finfo(dtype).bits
    dtype_c = getattr(jnp, f"complex{2 * dtype_num}")
    periodic_wrap = lambda v: jnp.mod(v + boxsize, boxsize)

    if quantity is not None:
        assert pos.shape[0] == quantity.shape[0], "pos and quantity must have the same number of particles!"

    n_part_tot = pos.shape[0]
    n_part = int(onp.round(n_part_tot ** (1 / dim)))

    deposit_on_slice_or_skewer = fixed_inds is not None

    if method == "pm":
        scatter_fun_pm = partial(jax.jit,
                                 static_argnames=("with_jax", "dtype_num", "chunk_size", "worder", "n_part_tot",
                                                  "res", "boxsize", "index_lists", "requires_jacfwd"))(
            scatter) if try_to_jit else scatter

    elif method == "nufftpm":
        assert dim == 3, "NUFFT is only supported for 3D fields!"
        assert not deposit_on_slice_or_skewer, "Depositing a field on a slice or skewer with NUFFT is not supported!"
        assert not (in_redshift_space and average_radial_velocity), ("Redshift space distortions with average radial "
                                                                     "velocity with NUFFT are currently not supported!")

        def scatter_fun_nufftpm(x, weight):
            scale_to_0_2pi = lambda x_: periodic_wrap(x_) / boxsize * 2 * jnp.pi
            x_scaled_to_0_2pi = scale_to_0_2pi(x)  # Map points into [0, 2pi]^dim
            if eps_nufft is None:
                if kernel_size_nufft is not None:
                    assert kernel_size_nufft % 2 == 0, "kernel_size must be even!"
                    gamma = 0.976
                    sigma = 1.25
                    eps = onp.exp(-(gamma * onp.pi * onp.sqrt(-(gamma ** 2) + (1 - 2 * sigma) ** 2)
                                    * (kernel_size_nufft - 1)) / (2 * sigma))
                else:
                    eps = 1e-10 if dtype_num == 64 else 1e-6

            if weight is None:
                weight_provided = False
                weight = jnp.ones(x.shape[0], dtype=dtype)
            else:
                weight_provided = True

            fieldf = (disconufft3d1r(x=x_scaled_to_0_2pi, f=weight, N=res, eps=eps, sigma=1.25,
                                     chunk_size=chunk_size, with_jax=True)).astype(dtype_c)
            # Cut off upper Nyquist frequency, as it's double in DiscoNUFFT, shift DC mode to the beginning, and irfftn
            field_out = jnp.fft.irfftn(jnp.fft.ifftshift(fieldf[:-1, :-1, :], axes=(0, 1)))

            # if no weights provided: get delta = rho / rho_mean - 1, consistent with behavior of CIC etc.
            if not weight_provided:
                field_out = field_out / field_out.mean() - 1.0
            return field_out
    else:
        raise ValueError("Method must be either 'pm' or 'nufftpm'!")

    if deposit_on_slice_or_skewer:
        assert not (in_redshift_space and average_radial_velocity), \
            "average_radial_velocity is not supported when computing slices / skewers in redshift space!"
        if n_resample > 1:
            assert not isinstance(pos, jax.core.Tracer), \
                ("Depositing a field on a slice or skewer together with n_resample > 1 inside a JIT context "
                 "is currently not supported!")

    if apply_fun_before_mapping_to_redshift_space is not None:
        assert in_redshift_space and average_radial_velocity, \
            "apply_fun_before_mapping_to_redshift_space is only supported when computing fields in " \
            "redshift space with average_radial_velocity=True!"

    fixed_inds, was_integer_before = preprocess_fixed_inds(dim, fixed_inds)

    def field_internal(pos_local, quantity_local):
        # Prepare everything for density computation on slice or skewer
        if deposit_on_slice_or_skewer:
            def get_nearby_point_mask(p, return_above_and_below_masks=False):
                assert len(fixed_inds) == dim, "fixed_inds must have length equal to the dimension of the box!"
                # how far away from the slice do we consider particles?
                max_dist_to_consider = 4 * worder * boxsize / res
                # (4 is a safety factor because Fourier interpolation is not monotonic)

                mask = jnp.ones_like(p[..., 0], dtype=jnp.bool_)
                if return_above_and_below_masks:
                    mask_a = jnp.zeros_like(p[..., 0], dtype=jnp.bool_)
                    mask_b = jnp.zeros_like(p[..., 0], dtype=jnp.bool_)

                for d, i in zip(range(dim), fixed_inds):
                    # unsliced dimension
                    if i is None:
                        continue
                    # if the slice is infinitesimally thin, only need to compute one distance
                    elif i[1] == i[0] + 1:
                        d_coord_at_i = boxsize * i[0] / res
                        periodic_dist_in_d_dimension = jnp.abs(
                            jnp.mod(p[..., d] - d_coord_at_i + boxsize / 2, boxsize)
                            - boxsize / 2)
                        mask &= (periodic_dist_in_d_dimension <= max_dist_to_consider)
                        if return_above_and_below_masks:
                            new_above_mask = (jnp.mod(p[..., d] - d_coord_at_i, boxsize)
                                              < boxsize / 2)
                            mask_a |= new_above_mask
                            mask_b |= ~new_above_mask

                    # if the slice is thick, need to compute two distances
                    else:
                        d_coord_1 = boxsize * i[0] / res
                        d_coord_2 = boxsize * (i[1] - 1) / res
                        periodic_dist_in_d_dimension_1 = jnp.abs(
                            jnp.mod(p[..., d] - d_coord_1 + boxsize / 2, boxsize)
                            - boxsize / 2)
                        periodic_dist_in_d_dimension_2 = jnp.abs(
                            jnp.mod(p[..., d] - d_coord_2 + boxsize / 2, boxsize)
                            - boxsize / 2)
                        # Also check if the particle is WITHIN the slice of finite thickness
                        mask &= ((periodic_dist_in_d_dimension_1 < max_dist_to_consider)
                                 | (periodic_dist_in_d_dimension_2 < max_dist_to_consider)
                                 | jnp.logical_and(d_coord_2 >= periodic_wrap(p[..., d]),
                                                   periodic_wrap(p[..., d]) >= d_coord_1))  # in slice
                        # particle is within half a boxsize above upper edge
                        new_above_mask = jnp.mod(p[..., d] - d_coord_2, boxsize) < boxsize / 2
                        # particle is within half a boxsize below lower edge
                        new_below_mask = jnp.mod(d_coord_1 - p[..., d], boxsize) < boxsize / 2
                        mask_a |= new_above_mask
                        mask_b |= new_below_mask

                if return_above_and_below_masks:
                    return mask, mask_a, mask_b
                else:
                    return mask

            # In each dimension, get the indices of the grid points that are close to the slice
            # note: this function uses NumPy
            def get_index_lists():
                i_lists = []
                for d in range(dim):
                    # Unsliced dim.
                    if fixed_inds[d] is None:
                        i_list_dim = onp.arange(res, dtype=onp.int16)
                    else:
                        dim_length = fixed_inds[d][1] - fixed_inds[d][0]
                        i_list_dim = dim_length * onp.ones(res, dtype=jnp.int16)
                        # dim_length is one larger than the largest index -> out of bounds
                        i_list_dim = set_indices_to_val(i_list_dim, onp.arange(fixed_inds[d][0],
                                                                               fixed_inds[d][1]),
                                                        onp.arange(0, dim_length, dtype=onp.int16))
                    i_lists.append(i_list_dim)
                return i_lists

            n_inds_per_dim_list = tuple([res if i is None else i[1] - i[0] for i in fixed_inds])

        # Without resampling
        if n_resample <= 1:
            if deposit_on_slice_or_skewer:
                mesh_to_scatter = jnp.zeros(tuple(n_inds_per_dim_list), dtype=dtype)
                index_lists = get_index_lists()
            else:
                mesh_to_scatter = jnp.zeros((res,) * dim, dtype=dtype)
                index_lists = None

            # if computing it in redshift space with particle velocities: move particles to redshift space
            if in_redshift_space and not average_radial_velocity:
                pos_local = add_to_indices(pos_local, (slice(None), radial_dim),
                                           vel[:, radial_dim] / (a ** 2 * cosmo.E(a)))

            # Compute the density field on the mesh
            if method == "pm":
                field_loc = scatter_fun_pm(mesh_to_scatter, pos_local, n_part_tot=n_part_tot, res=res, boxsize=boxsize,
                                           dtype_num=dtype_num, chunk_size=chunk_size, worder=worder,
                                           index_lists=index_lists, with_jax=True, weight=quantity_local,
                                           requires_jacfwd=requires_jacfwd)
            elif method == "nufftpm":
                field_loc = scatter_fun_nufftpm(pos_local, quantity_local)

        # With resampling
        else:
            grid_flattened = get_shift_vecs_for_sheet_interpolation(dim, n_resample)

            if deposit_on_slice_or_skewer:
                field_loc = jnp.zeros(n_inds_per_dim_list, dtype=dtype)
            else:
                field_loc = jnp.zeros([res] * dim, dtype=dtype)

            # If depositing onto slice / skewer: size of interpolated particle sets is not constant, so need to
            # first compute the interpolated particles, then compute the density
            if deposit_on_slice_or_skewer:
                # When using sheet-based resampling, we need to include particles if there is the chance that
                # particles spawned from their vicinity might need to be deposited -> check all neighbors
                roll_inds = jnp.asarray(list(product(*((range(-1, 2),) * dim))))  # all neighbors
                initial_mask = jnp.zeros((n_part,) * dim, dtype=jnp.bool_)
                initial_mask_above = jnp.zeros((n_part,) * dim, dtype=jnp.bool_)
                initial_mask_below = jnp.zeros((n_part,) * dim, dtype=jnp.bool_)

                # if in redshift space with particle velocities: need to perform the check for the particles
                # in redshift space!
                if in_redshift_space and not average_radial_velocity:
                    pos_check = add_to_indices(pos_local, (slice(None), radial_dim),
                                               vel[:, radial_dim] / (a ** 2 * cosmo.E(a)))
                else:
                    pos_check = pos_local

                def find_relevant_particles(carry, i_roll):
                    # carry contains a mask of the particles for which at least one roll is close to the slice
                    p, mask, old_m_a, old_m_b = carry
                    p_rolled = jnp.roll(p, i_roll, axis=tuple(range(dim)))
                    new_m, new_m_a, new_m_b = get_nearby_point_mask(p_rolled,
                                                                    return_above_and_below_masks=True)
                    return (p, new_m | mask, new_m_a | old_m_a, new_m_b | old_m_b), None

                pos_carry = pos_check.reshape((n_part,) * dim + (dim,))
                mask_for_relevant_particles, m_a, m_b = jax.lax.scan(find_relevant_particles,
                                                                     (pos_carry,
                                                                      initial_mask, initial_mask_above,
                                                                      initial_mask_below),
                                                                     roll_inds)[0][
                                                        1:]  # [0]: carry -> [1]: mask

                mask_for_relevant_particles |= (m_a & m_b)

            def body_fun(d_loc):
                # interpolate the particle positions
                X_sheet = spawn_interpolated_particles(
                    jnp.reshape(pos_local, ((n_part,) * dim + (dim,))),
                    q=q, dshift=d_loc, boxsize=boxsize, dtype_num=dtype_num, linear=False, with_jax=True)

                # interpolate the particle quantity
                if quantity_local is not None:
                    quantity_sheet = fourier_interpolate_field(
                        dim, jnp.reshape(quantity_local, ([n_part] * dim + [1])),
                        d_loc, boxsize, dtype_num, with_jax=True)
                else:
                    quantity_sheet = None

                # if in redshift space with particle velocities: move particles to redshift space
                if in_redshift_space and not average_radial_velocity:
                    vel_sheet = fourier_interpolate_field(
                        dim, jnp.reshape(vel[:, radial_dim],
                                         ([n_part] * dim + [1])),
                        d_loc, boxsize, dtype_num, with_jax=True)
                    X_sheet = add_to_indices(X_sheet, tuple([slice(None)] * dim + [radial_dim]),
                                             vel_sheet[..., 0] / (a ** 2 * cosmo.E(a)))

                if deposit_on_slice_or_skewer:
                    # Only keep the particles that are close to the slice
                    X_sheet = X_sheet[mask_for_relevant_particles, :]
                    # Only keep the corresponding quantities
                    if quantity_sheet is not None:
                        quantity_sheet = quantity_sheet[mask_for_relevant_particles, :]
                    mesh = jnp.zeros(tuple(n_inds_per_dim_list), dtype=dtype)
                    i_lists = get_index_lists()

                else:
                    mesh = jnp.zeros((res,) * dim, dtype=dtype)
                    i_lists = None

                # Paint the sheet
                X_sheet_flat = X_sheet.reshape(-1, dim)
                if quantity_sheet is not None:
                    quantity_sheet_flat = quantity_sheet.reshape(-1, 1)
                else:
                    quantity_sheet_flat = None

                if method == "pm":
                    return scatter(mesh, X_sheet_flat, n_part_tot, res, boxsize, dtype_num=dtype_num,
                                   worder=worder, index_lists=i_lists, chunk_size=chunk_size,
                                   with_jax=True, weight=quantity_sheet_flat, requires_jacfwd=requires_jacfwd)
                elif method == "nufftpm":
                    return scatter_fun_nufftpm(X_sheet_flat, quantity_sheet_flat)

            def scan_fun(carry, dd):
                return carry + body_fun(dd), None

            field_loc = jax.lax.scan(scan_fun, field_loc, grid_flattened)[0]

            # Divide by the number of resamplings
            field_loc /= (n_resample ** dim)

        return field_loc

    # Make sure quantity has a shape that is compatible with the particle positions (N x 1)
    if quantity is not None:
        if quantity.ndim == 1:
            quantity = quantity[:, None]

    # if is redshift space and we want to average the radial velocity, compute the shifts
    # (using the same resampling settings etc.)
    if in_redshift_space and average_radial_velocity:
        v_rad = compute_field_quantity_from_particles(pos=pos, res=res, boxsize=boxsize, cosmo=cosmo,
                                                      requires_jacfwd=requires_jacfwd, q=q,
                                                      quantity=vel[:, radial_dim: radial_dim + 1], method=method,
                                                      worder=worder, n_resample=n_resample, antialias=antialias,
                                                      chunk_size=chunk_size, fixed_inds=None, in_redshift_space=False)
        shift = v_rad / (a ** 2 * cosmo.E(a))
        dx = boxsize / res
        meshgrid_vec = (onp.tile(onp.arange(res) * dx, [dim, 1])).astype(dtype)
        q_mesh = rearrange(jnp.asarray(onp.meshgrid(*meshgrid_vec, indexing="ij")),
                           "d ... -> ... d")
        q_shifted = jnp.copy(q_mesh)
        q_shifted = periodic_wrap(set_indices_to_val(q_shifted,
                                                     tuple([slice(None)] * dim + [radial_dim]),
                                                     q_shifted[..., radial_dim] + shift))

        # Compute the field with weights "1" as the normalization will be computed after mapping to redshift space
        # NOTE: if a function is applied before mapping to redhdshift space, it should be applied to delta, not rho,
        # so in this case, leave the quantity as None!
        if quantity is None and apply_fun_before_mapping_to_redshift_space is None:
            quantity_feed = jnp.ones_like(pos[:, :1])
        else:
            quantity_feed = quantity
        field = field_internal(pos, quantity_feed)

    # Compute the field
    else:
        field = field_internal(pos, quantity)

    # if a field other than the density is requested, compute the density and divide by it
    if quantity is not None and normalize_by_density:
        rho_field = compute_field_quantity_from_particles(pos=pos, res=res, boxsize=boxsize, cosmo=cosmo,
                                                          requires_jacfwd=requires_jacfwd, q=q,
                                                          quantity=jnp.ones((n_part_tot, 1), dtype=dtype),
                                                          method=method, worder=worder, n_resample=n_resample,
                                                          antialias=antialias, chunk_size=chunk_size,
                                                          fixed_inds=fixed_inds, try_to_jit=try_to_jit,
                                                          normalize_by_density=False,
                                                          in_redshift_space=in_redshift_space, vel=vel, a=a,
                                                          average_radial_velocity=False, radial_dim=radial_dim)
        field /= rho_field

    if antialias:
        assert fixed_inds is None, "Antialiasing is not implemented for slices/skewers."
        shift = boxsize / (2 * res)
        k_vecs_for_res = \
            get_fourier_grid((res,) * dim, boxsize, sparse_k_vecs=True, full=False, relative=False,
                             dtype_num=dtype_num, with_jax=False)["k_vecs"]
        field_shifted = jnp.fft.irfftn(
            jnp.fft.rfftn(field_internal(pos + shift, quantity))
            * jnp.exp(1j * shift * sum(k_vecs_for_res)))
        field = (field + field_shifted) / 2

    # Note: for NUFFT, deconvolution is always done as part of NUFFT, so no need to do it here
    if deconvolve and method == "pm":
        assert fixed_inds is None, "Deconvolution is not implemented for slices/skewers."
        field_out = deconvolve_mak(field, double_exponent=False, worder=worder, with_jax=True)
    else:
        field_out = field

    # if is redshift space and we want to average the radial velocity, map field to redshift space with
    # the computed shifts
    # TODO: resampling should also be applied in this step!
    if in_redshift_space and average_radial_velocity:
        if apply_fun_before_mapping_to_redshift_space is not None:
            field_out = apply_fun_before_mapping_to_redshift_space(field_out)
        assert not onp.any(onp.isnan(q_shifted)), "NaNs in q_shifted!"
        field_out = scatter_fun_pm(mesh=jnp.zeros([res] * dim, dtype=dtype),
                                   positions=jnp.reshape(q_shifted, (-1, dim)),
                                   res=res,
                                   n_part_tot=res ** dim,
                                   boxsize=boxsize,
                                   dtype_num=dtype_num,
                                   weight=jnp.reshape(field_out, (-1, 1)),
                                   with_jax=True,
                                   worder=worder,
                                   requires_jacfwd=requires_jacfwd)

        # Get delta from rho
        if quantity is None and apply_fun_before_mapping_to_redshift_space is None:
            rhomean = pos.shape[0] / res ** dim
            field_out = field_out / rhomean - 1.0

    # Extract the slice or skewer (delta has length `ind[1] - ind[0]` in the sliced dimensions):
    if fixed_inds is not None:
        field_out = field_out[tuple(slice(None) if ind is None else (0 if was_int else slice(0, ind[1] - ind[0]))
                                    for (was_int, ind) in zip(was_integer_before, fixed_inds))]

    return field_out
