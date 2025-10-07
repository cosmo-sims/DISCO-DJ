# Functions for computing acceleration
import jax
import jax.numpy as jnp
import numpy as onp
from itertools import product
import functools
from collections.abc import Callable
from ..core.scatter_and_gather import gather, scatter, spawn_interpolated_particles
from ..core.utils import set_0_to_val, set_indices_to_val, scan_fct_np
from ..core.kernels import inv_mak_kernel, gradient_kernel, inv_laplace_kernel
from ..core.types import AnyArray
from discodj_native import nbody2d, nbody3d

__all__ = ["calc_acc_exact_1d", "calc_acc_nufft", "calc_acc_PM", "calc_acc_Tree_PM", "dealias"]


def calc_acc_exact_1d(psi: AnyArray, boxsize: float, n_part: int, with_jax: bool = True) -> AnyArray:
    """Exact acceleration for 1D case (using particle sorting).
    NOTE: Do NOT differentiate through this function, as the discrete nature of the
    sorting operation will lead to non-differentiable points and incorrect results!

    :param psi: particle displacements
    :param boxsize: box size
    :param n_part: number of particles
    :param with_jax: use Jax?
    :return: acceleration field
    """
    np = jnp if with_jax else onp
    dx_for_q = boxsize / n_part
    q_vec = (np.arange(n_part) * dx_for_q).astype(psi.dtype)[:, None]
    Xs = np.fmod(psi + q_vec, boxsize)
    Xc = np.mean(Xs)
    sorted_ind = np.argsort(Xs, axis=0).flatten()
    acc = np.zeros_like(psi)
    rank = np.arange(len(psi))
    first_term = ((rank + 0.5) / len(psi))[:, np.newaxis]
    second_term = (Xs[sorted_ind] - Xc) / boxsize + 0.5
    acc = set_indices_to_val(acc, sorted_ind, - first_term + second_term) * boxsize
    return acc


def calc_acc_nufft(psi, dim: int, boxsize: float, res_pm: int, n_part: int, dtype_num: int, eps: float | None = None,
                   kernel_size: int | None = None, sigma: float = 1.25, grad_order: int = 0, n_resample: int = 1,
                   resampling_method: str = "fourier", return_on_grid: bool = False, use_finufft: bool = False,
                   chunk_size: int | None = None, with_jax: bool = True) -> AnyArray:
    """Acceleration using NUFFT.

    :param psi: particle displacements
    :param dim: dimensionality
    :param boxsize: box size
    :param res_pm: resolution of the Fourier grid per dimension
    :param n_part: number of particles per dimension
    :param dtype_num: 32 or 64 (float32 or float64)
    :param eps: accuracy of NUFFT (if None: 1e-10 for double, 2e-6 for single precision)
    :param kernel_size: size of the kernel, should be even for now. Specify EITHER kernel_size OR eps.
    :param sigma: upsampling ratio (only affects custom DiscoDJ NUFFT)
    :param grad_order: order of the gradient kernel (0: use ik, -1: use NUFFT kernel derivative)
    :param n_resample: resampling factor (1: no resampling, 2: resample particles 2x per dim., etc.)
    :param resampling_method: "fourier" or "linear"
    :param return_on_grid: if True: return acceleration on grid, else: return acceleration at particle positions
    :param use_finufft: if True: use finufft, else: use custom DiscoDJ NUFFT
    :param chunk_size: split up the computation in chunks of particles to reduce memory usage
    :param with_jax: use Jax?
    :return: acceleration field
    """
    if return_on_grid:
        raise NotImplementedError("return_on_grid is not yet implemented for NUFFT")
    if not use_finufft:
        assert with_jax, "Custom NUFFT is only implemented with JAX"
    if grad_order != 0:
        assert with_jax and not use_finufft, "FINUFFT implementation does not support kernel derivatives"

    if with_jax:
        if use_finufft:
            from jax_finufft import nufft1, nufft2
            nufft_fw_fun = lambda x, c, n, eps_: nufft1((n,) * dim, c, *(x.T),
                                                        eps=eps_)  # signature in jax_finufft is flipped!
            nufft_bw_fun = lambda f, x, eps_: nufft2(f, *(x.T), eps=eps_)
        else:
            assert dim == 3, "DiscoNUFFT is only implemented for 3D"
            from ..core.disco_nufft import disconufft3d1r, disconufft3d2r
            nufft_fw_fun = lambda x, c, n, eps_: disconufft3d1r(x, c, n, eps=eps_, sigma=sigma, chunk_size=chunk_size)
            nufft_bw_fun = lambda f, x, eps_, grad_=False: disconufft3d2r(x, f, eps=eps_, sigma=sigma,
                                                                          chunk_size=chunk_size, grad=grad_)
    else:
        import finufft
        nufft_fw = getattr(finufft, f"nufft{dim}d1")
        nufft_bw = getattr(finufft, f"nufft{dim}d2")
        nufft_fw_fun = lambda x, c, n, eps_: nufft_fw(*(x.T), c, (n,) * dim, eps=eps_)
        nufft_bw_fun = lambda f, x, eps_: nufft_bw(*(x.T), f, eps=eps_)

    np = jnp if with_jax else onp
    dx_for_q = boxsize / n_part

    assert not (eps is not None and kernel_size is not None), "Specify EITHER eps OR kernel_size, not both!"
    if eps is None:
        if kernel_size is not None:
            assert kernel_size % 2 == 0, "kernel_size must be even!"
            gamma = 0.976
            eps = onp.exp(-(gamma * onp.pi * onp.sqrt(-(gamma ** 2) + (1 - 2 * sigma) ** 2)
                            * (kernel_size - 1)) / (2 * sigma))
        else:
            eps = 1e-10 if psi.dtype == np.float64 else 1e-6

    dtype = getattr(np, f"float{dtype_num}")
    dtype_c = getattr(np, f"complex{2 * dtype_num}")
    q_vec = np.tile(np.arange(n_part) * dx_for_q, [dim, 1]).astype(dtype)
    q = np.moveaxis(np.asarray(np.meshgrid(*q_vec, indexing="ij")), 0, -1)  # q in mesh form
    X = psi + q.reshape(-1, dim)

    scale_to_0_2pi = lambda x: np.fmod(boxsize + x, boxsize) / boxsize * 2 * np.pi
    X_scaled_to_0_2pi = scale_to_0_2pi(X)  # Map points into [0, 2pi]^dim

    n_modes = res_pm

    # DiscoNUFFT uses real layout WITH DOUBLED NYQUIST MODE!
    if with_jax and not use_finufft:
        k_vecs = []
        for d in range(dim - 1):
            inds_to_expand = tuple(d * [None] + [slice(None)] + (dim - 1 - d) * [None])
            k_vecs.append((2.0 * np.pi * np.arange(-res_pm // 2, res_pm // 2 + 1) / res_pm)[inds_to_expand])
        inds_to_expand = tuple((dim - 1) * [None] + [slice(None)])
        k_vecs.append((2.0 * np.pi * np.arange(0, res_pm // 2 + 1) / res_pm)[inds_to_expand])
        k2 = sum(k ** 2 for k in k_vecs)
        k2 = set_indices_to_val(k2, (res_pm // 2,) * (dim - 1) + (0,), 1.0).astype(dtype)

    # (cu)FINUFFT needs full layout
    else:
        k_vecs = []
        for d in range(dim):
            inds_to_expand = tuple(d * [None] + [slice(None)] + (dim - 1 - d) * [None])
            k_vecs.append(onp.fft.fftfreq(n_modes, d=1.0 / (2.0 * onp.pi))[inds_to_expand])
        k2 = sum(k ** 2 for k in k_vecs)
        k2 = set_indices_to_val(k2, (0,) * dim, 1.0).astype(dtype)
        k_vecs = [onp.fft.fftshift(k).astype(dtype) for k in k_vecs]
        k2 = onp.fft.fftshift(k2).astype(dtype)

    def get_fourier_delta(X_scaled_to_0_2pi_):
        # All particles have the same mass (=1)
        dtype_nufft = dtype if (with_jax and not use_finufft) else dtype_c  # DiscoNUFFT uses real NUFFT
        c = np.ones(n_part ** dim, dtype=dtype_nufft)

        # Do non-uniform FFT
        return (nufft_fw_fun(X_scaled_to_0_2pi_, c, n_modes, eps_=eps) / n_modes).astype(dtype_c)

    def compute_acc(f, X_scaled_to_0_2pi_):
        # Compute the RHS for each dimension (Note: - from -1/k^2 and - from -grad phi give +)
        if grad_order == 0:
            rhs = np.asarray([(f * 1j * k / k2).astype(dtype_c) for k in k_vecs])  # Shape: (dim, n_modes, ...)

            # Use adjoint NUFFT
            acc = (boxsize / (n_part ** dim)
                   * np.asarray([nufft_bw_fun(rhs_k, X_scaled_to_0_2pi_, eps_=eps)
                                 for rhs_k in rhs]).T.real.astype(dtype))

        # Kernel derivative
        elif grad_order == -1:
            rhs = (f / k2).astype(dtype_c)
            acc = ((boxsize / (n_part ** dim)
                   * nufft_bw_fun(rhs, X_scaled_to_0_2pi_, eps_=eps, grad_=True).real.astype(dtype))
                   * 2.0 * np.pi / res_pm)

        else:
            raise ValueError("grad_order for NUFFT must be 0 (ik derivative) or -1 (NUFFT kernel derivative)")

        return acc

    if n_resample > 1:
        fdelta = get_fourier_delta(X_scaled_to_0_2pi)

        d = onp.linspace(0, 1, n_resample, endpoint=False)
        grid = np.meshgrid(*((d,) * dim), indexing="ij")
        grid_flattened = [g.flatten()[1:] for g in grid]  # (0, 0, ...) is not needed

        def body_fun(d_loc):
            linear = resampling_method == "linear"
            X_sheet = spawn_interpolated_particles(np.reshape(X, ((n_part,) * dim + (dim,))), q=q,
                                                   dshift=d_loc, boxsize=boxsize, dtype_num=dtype_num,
                                                   with_jax=with_jax, linear=linear).reshape(X.shape)
            return get_fourier_delta(scale_to_0_2pi(X_sheet))

        if with_jax:
            def scan_fun(carry, dd):
                return carry + body_fun(dd), None

            fdelta = jax.lax.scan(scan_fun, fdelta, grid_flattened)[0]

        else:
            grid_flattened_for_map = zip(*grid_flattened)
            fdelta += sum(map(body_fun, grid_flattened_for_map))

        fdelta /= (n_resample ** dim)
        return compute_acc(fdelta, X_scaled_to_0_2pi)

    else:
        fdelta = get_fourier_delta(X_scaled_to_0_2pi)
        return compute_acc(fdelta, X_scaled_to_0_2pi)


def calc_acc_PM_or_Tree_PM(with_tree: bool, psi: AnyArray, dim: int, res_pm: int, n_part: int, k: AnyArray | None,
                           k_vecs: AnyArray | list | tuple, boxsize: float, antialias: int, grad_order: int,
                           lap_order: int, dtype_num: int, alpha: float | None = None, theta: float | None = None,
                           n_resample: int = 1, resampling_method: str = "fourier", worder: int = 2,
                           deconvolve: bool = False, short_or_long: str | None = None, chunk_size: int | None = None,
                           return_on_grid: bool = False, with_jax: bool = True, try_to_jit: bool = False,
                           use_custom_derivatives: bool = True, requires_jacfwd: bool = False) -> AnyArray:
    """Compute the acceleration using the PM or the TreePM method.

    :param with_tree: if True: use TreePM, else: use PM
    :param psi: particle displacements
    :param dim: dimensionality
    :param res_pm: resolution of the grid per dimension
    :param n_part: number of particles per dimension
    :param k: modulus of wave numbers
    :param k_vecs: wave numbers in each dimension
    :param boxsize: box size
    :param antialias: if > 0: use interlaced grid for antialiasing
    :param grad_order: order of the gradient kernel
    :param lap_order: order of the Laplacian kernel
    :param dtype_num: 32 or 64 (float32 or float64)
    :param alpha: tree force splitting scale
    :param theta: opening angle for tree force
    :param n_resample: resampling factor (1: no resampling, 2: resample particles 2x per dim., etc.).
        This sheet-based resampling suppresses discreteness effects by artificially increasing the number of
        gravity-source particles through interpolation.
    :param resampling_method: "fourier" or "linear"
    :param worder: order of the window function (2: CIC, 3: TSC, 4: PCS)
    :param deconvolve: if True: deconvolve the window function from the force
    :param short_or_long: if "short" / "long": only compute use short-range / long-range force
        with cutoff defined by alpha
    :param chunk_size: split up the computation in chunks of particles to reduce memory usage
    :param return_on_grid: if True: return acceleration on grid (only PM),
        else: return acceleration at particle positions
    :param with_jax: use Jax?
    :param try_to_jit: if True: try to JIT at the level of this function (default: False)
    :param use_custom_derivatives: if True: use custom derivatives for scatter & gather (default), else: JAX's default
    :param requires_jacfwd: whether support for jax.jacfwd is required (default: False).
    :return: acceleration field
    """
    np = jnp if with_jax else onp
    dtype = getattr(np, f"float{dtype_num}")

    if return_on_grid:
        assert not with_tree, "return_on_grid is only implemented for PM, not for TreePM"

    # Add Lagrangian coordinate q to displacement that is currently stored in X
    dx_for_q = boxsize / n_part
    q_vec = np.tile(np.arange(n_part) * dx_for_q, [dim, 1]).astype(dtype)
    q = np.moveaxis(np.asarray(np.meshgrid(*q_vec, indexing="ij")), 0, -1)  # q in mesh form
    X = psi + q.reshape(-1, dim)

    if short_or_long is not None:
        assert k is not None, "k must be provided for short_or_long!"

    @dealias
    def calc_acc_PM_(X_):
        if n_resample > 1:
            delta = scatter(np.zeros((res_pm,) * dim, dtype=dtype), X_, n_part ** dim, res_pm, boxsize,
                            dtype_num=dtype_num, with_jax=with_jax, worder=worder, chunk_size=chunk_size,
                            requires_jacfwd=requires_jacfwd, use_custom_derivatives=use_custom_derivatives)

            d = onp.linspace(0, 1, n_resample, endpoint=False)
            grid = np.meshgrid(*((d,) * dim), indexing="ij")
            grid_flattened = [g.flatten()[1:] for g in grid]  # (0, 0, ...) is not needed

            def body_fun(d_loc):
                linear = resampling_method == "linear"
                X_sheet = spawn_interpolated_particles(np.reshape(X_, ((n_part,) * dim + (dim,))), q=q,
                                                       dshift=d_loc, boxsize=boxsize, dtype_num=dtype_num,
                                                       with_jax=with_jax, linear=linear).reshape(X_.shape)
                return scatter(np.zeros((res_pm,) * dim, dtype=dtype), np.reshape(X_sheet, X_.shape),
                               n_part ** dim, res_pm, boxsize, dtype_num=dtype_num, worder=worder,
                               chunk_size=chunk_size, with_jax=with_jax, requires_jacfwd=requires_jacfwd,
                               use_custom_derivatives=use_custom_derivatives)

            if with_jax:
                def scan_fun(carry, dd):
                    return carry + body_fun(dd), None

                delta = jax.lax.scan(scan_fun, delta, grid_flattened)[0]

            else:
                grid_flattened_for_map = zip(*grid_flattened)
                delta += sum(map(body_fun, grid_flattened_for_map))

            delta /= (n_resample ** dim)

        else:
            delta = scatter(np.zeros((res_pm,) * dim, dtype=dtype), X_, n_part ** dim, res_pm, boxsize,
                            dtype_num=dtype_num, worder=worder, chunk_size=chunk_size, with_jax=with_jax,
                            requires_jacfwd=requires_jacfwd, use_custom_derivatives=use_custom_derivatives)

        fdelta_raw = np.fft.rfftn(delta)
        del delta

        # kernel is defined with negative power -> multiply for de-convolution!
        # NOTE: double_exponent = True (once for scatter, once for gather)
        if deconvolve:
            fdelta_raw *= inv_mak_kernel(k_vecs, account_for_shotnoise=False, double_exponent=True, worder=worder,
                                         with_jax=with_jax)
        fdelta = set_0_to_val(dim, fdelta_raw, 0.0)
        del fdelta_raw

        # Now, compute the acceleration by solving the Poisson equation and interpolating back to the particles
        def get_acc_per_dim(grad):
            if with_tree or short_or_long == "long":
                long_short_kernel = np.exp(-0.5 * k ** 2 * alpha ** 2)
            elif short_or_long == "short":
                long_short_kernel = 1.0 - np.exp(-0.5 * k ** 2 * alpha ** 2)
            else:
                long_short_kernel = 1.0

            facc = - grad * long_short_kernel * inv_laplace_kernel(k_vecs, order=lap_order, with_jax=with_jax) * fdelta

            if return_on_grid:
                return np.fft.irfftn(facc)
            else:
                return gather(np.fft.irfftn(facc), X_, n_part_tot=n_part ** dim, res=res_pm, boxsize=boxsize,
                              dtype_num=dtype_num, worder=worder, chunk_size=chunk_size, with_jax=with_jax,
                              requires_jacfwd=requires_jacfwd, use_custom_derivatives=use_custom_derivatives)

        # Get the gradient kernels (note: this implementation assumes the same resolution & boxsize in all dimensions!)
        gradients = [gradient_kernel(k_vecs, axis=d, order=grad_order, with_jax=with_jax) for d in range(dim)]

        def scan_fun_acc(carry, direction):
            shape_for_x = [1] + [res_pm] * (dim - 2) + [res_pm // 2 + 1] * (dim > 1)
            shape_for_y = [res_pm] + [1] + [res_pm // 2 + 1] * (dim > 2)
            shape_for_z = [res_pm] * 2 + [1]
            shapes = [shape_for_x, shape_for_y, shape_for_z]

            # noinspection PyUnboundLocalVariable
            g_fun_list = [lambda d=d: np.tile(gradients[d], shapes[d]) for d in range(dim)]
            # note: default d is needed here to avoid late-binding behavior of lambda functions

            if with_jax:
                g = jax.lax.switch(direction, g_fun_list)
            else:
                g = g_fun_list[direction]()

            return None, get_acc_per_dim(g)

        if with_jax:
            accs = jax.lax.scan(scan_fun_acc, None, np.arange(dim))[1].T
        else:
            accs = scan_fct_np(scan_fun_acc, None, np.arange(dim))[1].T

        return accs

    # TreePM
    if with_tree:
        if with_jax and try_to_jit:
            AX = functools.partial(jax.jit, static_argnames=("dim_", "res_pm_", "dtype_num_", "antialias_",
                                                             "with_jax_"))(calc_acc_PM_)(
                X_=X, dim_=dim, res_pm_=res_pm, boxsize_=boxsize, dtype_num_=dtype_num, antialias_=antialias,
                with_jax_=with_jax)
        else:
            AX = calc_acc_PM_(X_=X, dim_=dim, res_pm_=res_pm, boxsize_=boxsize, dtype_num_=dtype_num,
                              antialias_=antialias, with_jax_=with_jax)

        # short range: always double precision for now
        if dim == 2:
            nb = nbody2d(onp.asarray(X, dtype=np.float64), boxsize)
            AX -= np.asarray(nb.get_acc_M2P(theta, alpha), dtype=dtype)
        elif dim == 3:
            nb = nbody3d(onp.asarray(X, dtype=np.float64), boxsize)
            AX -= np.asarray(nb.get_acc_M2P(theta, alpha), dtype=dtype)
        else:
            raise NotImplementedError


    # PM only
    else:
        # k and alpha are not needed in this case
        AX = calc_acc_PM_(X_=X, dim_=dim, res_pm_=res_pm, boxsize_=boxsize, dtype_num_=dtype_num,
                          antialias_=antialias, with_jax_=with_jax)

    # Return total acceleration
    return AX


def calc_acc_PM(psi: AnyArray, dim: int, res_pm: int, n_part: int, k: AnyArray | None, k_vecs: AnyArray | list | tuple,
                boxsize: float, antialias: int, grad_order: int, lap_order: int, dtype_num: int,
                alpha: float | None = None, n_resample: int = 1, resampling_method: str = "fourier", worder: int = 2,
                deconvolve: bool = False, short_or_long: str | None = None, chunk_size: int | None = None,
                return_on_grid: bool = False, with_jax: bool = True, try_to_jit: bool = False,
                use_custom_derivatives: bool = True, requires_jacfwd: bool = False) -> AnyArray:
    """Compute the acceleration using the PM method, wrapper around calc_acc_PM_or_Tree_PM.
    """
    return calc_acc_PM_or_Tree_PM(with_tree=False, psi=psi, dim=dim, res_pm=res_pm, n_part=n_part, k=k, k_vecs=k_vecs,
                                  grad_order=grad_order, lap_order=lap_order, boxsize=boxsize, antialias=antialias,
                                  dtype_num=dtype_num, alpha=alpha, theta=None, n_resample=n_resample,
                                  resampling_method=resampling_method, worder=worder,
                                  deconvolve=deconvolve, short_or_long=short_or_long, chunk_size=chunk_size,
                                  return_on_grid=return_on_grid, with_jax=with_jax, try_to_jit=try_to_jit,
                                  use_custom_derivatives=use_custom_derivatives, requires_jacfwd=requires_jacfwd)


def calc_acc_Tree_PM(psi: AnyArray, dim: int, res_pm: int, n_part: int, k: AnyArray, k_vecs: AnyArray | list | tuple,
                     boxsize: float, antialias: int, grad_order: int, lap_order: int, dtype_num: int, alpha: float,
                     theta: float, n_resample: int = 1, resampling_method: str = "fourier", worder: int = 2,
                     deconvolve: bool = True, chunk_size: int | None = None, with_jax: bool = True,
                     try_to_jit: bool = False, use_custom_derivatives: bool = True, requires_jacfwd: bool = False)\
        -> AnyArray:
    """Compute the acceleration using the TreePM method, wrapper around calc_acc_PM_or_Tree_PM.
    """
    if isinstance(psi, jax.core.Tracer) or try_to_jit:
        raise NotImplementedError("TreePM is not yet implemented with JAX; can't be used in JIT mode!")
    return calc_acc_PM_or_Tree_PM(with_tree=True, psi=psi, dim=dim, res_pm=res_pm, n_part=n_part, k=k, k_vecs=k_vecs,
                                  grad_order=grad_order, lap_order=lap_order, boxsize=boxsize, antialias=antialias,
                                  dtype_num=dtype_num, alpha=alpha, theta=theta, n_resample=n_resample,
                                  resampling_method=resampling_method, worder=worder, deconvolve=deconvolve,
                                  chunk_size=chunk_size, with_jax=with_jax, try_to_jit=try_to_jit,
                                  use_custom_derivatives=use_custom_derivatives, requires_jacfwd=requires_jacfwd)


def dealias(func: Callable) -> Callable:
    """Decorator to dealias the input field by calling the function with shifted grids and averaging the results.

    :param func: acceleration function to be decorated
    :return: dealiased acceleration function
    """

    @functools.wraps(func)
    def wrapper_dealias(**kwargs):
        assert onp.all([k in kwargs.keys() for k in ("X_", "dim_", "res_pm_", "boxsize_", "dtype_num_", "antialias_",
                                                     "with_jax_")])
        with_jax = kwargs.pop("with_jax_")
        np = jnp if with_jax else onp

        dim = kwargs.pop("dim_")
        res_pm = kwargs.pop("res_pm_")
        boxsize = kwargs.pop("boxsize_")
        dtype_num = kwargs.pop("dtype_num_")
        antialias = kwargs.pop("antialias_")
        X = kwargs.pop("X_")  # X will potentially be shifted by this wrapper

        dhalf = 0.5 * boxsize / res_pm

        # no antialiasing
        if antialias == 0:
            return func(X_=X, **kwargs)

        # with -1, the particles will be shifted by a half grid spacing in each direction
        elif antialias == -1:
            shift_list = [(dhalf,) * dim]

        # 1x AA: only (0, 0, 0) and (0.5, 0.5, 0.5)
        elif antialias == 1:
            shift_list = [(0,) * dim, (dhalf,) * dim]

        # 2x AA: (0, 0, 0), (0, 0, 0.5), (0, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0), (0.5, 0, 0.5), (0.5, 0.5, 0), (0.5, 0.5, 0.5)
        elif antialias == 2:
            shift_list = list(product(*([0, dhalf],) * dim))

        # 3x AA: same, but also with quarter shifts
        elif antialias == 3:
            shift_list = list(product(*([0, dhalf],) * dim))
            shift_list += list(product(*([-dhalf / 2, dhalf / 2],) * dim))

        else:
            raise NotImplementedError

        def body_fun(s_loc):
            s_loc_arr = np.expand_dims(s_loc, 0)
            return func(X + s_loc_arr, **kwargs)

        dtype = np.float32 if dtype_num == 32 else np.float64
        shift_list_arr = np.asarray(shift_list, dtype=dtype)

        if with_jax:
            def scan_fun(carry, s):
                return carry + body_fun(s), None

            return jax.lax.scan(scan_fun, np.zeros_like(X), shift_list_arr)[0] / len(shift_list)

        else:
            shift_list_arr_for_map = list(shift_list_arr)
            return sum(map(body_fun, shift_list_arr_for_map)) / len(shift_list)

    return wrapper_dealias
