# Gaussian random field generation for DISCO-DJ
import jax
import jax.numpy as jnp
import numpy as onp
from ..core.utils import set_0_to_val, set_indices_to_val
from ..core.kernels import inv_laplace_kernel
from ..core.grids import enforce_hermitian_symmetry, get_fourier_grid
from jax import Array
from functools import partial
from collections.abc import Iterable
from discodj_native import rng_ngenic

__all__ = ["generate_grf", "get_ngenic_wnoise", "get_k_ordering"]


def get_ngenic_wnoise(seed: int, res: int, dtype_num: int = 32) -> Array:
    """Generates a 3D white noise field compatible with the N-GenIC random number generator.

    :param seed: the rand number generator seed
    :param res: the linear resolution (i.e. resolution per dimension)
    :param dtype_num: 32 or 64 for single or double
    :return: array with dimensions (res, res, res) holding the white noise field
    """
    dtype = getattr(jnp, f'float{dtype_num}')

    # Multiple seeds
    if isinstance(seed, Iterable):
        fields_raw = jnp.stack([rng_ngenic(s, res).get_field() for s in seed], 0)
        trafo = lambda f: jnp.fft.irfftn(f * res ** (3 / 2)).astype(dtype)
        return jax.vmap(trafo, 0, 0)(fields_raw)

    # Single seed
    else:
        rng = rng_ngenic(seed, res)
        return (jnp.fft.irfftn(rng.get_field() * res ** (3 / 2))).astype(dtype)


def generate_grf(sampling_space: str, dim: int, pk_table: dict, res: int, boxsize: float,
                 rand_key: Array | int, wavelet_type: str = "db4", wavelet_baseres: int = 16,
                 k_order_fourier: str | None = None, enforce_mode_stability: bool = False,
                 base_field: Array | None = None, sphere_mode: bool = False, with_nufft: bool = False,
                 q: Array | None = None, dtype_num: int = 32, dtype_c_num: int = 64, with_jax: bool = True,
                 try_to_jit: bool = True) -> Array:
    """
    Generates a Gaussian random field.
    If sampling space is not "wavelet", with_jax is True, and mode stability is not being enforced, uses jitted version.

    :param sampling_space: one of "real", "wavelet", or "fourier"
    :param dim: dimension of the field
    :param pk_table: power spectrum table containing k and P(k)
    :param res: resolution per dimension (given by the number of grid points)
    :param boxsize: boxsize in Mpc / h per dimension
    :param rand_key: random key (jax.random.PRNGKey) for jax, or seed (int) for numpy
    :param wavelet_type: type of wavelet, will be passed to pywt.Wavelet. Defaults to "db4".
    :param wavelet_baseres: base resolution for the wavelet-based IC generation
    :param k_order_fourier:
        "stable" for stable ordering of wave vectors (large -> small scales, stable when increasing resolution)
        "natural" for natural ordering (increasing in each dimension, with +/- modes subsequently)
        None for no ordering (default, simply reshape the random numbers to (res,) * dim + (res // 2 + 1,))
    :param enforce_mode_stability: if True: use numpy RNG that has sequential equivalence guarantee
    :param base_field: if provided, this field will be used instead of random white noise
    :param sphere_mode: if True, all modes with k > k_Nyquist are set to zero
    :param with_nufft: if True, use NUFFT for mapping to real space
    :param q: if with_nufft is True, q needs to be provided
    :param dtype_num: 32 or 64 for single or double
    :param dtype_c_num: 64 or 128 for single or double
    :param with_jax: if True: use jax.numpy, else numpy
    :param try_to_jit: if True: try to jit the function
    :return: Gaussian random field for potential phi in Fourier space (real layout)
    """
    if enforce_mode_stability and sampling_space == "fourier":
        assert k_order_fourier == "stable", "Mode stability is only possible with stable ordering of wave vectors!"

    jittable = with_jax and try_to_jit and not enforce_mode_stability and sampling_space != "wavelet"
    jit_fct = partial(jax.jit, static_argnames=("sampling_space", "dim", "res", "boxsize", "k_order_fourier",
                                                "sphere_mode", "with_nufft", "dtype_num", "dtype_c_num",
                                                "wavelet_type", "wavelet_baseres", "enforce_mode_stability")) \
        if jittable else lambda x: x

    core_fun = jit_fct(generate_grf_core)

    if with_nufft:
        assert with_jax, "NUFFT is only implemented for JAX mode!"

    general_args = {"sampling_space": sampling_space, "dim": dim, "pk_table": pk_table, "res": res, "boxsize": boxsize,
                    "k_order_fourier": k_order_fourier, "sphere_mode": sphere_mode, "with_nufft": with_nufft, "q": q,
                    "dtype_num": dtype_num, "dtype_c_num": dtype_c_num,
                    "wavelet_type": wavelet_type, "wavelet_baseres": wavelet_baseres,
                    "enforce_mode_stability": enforce_mode_stability}

    return core_fun(rand_key=rand_key, base_field=base_field, **general_args)

def get_k_ordering(k_order_fourier: str, res: int, dim: int, with_jax: bool) -> Array:
    np = jnp if with_jax else onp

    # need to re-order to FFT layout: 0, pos, neg (for all but last dimension): 0, 1, res - 1, 2, res - 2, ...
    resort_order = np.concatenate([np.asarray([0]),
                                   np.vstack([np.arange(1, res // 2 + 1),
                                              np.arange(res - 1, res // 2 - 1, -1)]).flatten(order="F")])[:-1]

    if k_order_fourier == "stable":
        # get all (kx, ky, kz, ...) index combinations
        range_tiled = np.tile(np.arange(res)[np.newaxis, :], [dim - 1, 1])
        combs = np.stack(np.meshgrid(*range_tiled, np.arange(res // 2 + 1), indexing="ij"), -1).reshape(
            -1, dim).T

        # rescale all but last index: except for the constant mode, there is -k and +k
        combs_rescaled = np.vstack([*combs[:-1, :], np.clip((combs[-1, :] * 2 - 1), 0, onp.inf)])
        combs_max = combs_rescaled.max(axis=0)

        # make a combined array with max, x, y, .. (rescaled)
        combs_with_max = np.concatenate([np.flipud(combs_rescaled), combs_max[np.newaxis, :]], axis=0)

        # get indices required to sort by max and then each dim. (lexsort starts with last dim and moves backward!)
        order = np.lexsort(combs_with_max)

        # sort the combinations
        all_combs = combs[:, order]
        all_combs_final = set_indices_to_val(np.copy(all_combs), (slice(0, -1), slice(None)),
                                             resort_order[all_combs[:-1, :]])

    elif k_order_fourier == "natural":
        all_combs_final = np.stack(np.meshgrid(*[resort_order] * (dim - 1),
                                               np.arange(res // 2 + 1), indexing="ij"), -1).reshape(-1, dim).T
    else:
        raise ValueError("k_order_fourier must be 'stable' or 'natural'!")

    return all_combs_final


# Core function for generating Gaussian random fields
def generate_grf_core(sampling_space: str, dim: int, pk_table: dict, res: int,
                      boxsize: float, rand_key: Array | int, dtype_num: int, dtype_c_num: int,
                      wavelet_type: str, wavelet_baseres: int, k_order_fourier: str | None,
                      enforce_mode_stability: bool, sphere_mode: int, base_field: Array | None = None,
                      with_nufft: bool = False, q: Array | None = None, with_jax: bool = True) -> Array:
    """Core of GRF generation."""
    np = jnp if with_jax else onp
    dtype = getattr(np, f'float{dtype_num}')
    dtype_c = np.complex64 if dtype_c_num == 64 else np.complex128

    k_dict = get_fourier_grid([res] * dim, boxsize=boxsize, sparse_k_vecs=True,
                              full=False, dtype_num=dtype_num, relative=False, with_jax=with_jax)
    k, k_vecs = k_dict["|k|"], k_dict["k_vecs"]
    Pk = np.interp(k, pk_table["k"], pk_table["Pk"])

    # Generate phi_ini in real space
    shape = dim * [res]
    if sampling_space == "real":
        if base_field is None:
            if with_jax:
                phi_ini_raw = jax.random.normal(rand_key, shape=shape).astype(dtype)
            else:
                onp.random.seed(rand_key)  # jax.random.PRNGKey(n) = [0, n]
                phi_ini_raw = onp.random.normal(0.0, 1.0, size=shape).astype(dtype)
        else:
            phi_ini_raw = np.asarray(base_field).astype(dtype)
        norm_fac = (res / boxsize) ** dim  # comoving resolution in (Mpc/h)^(-dim)
        fphi = (np.fft.rfftn(phi_ini_raw) * (Pk * norm_fac) ** 0.5
                * inv_laplace_kernel(k_vecs, with_jax=with_jax)).astype(dtype_c)

    # Generate phi_ini in wavelet space
    elif sampling_space == "wavelet":
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets is required for wavelet-based GRF generation!"
                              "See https://pywavelets.readthedocs.io/en/latest/install.html")

        assert base_field is None, "Base field option is not implemented for wavelet method!"

        wavelet = pywt.Wavelet(wavelet_type)

        # compute center of mass of wavelet
        center = 0.0
        for i, w in enumerate(wavelet.dec_lo):
            center += i * w
        center /= np.sum(np.asarray(wavelet.dec_lo))

        # check that requested resolution is a power of two times the base resolution (which can be anything)
        assert (res // wavelet_baseres) % 2 == 0
        levels = int(np.round(np.log2(res // wavelet_baseres)))

        if enforce_mode_stability or not with_jax:
            onp.random.seed(rand_key)
            phi_ini_raw = np.asarray(onp.random.normal(0.0, 1.0, size=[wavelet_baseres] * dim))
        else:
            phi_ini_raw = jax.random.normal(rand_key, shape=[wavelet_baseres] * dim)
            k1 = rand_key  # need to split key

        for ilev in range(levels):
            if not enforce_mode_stability and with_jax:
                k1, k2 = jax.random.split(k1)

            # use previous phi_ini_raw as 'coarse' input
            phi_coarse = phi_ini_raw
            rescoarse = phi_coarse.shape[0]

            # generate unconstrained 'fine' input
            if enforce_mode_stability or not with_jax:
                phi_fine = np.asarray(onp.random.normal(0.0, 1.0, size=dim * [2 * rescoarse]))
            else:
                phi_fine = jax.random.normal(k1, shape=dim * [2 * rescoarse])

            # get the fourier grid
            k_dict_coarse = get_fourier_grid([rescoarse] * dim, boxsize=boxsize, sparse_k_vecs=False, full=False,
                                             dtype_num=dtype_c, relative=False, with_jax=True)

            # shift the coarse grid to the center of mass of the wavelet
            shift = (wavelet.dec_len / 2 - center) / (2 * rescoarse)

            # shift_term: exp(1j * shift * (kkx + kky + ...))
            shift_term = np.exp(1j * shift * np.asarray([ks for ks in k_dict_coarse["k_vecs"]]).sum(0))
            phi_coarse = np.fft.irfftn(np.fft.rfftn(phi_coarse) * shift_term)

            # perform forward DWT to get the decimated coefficients
            cDict = pywt.dwtn(phi_fine, wavelet=wavelet_type, mode="periodization")

            # perform backward DWT but now using coarse grid to replace average component
            cDict_mod = cDict.copy()
            avg_key = "a" * dim
            cDict_mod[avg_key] = phi_coarse
            phi_ini_raw = pywt.idwtn(cDict_mod, wavelet=wavelet_type, mode='periodization')

        assert np.all(np.asarray(phi_ini_raw.shape) == res)

        # Multiply with sqrt(P(k)) and normalization factor
        norm_fac = (res / boxsize) ** dim  # comoving resolution in (Mpc/h)^(-dim)
        fphi = (np.fft.rfftn(phi_ini_raw) * (Pk * norm_fac) ** 0.5
                * inv_laplace_kernel(k_vecs, with_jax=with_jax)).astype(dtype_c)

    # Generate phi_ini in Fourier space
    # NOTE: mode stability is not possible with JAX! No sequential equivalence guarantee!
    elif sampling_space == "fourier":
        # total number of modes for real FFT
        n_k_tot = res ** (dim - 1) * (res // 2 + 1)

        if dim == 1:
            all_combs_final = np.arange(n_k_tot, dtype=dtype)[np.newaxis, :]
        else:
            if k_order_fourier is not None:
                all_combs_final = get_k_ordering(k_order_fourier, res, dim, with_jax=True)

        # draw real and imaginary parts from Gaussian
        if base_field is None:
            if enforce_mode_stability or not with_jax:
                onp.random.seed(rand_key)
                rand_nos_flat = np.asarray(onp.random.normal(0.0, 1.0, size=(2 * n_k_tot,))).astype(dtype)
            else:
                rand_nos_flat = jax.random.normal(key=rand_key, shape=(2 * n_k_tot,)).astype(dtype)

            # split up into re and im parts
            rand_nos_real_im = rand_nos_flat.reshape([n_k_tot, 2])

        # given white noise field
        else:
            assert base_field.shape == (n_k_tot, 2), "Base field must have dimensions 'n_k_tot x 2' for Fourier method!"
            rand_nos_real_im = np.asarray(base_field, dtype=dtype)

        # assign random values in the right order
        shape_fourier = (dim - 1) * [res] + [res // 2 + 1]

        # if no order is specified: simply assign the random numbers by reshaping the flat array to shape_fourier
        if k_order_fourier is None:
            fphi_raw = (rand_nos_real_im[:, 0] + 1j * rand_nos_real_im[:, 1]).reshape(shape_fourier).astype(dtype_c)
        else:
            fphi_raw = np.empty(shape_fourier, dtype=dtype_c)
            inds_to_set = tuple([all_combs_final[d, :] for d in range(dim)])
            fphi_raw = set_indices_to_val(fphi_raw, inds_to_set, rand_nos_real_im[:, 0] + 1j * rand_nos_real_im[:, 1])

        # impose Hermitian symmetry
        fphi_raw = enforce_hermitian_symmetry(fphi_raw, with_jax=with_jax)

        # multiply with sqrt(P(k)) and normalization factors
        norm_fac = (res / boxsize) ** dim  # comoving resolution in (Mpc/h)^(-dim)
        fphi = fphi_raw * (Pk * norm_fac) ** 0.5 * inv_laplace_kernel(k_vecs, with_jax=with_jax)
        fphi /= np.sqrt(2)  # divide by sqrt(2) (real & imag.)
        fphi *= res ** (dim / 2)  # another factor of res ** (dim / 2) because the Gaussian is drawn in Fourier space

    else:
        raise NotImplementedError("Sampling space must be either 'real', 'fourier', or 'wavelet'!")

    # Set DC term to 0
    fphi = set_0_to_val(dim, fphi, 0)

    if sphere_mode:
        # Set all modes with k > k_Nyquist to zero
        k_Nyquist = np.pi * res / boxsize
        k_corner_modes_mask = k > k_Nyquist
        fphi = np.where(k_corner_modes_mask, 0, fphi)

    if with_nufft:
        use_finufft = False

        scale_to_0_2pi = lambda x: np.fmod(boxsize + x, boxsize) / boxsize * 2 * np.pi
        q_scaled_to_0_2pi = scale_to_0_2pi(q)
        fphi_full = np.fft.fftshift(np.fft.fftn(np.fft.irfftn(fphi))).astype(dtype_c)

        if use_finufft:
            from jax_finufft import nufft2
            nufft_bw_fun = lambda f, x, eps: nufft2(f, *x, eps=eps)
            phi_nufft = (nufft_bw_fun(fphi_full, q_scaled_to_0_2pi.reshape(-1, dim).T,
                                      1e-6).real.reshape((res,) * dim) / (res ** dim))

            # Account for different ordering of FINUFFT
            if dim == 3:
                phi = np.roll(phi_nufft[::-1, ::-1, ::-1], (1, 1, 1), axis=(0, 1, 2))
            elif dim == 2:
                phi = np.roll(phi_nufft[::-1, ::-1], (1, 1), axis=(0, 1))
            else:
                phi = np.roll(phi_nufft[::-1], 1)

        else:
            assert dim == 3, \
                f"DiscoNUFFT is only implemented for 3D fields! Use FINUFFT for {dim}D fields (see grf.py)."
            from ..core.disco_nufft import disconufft3d2r, disconufft3d2
            nufft_bw_fun = lambda f, x, eps: disconufft3d2(x, f, eps=eps)
            phi = nufft_bw_fun(fphi_full, q_scaled_to_0_2pi, 1e-6).real.reshape((res,) * dim)
            # TODO: use DiscoNUFFT with real layout

        # Forward transform again
        fphi = np.fft.rfftn(phi)

    # Return phi in Fourier space
    return fphi
