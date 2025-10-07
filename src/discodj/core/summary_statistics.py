import jax
import jax.numpy as jnp
from jax import Array
import numpy as onp
from collections.abc import Iterable
from ..core.grids import get_fourier_grid
from ..core.utils import set_indices_to_val
from ..core.types import AnyArray

__all__ = ['power_spectrum', 'cross_power_spectrum']


def power_spectrum_core(fourier_fields: AnyArray | list[AnyArray], cellsize: float, bins: int | tuple = 16,
                        dtype_num: int = 32, compute_std: bool = False, logarithmic: bool = True,
                        k_effective: bool = True, full_shape: bool = False, with_jax: bool = True) \
        -> tuple[AnyArray, AnyArray, AnyArray | None]:
    """Core of the power spectrum calculation.

    :param fourier_fields: complex field (or list of complex fields)
    :param cellsize: size of a single cell in field in Mpc/h
    :param bins: number of bins or alternatively tuple of bin edges
    :param dtype_num: 32 or 64 (float32 or float64)
    :param compute_std: compute standard deviation of power in each bin (default: False)? If not, return None.
    :param logarithmic: use logarithmic binning (default: True)?
    :param k_effective: use effective k (default: True)?
    :param full_shape: whether the input Fourier fields have full layout of complex FFT (default: False)?
    :param with_jax: use Jax?
    :return:
        kbins (the central value of the bins for plotting),
        P(k) (real valued array of power in each bin),
        P(k) std (standard deviation of power in each bin) or None if compute_std is False
    """
    np = jnp if with_jax else onp
    dtype = getattr(jnp, f'float{dtype_num}')
    if isinstance(fourier_fields, AnyArray):
        fourier_fields = [fourier_fields]
        single_field = True
    elif isinstance(fourier_fields, Iterable):
        single_field = False
    else:
        raise ValueError("fourier_fields must be an array or a list of arrays!")
    fourier_shape = fourier_fields[0].shape
    dim = len(fourier_shape)
    assert onp.all([f.shape == fourier_shape for f in fourier_fields]), "All fields must have the same shape!"
    shape = fourier_shape if full_shape else fourier_shape[:-1] + (fourier_shape[-1] * 2 - 2,)
    assert onp.all([d == shape[0] for d in shape]), \
        "Different resolutions for different dimensions are not yet supported."
    bessel_correction = True
    boxsize = shape[0] * cellsize
    kmin = 2 * np.pi / boxsize
    kmax = np.pi / cellsize

    if isinstance(bins, int):
        nbins = bins

        if logarithmic:
            dlogk = (np.log10(kmax) - np.log10(kmin)) / (nbins - 1)
            kedges = np.geomspace(kmin * 10 ** (-dlogk / 2), kmax * 10 ** (dlogk / 2),
                                  nbins + 1, dtype=dtype, endpoint=True)
            kbins = np.sqrt(kedges[1:] * kedges[:-1])
        else:
            dk = (kmax - kmin) / (nbins - 1)
            kedges = np.linspace(kmin - dk / 2, kmax + dk / 2, nbins + 1, dtype=dtype, endpoint=True)
            kbins = (kedges[1:] + kedges[:-1]) / 2

    elif isinstance(bins, (list, tuple, AnyArray)):
        kedges = onp.asarray(bins, dtype=dtype)
        assert onp.all(kedges > 0), "All bin edges must be positive!"
        assert onp.all(kedges[1:] > kedges[:-1]), "Bin edges must be strictly increasing!"
        nbins = len(kedges) - 1

        if logarithmic:
            kbins = np.sqrt(kedges[1:] * kedges[:-1])
        else:
            kbins = (kedges[1:] + kedges[:-1]) / 2

    else:
        raise ValueError("bins must be an integer or a list of bin edges!")

    # normalization for power spectra
    norm = np.asarray((boxsize ** dim) / (onp.prod(onp.asarray(shape, dtype=dtype)) ** 2), dtype)
    # get a Fourier grid
    kmag = get_fourier_grid(shape, boxsize, dtype_num=dtype_num, with_jax=True, full=full_shape)["|k|"].reshape(-1)
    dig = np.digitize(kmag, kedges)
    if onp.prod(shape) <= 2147483647:
        dig = dig.astype('int32')

    rfft_factor = np.ones_like(fourier_fields[0], dtype=int)
    rfft_factor = set_indices_to_val(rfft_factor, (..., slice(1, -1)), 2).reshape(-1)
    
    # count number of modes in each bin
    add_args = {"length": nbins + 2} if with_jax else {}
    N_modes = np.bincount(dig, minlength=nbins + 2, weights=rfft_factor, **add_args)

    # compute effective bin centers
    if k_effective:
        kbins = (np.bincount(dig, minlength=nbins + 2, weights=kmag * rfft_factor, **add_args) / N_modes)[1:-1]


    def _get_Pk(f, d, n):
        f = f.reshape(-1) * norm
        Psum = np.bincount(dig, minlength=nbins + 2, weights=rfft_factor * f, **add_args)
        # compute normalization
        P = Psum / n

        if compute_std:
            Pstd = np.zeros(nbins + 2, dtype=dtype)
            Pstd = Pstd.at[d].add((f - P[d]) ** 2)  # this is (P(k) - mean P(k)) ** 2
            std_denom = n - 1 if bessel_correction else n
            Pstd = np.sqrt(Pstd / std_denom)
        else:
            Pstd = None

        # cut off everything outside the bins
        P = P[1:-1]
        if compute_std:
            Pstd = Pstd[1:-1]

        return P, Pstd

    if with_jax:
        P, Pstd = jax.vmap(lambda fs: _get_Pk(fs, dig, N_modes), in_axes=0, out_axes=0)(np.asarray(fourier_fields))
    else:
        P, Pstd = zip(*[_get_Pk(f, dig, N_modes) for f in fourier_fields])

    if single_field:
        P = P[0]
        if compute_std:
            Pstd = Pstd[0]

    return kbins, P, Pstd


def power_spectrum(field: AnyArray, cellsize: float, bins: int | tuple = 16, with_jax: bool = True, dtype_num: int = 32,
                   compute_std: bool = False, logarithmic: bool = True) -> tuple[AnyArray, AnyArray, AnyArray | None]:
    """Calculate the power spectrum given real space field.

    :param field: real valued field
    :param cellsize: size of a single cell in field in Mpc/h
    :param bins: number of bins or alternatively tuple of bin edges
    :param with_jax: use Jax?
    :param dtype_num: 32 or 64 (float32 or float64)
    :param compute_std: compute standard deviation of power in each bin (default: False)? If not, return None.
    :param logarithmic: use logarithmic binning (default: True)?
    :return:
        kbins (the central value of the bins for plotting),
        P(k) (real valued array of power in each bin),
        P(k) std (standard deviation of power in each bin) or None if compute_std is False
    """
    dtype = getattr(jnp, f'float{dtype_num}')
    fourier_field = jnp.abs(jnp.fft.rfftn(jnp.asarray(field, dtype=dtype))) ** 2
    return power_spectrum_core(fourier_field, cellsize=cellsize, bins=bins, dtype_num=dtype_num,
                               compute_std=compute_std, logarithmic=logarithmic, with_jax=with_jax)


def cross_power_spectrum(field1: AnyArray, field2: AnyArray, cellsize: float, bins: int | tuple = 16,
                         with_jax: bool = True, dtype_num: int = 32, logarithmic: bool = True) \
        -> tuple[AnyArray, AnyArray, AnyArray, AnyArray]:
    """
    Calculate the cross-power spectrum given two real space fields.

    :param field1: first real valued field
    :param field2: second real valued field
    :param cellsize: size of a single cell in field in Mpc/h
    :param bins: number of bins or alternatively tuple of bin edges
    :param with_jax: use Jax?
    :param dtype_num: 32 or 64 (float32 or float64)
    :param logarithmic: use logarithmic binning (default: True)?
    :return:
        kbins (the central value of the bins for plotting),
        XP(k) (cross power),
        P1(k),
        P2(k)
    """
    np = jnp if with_jax else onp
    dtype = getattr(np, f'float{dtype_num}')
    assert field1.shape == field2.shape, "The two fields must have the same shape!"

    kwargs = {"cellsize": cellsize, "bins": bins, "with_jax": with_jax, "dtype_num": dtype_num, "compute_std": False,
              "logarithmic": logarithmic}
    k, XPk_raw, _ = power_spectrum_core((np.fft.rfftn(np.asarray(field1, dtype=dtype))
                                        * np.fft.rfftn(np.asarray(field2, dtype=dtype)).conj()).real,
                                        **kwargs)
    _, Pk1, _ = power_spectrum_core(np.abs(np.fft.rfftn(np.asarray(field1, dtype=dtype))) ** 2, **kwargs)
    _, Pk2, _ = power_spectrum_core(np.abs(np.fft.rfftn(np.asarray(field2, dtype=dtype))) ** 2, **kwargs)

    return k, XPk_raw / np.sqrt(Pk1 * Pk2), Pk1, Pk2


def bispectrum(field: Array, boxsize: float, bins: int | tuple = 16,  fast: bool = True, only_B: bool = True,
               only_equilateral: bool = False, compute_norm: bool = False) -> dict:
    """Evaluate the full bispectrum of a field using Jax.

    :param field: field of which to compute the bispectrum
    :param boxsize: size of the box in Mpc/h
    :param bins: number of bins for the spectrum or alternatively a tuple of bin edges
    :param fast: whether to use the fast algorithm that precomputes all bins
        (due to memory restrictions in 3D this is limited to resolutions < ~512^3 on a 40GB GPU)
    :param only_B: whether to return only the bispectrum measurements or also power spectrum and triangles.
        Should be true when trying to compute the gradient of the bispectrum!
    :param only_equilateral: if true, only compute the equilateral bispectrum
    :param compute_norm: whether to compute the normalization (triangle counts) instead of the bispectrum.
        Can be stored for repeated bispectrum calculations.
    :return: dictionary with triangle centers, Pk, Bk
    """
    dim = field.ndim
    kmin = 2 * onp.pi / boxsize
    res = field.shape[0]    
    dtype_num = onp.finfo(field.dtype).bits
    dtype_c = jnp.complex64 if dtype_num == 32 else jnp.complex128
    assert onp.all([d == res for d in field.shape]), "Different resolutions for different dimensions are not supported."
    k_dict = get_fourier_grid(shape=field.shape, boxsize=boxsize, full=False, sparse_k_vecs=True, relative=False, 
                              dtype_num=dtype_num, with_jax=False)
    k = k_dict["|k|"]
    
    if isinstance(bins, int):
        eps = onp.finfo(k.dtype).eps
        nbins = bins
        bin_edges = onp.linspace(1, res / 3 - eps, nbins + 1) * kmin
    elif isinstance(bins, (list, tuple, AnyArray)):
        bin_edges = bins
        nbins = len(bins) - 1
    else:
        raise ValueError("bins must be an integer or a list of bin edges!")

    kmax = bin_edges[-1]
    kmax_resolution = kmin * res / 3
    assert kmax < kmax_resolution, (f"kmax of the chosen binning exceeds the limit of the FFT estimator (2/3 * kNyq): "
                                    f"{kmax} > {kmax_resolution}")

    fft_axes = onp.arange(-dim, 0)

    if not compute_norm:
        assert len(field.shape) == dim
        field = jnp.fft.rfftn(field, axes=fft_axes)
    else:
        field = 1.

    triangle_centers = []
    triangle_indices = []
    for i in range(nbins):
        a = (bin_edges[i + 1] + bin_edges[i]) / 2
        wa = (bin_edges[i + 1] - bin_edges[i]) / 2
        for j in range(i + 1):
            b = (bin_edges[j + 1] + bin_edges[j]) / 2
            wb = (bin_edges[j + 1] - bin_edges[j]) / 2
            for l in range(j + 1):
                c = (bin_edges[l + 1] + bin_edges[l]) / 2
                wc = (bin_edges[l + 1] - bin_edges[l]) / 2
                if a - wa < (b + wb) + (c + wc):
                    triangle_centers.append((a, b, c))
                    triangle_indices.append((i, j, l))

    triangle_centers = onp.array(triangle_centers)
    triangle_indices = onp.array(triangle_indices)

    if only_equilateral:
        equilateral_mask = onp.all(triangle_indices == triangle_indices[:, :1], axis=1)
        triangle_centers = triangle_centers[equilateral_mask]
        triangle_indices = triangle_indices[equilateral_mask]

    bin_edges = jnp.asarray(bin_edges)

    if fast:
        bin_filters = jnp.asarray(
            [(k >= bin_edges[n]) * (k < bin_edges[n + 1]) for n in range(nbins)],
            dtype=dtype_c)
        bin_filters = bin_filters * field
        bin_filters = jnp.fft.irfftn(bin_filters, axes=fft_axes)

        _, Bk = jax.lax.scan(jax.checkpoint(
            lambda c, t: (0., (bin_filters[t[0]] * bin_filters[t[1]] * bin_filters[t[2]]).sum(fft_axes))), init=0.,
            xs=triangle_indices)

        if not only_B:
            Pk = (bin_filters ** 2).sum(fft_axes)

    else:
        triangle_indices_expanded = jnp.expand_dims(triangle_indices, range(2, dim + 2))


        def _P(carry, t):
            fields = (k >= bin_edges[t]) * (k < bin_edges[t + 1]) * field
            fields = jnp.fft.irfftn(fields, axes=fft_axes)
            return 0., (fields ** 2.).sum()


        def _B(carry, t):
            fields = (k >= bin_edges[t]) * (k < bin_edges[t + 1]) * field
            fields = jnp.fft.irfftn(fields, axes=fft_axes)
            return 0., fields.prod(0).sum()


        _, Bk = jax.lax.scan(jax.checkpoint(_B), init=0., xs=triangle_indices_expanded)

        if not only_B:
            _, Pk = jax.lax.scan(_P, init=0., xs=jnp.arange(nbins))

    if compute_norm:
        Bk = Bk * float(res) ** (2 * dim)
        if not only_B:
            Pk = Pk * float(res) ** dim
    else:
        Bk = Bk * boxsize ** (2 * dim) / float(res) ** dim
        if not only_B:
            Pk = Pk * boxsize ** dim / float(res) ** dim

    out_dict = {"k": triangle_centers, "Bk": Bk}
    if not only_B:
        out_dict["Pk"] = Pk
    return out_dict
