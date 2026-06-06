"""HEALPix shell maps and Born-approximation convergence from lightcones.

Bins lightcone crossings into a stack of HEALPix maps (one per redshift /
scale-factor shell) and turns the resulting overdensity shells into a Born
weak-lensing convergence map. All array ops are pure JAX, so map-level
observables (e.g. C_ell) stay differentiable w.r.t. the input catalogue and,
through the lensing kernel, w.r.t. cosmology.

The accumulation can be run two ways:
  - **post-hoc** on an in-memory (or batched) catalogue via
    :func:`accumulate_shell_maps`;
  - **on-the-fly** during catalogue generation by passing a :class:`MapSpec`
    to ``evaluate_lpt_lightcone_to_hdf5`` — the crossings are binned inside the
    existing chunk/replica loop, so no second pass and the particle catalogue
    need not even be written.

Conventions: shells are indexed by ascending scale factor ``a`` (so descending
redshift / descending comoving distance). Overdensity uses a full-sky mean over
all pixels in a shell — appropriate for a box-tiled lightcone whose replicas
cover the sky out to ``chi_max``.
"""

from __future__ import annotations

import numpy as onp
import jax.numpy as jnp

from ..core.healpix import nside2npix, vec2ang, ang2pix_ring, accumulate_map

__all__ = ["MapSpec", "accumulate_shell_maps", "shells_to_overdensity",
           "density_shells_to_kappa"]

# (H0/c) in (Mpc/h)^-1 — inverse of the c/H0 = 2997.92458 Mpc/h used elsewhere.
_H0_OVER_C = 1.0 / 2997.92458


class MapSpec:
    """Declarative request for a stack of HEALPix shell maps.

    :param nside: HEALPix resolution (RING scheme).
    :param a_edges: ascending scale-factor bin edges, length ``n_bins + 1``.
        Build from a redshift grid with ``a = 1/(1+z)`` (then sort ascending),
        or from ``np.geomspace(a_far, a_near, n_bins + 1)`` to match the shell
        grid the lightcone evaluator uses.
    :param weighted: if True, weight each crossing by its particle mass
        (``mass_weight`` per row); if False, produce count maps.
    """

    def __init__(self, nside: int, a_edges, weighted: bool = True):
        self.nside = int(nside)
        self.npix = nside2npix(self.nside)
        self.a_edges = onp.asarray(a_edges, dtype=onp.float64)
        assert self.a_edges.ndim == 1 and self.a_edges.size >= 2, \
            "a_edges must be a 1-D array of length n_bins + 1"
        assert onp.all(onp.diff(self.a_edges) > 0), "a_edges must be ascending"
        self.n_bins = self.a_edges.size - 1
        self.weighted = bool(weighted)

    @property
    def shape(self):
        return (self.n_bins, self.npix)


def _shell_index(a_cross, a_edges):
    """Bin each crossing's a into [0, n_bins). Out-of-range -> -1 (dropped)."""
    a_edges = jnp.asarray(a_edges, dtype=a_cross.dtype)
    n_bins = a_edges.shape[0] - 1
    idx = jnp.searchsorted(a_edges, a_cross, side="right") - 1
    in_range = (a_cross >= a_edges[0]) & (a_cross <= a_edges[-1])
    return jnp.where(in_range, jnp.clip(idx, 0, n_bins - 1), -1)


def accumulate_shell_maps(x, a_cross, observer, spec: MapSpec,
                          mass_weight=None):
    """Bin crossings into a ``(n_bins, npix)`` map stack.

    :param x: ``(M, 3)`` crossing positions (box frame, replica applied).
    :param a_cross: ``(M,)`` crossing scale factor.
    :param observer: ``(3,)`` observer position.
    :param spec: a :class:`MapSpec`.
    :param mass_weight: ``(M,)`` per-row weight (e.g. particle mass). Required
        iff ``spec.weighted``; ignored otherwise (count map).
    :return: JAX array ``(n_bins, npix)``. Differentiable in ``mass_weight``.
    """
    x = jnp.asarray(x)
    a_cross = jnp.asarray(a_cross, dtype=x.dtype)
    observer = jnp.asarray(observer, dtype=x.dtype)
    rel = x - observer[None, :]
    theta, phi = vec2ang(rel)
    pix = ang2pix_ring(spec.nside, theta, phi)            # (M,)
    sh = _shell_index(a_cross, spec.a_edges)              # (M,) in [-1, n_bins)

    if spec.weighted:
        if mass_weight is None:
            raise ValueError("spec.weighted=True requires mass_weight.")
        w = jnp.asarray(mass_weight, dtype=jnp.float32)
    else:
        w = jnp.ones(pix.shape, dtype=jnp.float32)

    # Flatten (shell, pixel) into one index; out-of-range rows -> a throwaway
    # bin n_bins*npix that we slice off, with zero weight as a belt-and-braces.
    npix = spec.npix
    flat = jnp.where(sh >= 0, sh * npix + pix, spec.n_bins * npix)
    w = jnp.where(sh >= 0, w, 0.0)
    stacked = accumulate_map(flat, w, spec.n_bins * npix + 1)
    return stacked[:-1].reshape(spec.n_bins, npix)


def shells_to_overdensity(maps):
    """Convert per-shell count/mass maps to overdensity ``delta`` maps.

    Uses each shell's own full-sky mean over all pixels: ``delta = m/mean - 1``.
    Empty shells (mean == 0) map to all-zero.
    """
    maps = jnp.asarray(maps)
    mean = jnp.mean(maps, axis=1, keepdims=True)
    mean_safe = jnp.where(mean > 0, mean, 1.0)
    return jnp.where(mean > 0, maps / mean_safe - 1.0, 0.0)


def density_shells_to_kappa(delta_shells, a_edges, cosmo, z_source):
    """Born-approximation convergence from overdensity shells.

    ``kappa(n) = (3/2) Omega_m (H0/c)^2 sum_shell
                  [chi (chi_s - chi)/chi_s] (1/a) delta_shell(n) dchi``,

    summing only shells in front of the source (``chi < chi_s``).

    :param delta_shells: ``(n_bins, npix)`` overdensity maps
        (from :func:`shells_to_overdensity`).
    :param a_edges: the ``MapSpec.a_edges`` used to build the shells.
    :param cosmo: ``Cosmology`` with timetables (uses ``chi``).
    :param z_source: source-plane redshift (scalar).
    :return: ``(npix,)`` convergence map. Differentiable in ``delta_shells``.
    """
    a_edges = jnp.asarray(a_edges)
    chi_edges = cosmo.chi(a_edges)                     # descending in a-index
    chi_lo = chi_edges[1:]                             # nearer edge (larger a)
    chi_hi = chi_edges[:-1]                            # farther edge (smaller a)
    chi_mid = 0.5 * (chi_lo + chi_hi)
    dchi = jnp.abs(chi_hi - chi_lo)
    a_mid = 0.5 * (a_edges[1:] + a_edges[:-1])

    a_s = 1.0 / (1.0 + z_source)
    chi_s = cosmo.chi(jnp.asarray(a_s, dtype=chi_mid.dtype))

    prefac = 1.5 * cosmo.Omega_m * _H0_OVER_C ** 2
    lens_kernel = jnp.where(
        chi_mid < chi_s,
        chi_mid * (chi_s - chi_mid) / jnp.where(chi_s > 0, chi_s, 1.0),
        0.0)
    w_shell = prefac * lens_kernel * (1.0 / a_mid) * dchi   # (n_bins,)
    return jnp.sum(w_shell[:, None] * jnp.asarray(delta_shells), axis=0)
