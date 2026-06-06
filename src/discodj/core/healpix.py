"""Pure-JAX HEALPix utilities (RING scheme).

A small, dependency-free, fully differentiable subset of HEALPix sufficient
for binning lightcone crossings into sky maps:

  - ``ang2pix_ring`` — angular coordinates -> RING pixel index, branch-free
    (both the equatorial and polar-cap formulas are evaluated and selected
    with ``jnp.where``), so it is ``jit``/``vmap`` friendly. Validated to exact
    integer agreement against ``healpy.ang2pix`` in the test-suite.
  - ``vec2ang`` / ``ang2radec`` / ``radec_to_ang`` — coordinate conversions.
  - ``accumulate_map`` — scatter weights into a pixel map via
    ``jax.ops.segment_sum``; differentiable in ``weights`` (so map-level
    observables like C_ell flow gradients back to the input catalogue).

The pixelisation matches the HEALPix convention exactly: ``theta`` is the
colatitude in ``[0, pi]`` (0 at the north pole), ``phi`` the longitude in
``[0, 2 pi)`` measured eastward. Only the RING ordering is implemented (the
natural ordering for power-spectrum / map-level work).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["nside2npix", "npix2nside", "vec2ang", "ang2radec",
           "radec_to_ang", "ang2pix_ring", "accumulate_map"]


def nside2npix(nside: int) -> int:
    """Number of RING pixels for a given ``nside`` (``12 * nside**2``)."""
    return 12 * int(nside) * int(nside)


def npix2nside(npix: int) -> int:
    """Inverse of :func:`nside2npix`. Raises if ``npix`` is not ``12 nside^2``."""
    nside = int(round((int(npix) / 12.0) ** 0.5))
    if nside2npix(nside) != int(npix):
        raise ValueError(f"npix={npix} is not 12*nside^2 for any integer nside")
    return nside


def vec2ang(xyz):
    """Cartesian direction(s) ``(..., 3)`` -> ``(theta, phi)``.

    ``theta`` = colatitude (arccos of the normalised z), ``phi`` = longitude in
    ``[0, 2 pi)``. The vectors need not be unit-normalised.
    """
    xyz = jnp.asarray(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = jnp.sqrt(x * x + y * y + z * z)
    r_safe = jnp.where(r > 0, r, 1.0)
    theta = jnp.arccos(jnp.clip(z / r_safe, -1.0, 1.0))
    phi = jnp.arctan2(y, x)
    phi = jnp.mod(phi, 2.0 * jnp.pi)
    return theta, phi


def ang2radec(theta, phi):
    """``(theta, phi)`` in radians -> ``(ra, dec)`` in degrees.

    ``ra`` in ``[0, 360)``, ``dec`` in ``[-90, 90]`` (dec = 90 - colatitude).
    """
    ra = jnp.mod(jnp.rad2deg(phi), 360.0)
    dec = 90.0 - jnp.rad2deg(theta)
    return ra, dec


def radec_to_ang(ra, dec):
    """``(ra, dec)`` in degrees -> ``(theta, phi)`` in radians (inverse of
    :func:`ang2radec`)."""
    theta = jnp.deg2rad(90.0 - jnp.asarray(dec))
    phi = jnp.mod(jnp.deg2rad(jnp.asarray(ra)), 2.0 * jnp.pi)
    return theta, phi


def ang2pix_ring(nside: int, theta, phi):
    """Angular coordinates -> HEALPix RING pixel index (``int32``).

    Branch-free transcription of the reference ``ang2pix_ring``: the
    equatorial-belt and polar-cap pixel indices are both computed and selected
    with ``jnp.where`` on ``|cos theta| <= 2/3``. ``nside`` is a static Python
    int. Inputs broadcast; output has the broadcast shape.
    """
    nside = int(nside)
    nl4 = 4 * nside
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside

    theta = jnp.asarray(theta, dtype=jnp.float64 if jax.config.jax_enable_x64
                        else jnp.float32)
    phi = jnp.asarray(phi, dtype=theta.dtype)

    z = jnp.cos(theta)
    za = jnp.abs(z)
    # phi in [0, 2pi) -> tt in [0, 4)
    tt = jnp.mod(phi, 2.0 * jnp.pi) * (2.0 / jnp.pi)

    # int32 is sufficient for every practical nside (npix = 12 nside^2 stays
    # well under 2^31 up to nside = 8192), and avoids the x64 truncation path.
    i32 = jnp.int32

    # ---- Equatorial belt (|z| <= 2/3) ----
    temp1 = nside * (0.5 + tt)
    temp2 = nside * (z * 0.75)
    jp = jnp.floor(temp1 - temp2)         # ascending edge-line index
    jm = jnp.floor(temp1 + temp2)         # descending edge-line index
    ir_eq = (nside + 1 + jp - jm).astype(i32)   # ring index in {1 .. 2 nside + 1}
    kshift = 1 - jnp.mod(ir_eq, 2)              # 1 if even ring, else 0
    ip_eq = jnp.mod(
        jnp.floor((jp + jm - nside + kshift + 1) / 2.0).astype(i32), nl4)
    pix_eq = ncap + (ir_eq - 1) * nl4 + ip_eq

    # ---- Polar caps (|z| > 2/3) ----
    tp = tt - jnp.floor(tt)
    # guard the sqrt against tiny negatives from rounding when za -> 1
    tmp = nside * jnp.sqrt(jnp.clip(3.0 * (1.0 - za), 0.0, None))
    jp_c = jnp.floor(tp * tmp)
    jm_c = jnp.floor((1.0 - tp) * tmp)
    ir_c = (jp_c + jm_c).astype(i32) + 1   # ring from the nearest pole
    ip_c = jnp.floor(tt * ir_c.astype(theta.dtype)).astype(i32)
    ip_c = jnp.mod(ip_c, 4 * ir_c)
    pix_north = 2 * ir_c * (ir_c - 1) + ip_c
    pix_south = (npix - 2 * ir_c * (ir_c + 1)) + ip_c
    pix_cap = jnp.where(z > 0, pix_north, pix_south)

    pix = jnp.where(za <= 2.0 / 3.0, pix_eq, pix_cap)
    return jnp.clip(pix, 0, npix - 1).astype(jnp.int32)


def accumulate_map(pix, weights, npix: int):
    """Scatter ``weights`` into a length-``npix`` map at pixel indices ``pix``.

    Thin wrapper over ``jax.ops.segment_sum`` — differentiable in ``weights``.
    ``pix`` is an integer array; ``weights`` broadcasts to its shape (pass
    ``1.0`` for a pure count map).
    """
    pix = jnp.asarray(pix, dtype=jnp.int32)
    weights = jnp.broadcast_to(jnp.asarray(weights, dtype=jnp.float32), pix.shape)
    return jax.ops.segment_sum(weights, pix, num_segments=int(npix))
