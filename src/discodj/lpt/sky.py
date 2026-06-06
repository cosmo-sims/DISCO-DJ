"""Sky projection for lightcone catalogues.

Turns the Cartesian, box-frame crossing positions DiscoDJ writes into
observer-frame sky coordinates — ``(RA, Dec, redshift)`` — plus optional
redshift-space distortion from the line-of-sight velocity, and angular /
redshift masks. Everything is plain JAX so it composes with ``jit`` / ``grad``.

Conventions (matching the lightcone integration guide):
  - Positions are comoving Mpc/h in the box frame with the replica offset
    already applied; the comoving distance to the observer is ``|x - observer|``
    and equals ``chi(a_cross)`` to fp32 ulp.
  - ``RadialVelocity`` as stored is the Gadget ``sqrt(a)``-scaled line-of-sight
    velocity in km/s (positive = receding). The peculiar velocity used for RSD
    is ``v_radial / sqrt(a)``.
  - ``RA`` in ``[0, 360)`` deg, ``Dec`` in ``[-90, 90]`` deg.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core.healpix import (vec2ang, ang2radec, radec_to_ang, ang2pix_ring,
                            npix2nside)

__all__ = ["cartesian_to_sky", "sky_mask", "add_sky_columns"]

# c in km/s (= (c/H0) * 100 h with the c/H0 = 2997.92458 Mpc/h used elsewhere).
_C_KM_S = 299792.458


def cartesian_to_sky(x, observer, cosmo, v_radial=None, with_rsd=False):
    """Project box-frame Cartesian positions to ``(RA, Dec, redshift)``.

    :param x: ``(M, 3)`` crossing positions (Mpc/h, box frame, replica applied).
    :param observer: ``(3,)`` observer position (same frame).
    :param cosmo: a ``Cosmology`` with timetables built (uses ``chi_to_a``).
    :param v_radial: ``(M,)`` line-of-sight velocity in the stored Gadget
        ``sqrt(a)``-scaled km/s convention. Required iff ``with_rsd``.
    :param with_rsd: if True, also return ``z_obs`` including the peculiar
        line-of-sight Doppler shift.
    :return: dict with ``ra``, ``dec`` (deg), ``z_cosmo``, ``chi`` (Mpc/h), and
        ``z_obs`` if ``with_rsd``.
    """
    x = jnp.asarray(x)
    observer = jnp.asarray(observer, dtype=x.dtype)
    rel = x - observer[None, :]
    d = jnp.linalg.norm(rel, axis=-1)
    theta, phi = vec2ang(rel)
    ra, dec = ang2radec(theta, phi)
    a = cosmo.chi_to_a(d)
    z_cosmo = 1.0 / a - 1.0
    out = {"ra": ra, "dec": dec, "z_cosmo": z_cosmo, "chi": d}
    if with_rsd:
        if v_radial is None:
            raise ValueError("with_rsd=True requires v_radial.")
        a_safe = jnp.where(a > 0, a, 1.0)
        v_pec_los = jnp.asarray(v_radial) / jnp.sqrt(a_safe)  # km/s peculiar
        out["z_obs"] = (1.0 + z_cosmo) * (1.0 + v_pec_los / _C_KM_S) - 1.0
    return out


def sky_mask(ra, dec, z=None, *, healpix_mask=None, nside=None, z_range=None):
    """Boolean keep-mask combining an angular HEALPix mask and a redshift window.

    :param ra, dec: ``(M,)`` sky coordinates in degrees.
    :param z: ``(M,)`` redshift (required iff ``z_range`` is given).
    :param healpix_mask: optional ``(npix,)`` binary RING map; rows landing in a
        zero pixel are dropped. ``nside`` is inferred from its length if unset.
    :param z_range: optional ``(z_min, z_max)`` inclusive window.
    :return: ``(M,)`` bool array (True = keep).
    """
    ra = jnp.asarray(ra)
    keep = jnp.ones(ra.shape, dtype=bool)
    if z_range is not None:
        if z is None:
            raise ValueError("z_range given but z is None.")
        z_min, z_max = z_range
        keep = keep & (jnp.asarray(z) >= z_min) & (jnp.asarray(z) <= z_max)
    if healpix_mask is not None:
        healpix_mask = jnp.asarray(healpix_mask)
        if nside is None:
            nside = npix2nside(healpix_mask.shape[0])
        theta, phi = radec_to_ang(ra, dec)
        pix = ang2pix_ring(nside, theta, phi)
        keep = keep & (healpix_mask[pix] != 0)
    return keep


def add_sky_columns(in_h5: str, out_h5: str | None = None, *,
                    with_rsd: bool = True, batch: int = 1 << 21,
                    cosmo=None, verbose: bool = False) -> str:
    """Append ``RA``/``Dec``/``Redshift`` (+ ``RedshiftRSD``) datasets to a
    lightcone HDF5.

    :param in_h5: existing lightcone catalogue.
    :param out_h5: if None, the columns are appended in place; otherwise the
        input is copied to ``out_h5`` first and the columns added there.
    :param with_rsd: also compute ``RedshiftRSD`` from ``RadialVelocity``
        (requires a radial-velocity catalogue).
    :param cosmo: ``Cosmology`` to use for ``chi_to_a``; if None it is rebuilt
        from the file's ``Omega0``/``OmegaLambda``/``HubbleParam`` header.
    :return: path to the augmented file.
    """
    import shutil
    import h5py
    from ..core import io as _io  # registers the hdf5plugin filter path
    from ..core.io import read_lightcone_header

    path = in_h5
    if out_h5 is not None:
        shutil.copyfile(in_h5, out_h5)
        path = out_h5

    meta = read_lightcone_header(path)
    observer = jnp.asarray(meta["Observer"])
    if cosmo is None:
        from ..cosmology.cosmology import Cosmology
        # Rebuild a background-only cosmology from the header omegas (only
        # chi_to_a is used). The C/B split is irrelevant for chi; Omega_k is
        # fixed so Omega_de = OmegaLambda. NOTE: the header does not record
        # w0/wa, so for non-LCDM dark energy pass an explicit `cosmo`.
        omega_m = float(meta["Omega0"])
        omega_de = float(meta["OmegaLambda"])
        # sigma8 / n_s are irrelevant for the background chi(a) used here.
        cosmo = Cosmology(Omega_c=omega_m, Omega_b=0.0,
                          h=float(meta["HubbleParam"]), sigma8=0.8, n_s=0.96,
                          Omega_k=1.0 - omega_m - omega_de
                          ).compute_timetables(None)

    v_key = "RadialVelocity" if meta.get("v_mode") == "radial" else None
    if with_rsd and v_key is None:
        raise ValueError("with_rsd=True needs a radial-velocity catalogue "
                         "(v_mode='radial').")

    with h5py.File(path, "a") as f:
        g = f["PartType1"]
        M = g["Coordinates"].shape[0]
        # (re)create the sky columns
        for name in ("RA", "Dec", "Redshift", "RedshiftRSD"):
            if name in g:
                del g[name]
        d_ra = g.create_dataset("RA", shape=(M,), dtype="f4")
        d_dec = g.create_dataset("Dec", shape=(M,), dtype="f4")
        d_z = g.create_dataset("Redshift", shape=(M,), dtype="f4")
        d_zr = (g.create_dataset("RedshiftRSD", shape=(M,), dtype="f4")
                if with_rsd else None)

        for start in range(0, M, batch):
            end = min(start + batch, M)
            x = jnp.asarray(g["Coordinates"][start:end])
            vr = (jnp.asarray(g[v_key][start:end])
                  if (with_rsd and v_key) else None)
            sky = cartesian_to_sky(x, observer, cosmo,
                                   v_radial=vr, with_rsd=with_rsd)
            d_ra[start:end] = jnp.asarray(sky["ra"])
            d_dec[start:end] = jnp.asarray(sky["dec"])
            d_z[start:end] = jnp.asarray(sky["z_cosmo"])
            if d_zr is not None:
                d_zr[start:end] = jnp.asarray(sky["z_obs"])
            if verbose:
                print(f"  sky columns {start}-{end} / {M}", flush=True)
    return path
