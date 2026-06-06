"""Format interoperability for lightcone catalogues.

File -> file converters that read DiscoDJ's native lightcone HDF5 (via the
batched reader) and emit formats common in survey pipelines:

  - :func:`to_radec_table` — a minimal ``(RA, Dec, Redshift[, RedshiftRSD],
    Weight)`` columnar catalogue in HDF5 / Parquet / FITS.
  - :func:`to_skycatalog` — a CosmoDC2 / LSST-DESC SkyCatalog-style Parquet
    table (DM-particle proxy, not synthetic galaxies).
  - :func:`to_gadget_lightcone_hdf5` — re-headers the native file into the
    Gadget-4 / SWIFT lightcone conventions so swiftsimio / pynbody / yt ingest
    it directly.
  - :func:`write_healpix_fits` — writes a HEALPix map (e.g. from
    :mod:`discodj.lpt.lightcone_maps`) as a standard HEALPix FITS file.

Heavy dependencies (``pyarrow`` for Parquet, ``astropy`` for FITS) are imported
lazily so the core package stays dependency-light; install them via the
``discodj[sky]`` extra.
"""

from __future__ import annotations

import numpy as onp

from ..core.io import read_lightcone_header
from .sky import cartesian_to_sky

__all__ = ["to_radec_table", "to_skycatalog", "to_gadget_lightcone_hdf5",
           "write_healpix_fits"]


def _iter_sky(in_h5, *, with_rsd, batch, cosmo=None):
    """Yield ``(start, end, dict)`` of sky columns over batches of the input."""
    import h5py
    from ..core import io as _io  # registers hdf5plugin filter path
    meta = read_lightcone_header(in_h5)
    observer = onp.asarray(meta["Observer"])
    if cosmo is None:
        from ..cosmology.cosmology import Cosmology
        omega_m = float(meta["Omega0"]); omega_de = float(meta["OmegaLambda"])
        # sigma8 / n_s are irrelevant for the background chi(a) used here.
        cosmo = Cosmology(Omega_c=omega_m, Omega_b=0.0,
                          h=float(meta["HubbleParam"]), sigma8=0.8, n_s=0.96,
                          Omega_k=1.0 - omega_m - omega_de
                          ).compute_timetables(None)
    v_key = "RadialVelocity" if meta.get("v_mode") == "radial" else None
    if with_rsd and v_key is None:
        raise ValueError("with_rsd=True needs a radial-velocity catalogue.")
    with h5py.File(in_h5, "r") as f:
        g = f["PartType1"]
        M = g["Coordinates"].shape[0]
        has_mass = "Masses" in g
        for start in range(0, M, batch):
            end = min(start + batch, M)
            x = onp.asarray(g["Coordinates"][start:end])
            vr = onp.asarray(g[v_key][start:end]) if (with_rsd and v_key) else None
            sky = cartesian_to_sky(x, observer, cosmo,
                                   v_radial=vr, with_rsd=with_rsd)
            rec = {"RA": onp.asarray(sky["ra"], dtype=onp.float32),
                   "Dec": onp.asarray(sky["dec"], dtype=onp.float32),
                   "Redshift": onp.asarray(sky["z_cosmo"], dtype=onp.float32)}
            if with_rsd:
                rec["RedshiftRSD"] = onp.asarray(sky["z_obs"], dtype=onp.float32)
            rec["Weight"] = (onp.asarray(g["Masses"][start:end], dtype=onp.float32)
                             if has_mass else onp.ones(end - start, onp.float32))
            yield start, end, M, rec


def to_radec_table(in_h5: str, out: str, *, fmt: str = "hdf5",
                   with_rsd: bool = True, batch: int = 1 << 21,
                   cosmo=None, verbose: bool = False) -> str:
    """Export a plain ``(RA, Dec, Redshift[, RedshiftRSD], Weight)`` catalogue.

    :param fmt: ``"hdf5"`` (streamed), ``"parquet"`` (streamed via row groups),
        or ``"fits"`` (accumulated in memory then written).
    :return: the output path.
    """
    assert fmt in ("hdf5", "parquet", "fits"), f"unknown fmt {fmt!r}"
    cols = ["RA", "Dec", "Redshift"] + (["RedshiftRSD"] if with_rsd else []) + ["Weight"]

    if fmt == "hdf5":
        import h5py
        with h5py.File(out, "w") as f:
            grp = f.create_group("Catalog")
            ds = None
            for start, end, M, rec in _iter_sky(in_h5, with_rsd=with_rsd,
                                                 batch=batch, cosmo=cosmo):
                if ds is None:
                    ds = {c: grp.create_dataset(c, shape=(M,), dtype="f4")
                          for c in cols}
                for c in cols:
                    ds[c][start:end] = rec[c]
                if verbose:
                    print(f"  {end}/{M}", flush=True)
        return out

    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        try:
            for start, end, M, rec in _iter_sky(in_h5, with_rsd=with_rsd,
                                                 batch=batch, cosmo=cosmo):
                table = pa.table({c: rec[c] for c in cols})
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
                if verbose:
                    print(f"  {end}/{M}", flush=True)
        finally:
            if writer is not None:
                writer.close()
        return out

    # fits — accumulate then write
    from astropy.table import Table
    acc = {c: [] for c in cols}
    for start, end, M, rec in _iter_sky(in_h5, with_rsd=with_rsd,
                                        batch=batch, cosmo=cosmo):
        for c in cols:
            acc[c].append(rec[c])
    tab = Table({c: onp.concatenate(acc[c]) if acc[c] else onp.zeros(0, onp.float32)
                 for c in cols})
    tab.write(out, format="fits", overwrite=True)
    return out


def to_skycatalog(in_h5: str, out: str, *, with_rsd: bool = True,
                  batch: int = 1 << 21, cosmo=None, verbose: bool = False) -> str:
    """Export a CosmoDC2 / LSST-DESC SkyCatalog-style Parquet table.

    Columns: ``galaxy_id`` (sequential), ``ra``, ``dec``, ``redshift`` (the RSD
    redshift if available, else cosmological), ``redshift_true``. This is a
    **dark-matter-particle proxy**, not a synthetic galaxy catalogue.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    gid0 = 0
    try:
        for start, end, M, rec in _iter_sky(in_h5, with_rsd=with_rsd,
                                            batch=batch, cosmo=cosmo):
            n = end - start
            z_obs = rec.get("RedshiftRSD", rec["Redshift"])
            table = pa.table({
                "galaxy_id": onp.arange(gid0, gid0 + n, dtype=onp.int64),
                "ra": rec["RA"].astype(onp.float64),
                "dec": rec["Dec"].astype(onp.float64),
                "redshift": z_obs.astype(onp.float64),
                "redshift_true": rec["Redshift"].astype(onp.float64),
            })
            gid0 += n
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
            if verbose:
                print(f"  {end}/{M}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return out


def to_gadget_lightcone_hdf5(in_h5: str, out: str, *, flavor: str = "gadget4",
                             batch: int = 1 << 21) -> str:
    """Re-header the native lightcone into a Gadget-4 / SWIFT-style HDF5.

    Copies the ``PartType1`` datasets verbatim and writes the cosmology/box
    metadata under the group names those codes' readers expect (``Parameters``
    for Gadget-4, ``Cosmology`` + ``Units`` and a ``/h``-scaled ``BoxSize`` for
    SWIFT). ``ScaleFactor`` (per-particle) is retained — the standard
    snapshot-level ``Time`` is not meaningful for a lightcone.
    """
    import h5py
    from ..core import io as _io  # registers hdf5plugin filter path
    assert flavor in ("gadget4", "swift"), f"unknown flavor {flavor!r}"
    meta = read_lightcone_header(in_h5)
    h = float(meta["HubbleParam"])
    with h5py.File(in_h5, "r") as fin, h5py.File(out, "w") as fout:
        hin = fin["Header"]
        hdr = fout.create_group("Header")
        for k, v in hin.attrs.items():
            hdr.attrs[k] = v
        box = float(meta["BoxSize"])
        if flavor == "swift":
            hdr.attrs["BoxSize"] = box / h
            cg = fout.create_group("Cosmology")
            cg.attrs["Omega_m"] = float(meta["Omega0"])
            cg.attrs["Omega_lambda"] = float(meta["OmegaLambda"])
            cg.attrs["h"] = h
            ug = fout.create_group("Units")
            ug.attrs["Unit length in cgs (U_L)"] = 3.085678e24 / h  # Mpc/h -> cm
            ug.attrs["Unit mass in cgs (U_M)"] = 1.989e43 / h       # 1e10 Msun/h
            ug.attrs["Unit velocity in cgs (U_V)"] = 1.0e5          # km/s
        else:  # gadget4
            pg = fout.create_group("Parameters")
            pg.attrs["Omega0"] = float(meta["Omega0"])
            pg.attrs["OmegaLambda"] = float(meta["OmegaLambda"])
            pg.attrs["HubbleParam"] = h
        # Stream-copy PartType1 datasets, applying the SWIFT /h length rescale.
        gin = fin["PartType1"]
        gout = fout.create_group("PartType1")
        for name, dset in gin.items():
            M = dset.shape[0]
            out_ds = gout.create_dataset(name, shape=dset.shape, dtype=dset.dtype)
            for start in range(0, M, batch):
                end = min(start + batch, M)
                chunk = dset[start:end]
                if flavor == "swift" and name == "Coordinates":
                    chunk = chunk / h
                out_ds[start:end] = chunk
    return out


def write_healpix_fits(hmap, out: str, *, nest: bool = False,
                       column_name: str = "SIGNAL") -> str:
    """Write a HEALPix map to a standard HEALPix FITS file (readable by
    ``healpy.read_map``).

    :param hmap: ``(npix,)`` map; ``npix`` must be ``12 nside^2``.
    :param nest: RING (default) or NESTED ordering flag in the header.
    """
    from astropy.io import fits
    from ..core.healpix import npix2nside
    hmap = onp.asarray(hmap, dtype=onp.float32).ravel()
    nside = npix2nside(hmap.shape[0])
    col = fits.Column(name=column_name, format="E", array=hmap)
    hdu = fits.BinTableHDU.from_columns([col])
    hdr = hdu.header
    hdr["PIXTYPE"] = "HEALPIX"
    hdr["ORDERING"] = ("NESTED" if nest else "RING")
    hdr["NSIDE"] = nside
    hdr["FIRSTPIX"] = 0
    hdr["LASTPIX"] = hmap.shape[0] - 1
    hdr["INDXSCHM"] = "IMPLICIT"
    hdr["OBJECT"] = "FULLSKY"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(out, overwrite=True)
    return out
