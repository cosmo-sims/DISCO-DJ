"""Tests for lightcone format exporters."""

from __future__ import annotations

import numpy as np
import pytest

from discodj import DiscoDJ
from discodj.lpt.lightcone_export import (
    to_radec_table, to_skycatalog, to_gadget_lightcone_hdf5, write_healpix_fits)
from discodj.lpt.sky import cartesian_to_sky
from discodj.core.io import read_lightcone_header


def _make_lightcone(tmp_path, res=8):
    dj = DiscoDJ(dim=3, res=res, boxsize=512.0)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=2, convert_to_numpy=True)
    path = str(tmp_path / "lc.h5")
    dj.evaluate_lpt_lightcone_to_hdf5(
        path, a_far=0.3, a_near=1.0, n_shells=16,
        observer=np.array([256.0, 256.0, 256.0]),
        n_part_chunks=2, n_newton_iters=1, v_mode="radial")
    return path, dj


def test_radec_table_hdf5_roundtrip(tmp_path):
    import h5py
    lc, _ = _make_lightcone(tmp_path)
    M = read_lightcone_header(lc)["n_particles"]
    out = to_radec_table(lc, str(tmp_path / "t.h5"), fmt="hdf5", with_rsd=True)
    with h5py.File(out, "r") as f:
        g = f["Catalog"]
        for c in ("RA", "Dec", "Redshift", "RedshiftRSD", "Weight"):
            assert g[c].shape == (M,)
        assert np.all((g["RA"][:] >= 0) & (g["RA"][:] < 360.0))
        assert np.all((g["Dec"][:] >= -90.0) & (g["Dec"][:] <= 90.0))


def test_radec_table_parquet(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    lc, _ = _make_lightcone(tmp_path)
    M = read_lightcone_header(lc)["n_particles"]
    out = to_radec_table(lc, str(tmp_path / "t.parquet"), fmt="parquet",
                         with_rsd=True, batch=1000)
    tab = pq.read_table(out)
    assert tab.num_rows == M
    assert set(["RA", "Dec", "Redshift", "RedshiftRSD", "Weight"]).issubset(
        set(tab.column_names))


def test_radec_table_fits(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    lc, _ = _make_lightcone(tmp_path)
    M = read_lightcone_header(lc)["n_particles"]
    out = to_radec_table(lc, str(tmp_path / "t.fits"), fmt="fits", with_rsd=False)
    with fits.open(out) as hdul:
        assert hdul[1].data["RA"].shape[0] == M


def test_skycatalog_parquet(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    lc, _ = _make_lightcone(tmp_path)
    M = read_lightcone_header(lc)["n_particles"]
    out = to_skycatalog(lc, str(tmp_path / "sky.parquet"), with_rsd=True,
                        batch=1000)
    tab = pq.read_table(out)
    assert tab.num_rows == M
    assert set(["galaxy_id", "ra", "dec", "redshift", "redshift_true"]).issubset(
        set(tab.column_names))
    gid = tab.column("galaxy_id").to_numpy()
    assert gid[0] == 0 and gid[-1] == M - 1   # sequential across row groups


def test_gadget_swift_reheader(tmp_path):
    import h5py
    lc, _ = _make_lightcone(tmp_path)
    out = to_gadget_lightcone_hdf5(lc, str(tmp_path / "swift.h5"), flavor="swift")
    meta_in = read_lightcone_header(lc)
    h = float(meta_in["HubbleParam"])
    with h5py.File(out, "r") as f:
        assert "Cosmology" in f and "Units" in f
        # box and coordinates rescaled by 1/h
        np.testing.assert_allclose(f["Header"].attrs["BoxSize"],
                                   meta_in["BoxSize"] / h, rtol=1e-6)
        assert f["PartType1/Coordinates"].shape[0] == meta_in["n_particles"]

    out4 = to_gadget_lightcone_hdf5(lc, str(tmp_path / "g4.h5"), flavor="gadget4")
    with h5py.File(out4, "r") as f:
        assert "Parameters" in f
        np.testing.assert_allclose(f["Header"].attrs["BoxSize"],
                                   meta_in["BoxSize"], rtol=1e-6)


def test_write_healpix_fits_readable(tmp_path):
    healpy = pytest.importorskip("healpy")
    from discodj.core.healpix import nside2npix
    nside = 4
    rng = np.random.default_rng(0)
    m = rng.normal(size=nside2npix(nside)).astype(np.float32)
    out = write_healpix_fits(m, str(tmp_path / "map.fits"))
    m_read = healpy.read_map(out, verbose=False) if "verbose" in \
        healpy.read_map.__code__.co_varnames else healpy.read_map(out)
    np.testing.assert_allclose(m_read, m, rtol=1e-5)
