"""Tests for the pure-JAX HEALPix utilities and sky projection."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from discodj import DiscoDJ
from discodj.core import healpix as hp
from discodj.lpt.sky import cartesian_to_sky, sky_mask, add_sky_columns


def _random_angles(n, seed=0):
    rng = np.random.default_rng(seed)
    # cos(theta) uniform in [-1, 1] -> uniform on the sphere
    theta = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return theta, phi


@pytest.mark.parametrize("nside", [1, 2, 4, 16, 64, 256])
def test_ang2pix_matches_healpy(nside):
    healpy = pytest.importorskip("healpy")
    theta, phi = _random_angles(20000, seed=nside)
    ref = healpy.ang2pix(nside, theta, phi, nest=False)
    got = np.asarray(hp.ang2pix_ring(nside, jnp.asarray(theta), jnp.asarray(phi)))
    # float32 vs healpy's float64 can disagree by 1 pixel only for points
    # landing within ~1e-7 of a pixel boundary; require essentially-exact.
    frac_mismatch = np.mean(ref != got)
    assert frac_mismatch < 1e-3, f"nside={nside}: {frac_mismatch:.2e} mismatch"


def test_ang2pix_range_and_poles():
    nside = 32
    npix = hp.nside2npix(nside)
    theta, phi = _random_angles(5000, seed=1)
    pix = np.asarray(hp.ang2pix_ring(nside, jnp.asarray(theta), jnp.asarray(phi)))
    assert pix.min() >= 0 and pix.max() < npix
    # near-pole points land in the first/last polar ring; pixel 0 at north
    assert int(hp.ang2pix_ring(nside, jnp.asarray(1e-6), jnp.asarray(0.0))) == 0
    south = int(hp.ang2pix_ring(nside, jnp.asarray(np.pi - 1e-6),
                                jnp.asarray(0.0)))
    assert npix - 4 <= south < npix  # within the last (4-pixel) ring


def test_nside_npix_roundtrip():
    for nside in (1, 2, 8, 128, 512):
        assert hp.npix2nside(hp.nside2npix(nside)) == nside
    with pytest.raises(ValueError):
        hp.npix2nside(13)


def test_radec_roundtrip():
    theta, phi = _random_angles(1000, seed=2)
    ra, dec = hp.ang2radec(jnp.asarray(theta), jnp.asarray(phi))
    th2, ph2 = hp.radec_to_ang(ra, dec)
    np.testing.assert_allclose(np.asarray(th2), theta, atol=1e-5)
    np.testing.assert_allclose(np.mod(np.asarray(ph2), 2 * np.pi),
                               np.mod(phi, 2 * np.pi), atol=1e-5)


def test_accumulate_map_conserves_weight():
    nside = 8
    npix = hp.nside2npix(nside)
    theta, phi = _random_angles(10000, seed=3)
    pix = hp.ang2pix_ring(nside, jnp.asarray(theta), jnp.asarray(phi))
    w = jnp.asarray(np.random.default_rng(0).uniform(size=theta.shape))
    m = hp.accumulate_map(pix, w, npix)
    assert m.shape == (npix,)
    np.testing.assert_allclose(float(m.sum()), float(w.sum()), rtol=1e-4)


def _make_cosmo():
    from discodj.cosmology.cosmology import Cosmology
    from discodj.cosmology.predefined_cosmologies import get_cosmology_dict_from_name
    return Cosmology(**get_cosmology_dict_from_name("Planck18EEBAOSN")
                     ).compute_timetables(None)


def test_sky_projection_roundtrip():
    """chi from |x - observer| must invert to the redshift the position was
    placed at; RSD shifts z in the receding-positive direction."""
    cosmo = _make_cosmo()
    observer = jnp.array([0.0, 0.0, 0.0])
    # place a point at a known scale factor's comoving distance along +x
    a_true = 0.5
    chi_true = float(cosmo.chi(jnp.asarray(a_true)))
    x = jnp.array([[chi_true, 0.0, 0.0]])
    sky = cartesian_to_sky(x, observer, cosmo,
                           v_radial=jnp.array([300.0]), with_rsd=True)
    np.testing.assert_allclose(float(sky["z_cosmo"][0]), 1.0 / a_true - 1.0,
                               rtol=1e-3)
    # +x direction: dec=0, ra=0
    np.testing.assert_allclose(float(sky["dec"][0]), 0.0, atol=1e-4)
    np.testing.assert_allclose(float(sky["ra"][0]), 0.0, atol=1e-4)
    # receding (positive v_radial) -> z_obs > z_cosmo
    assert float(sky["z_obs"][0]) > float(sky["z_cosmo"][0])


def test_sky_mask_combines_angular_and_redshift():
    nside = 4
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.int8)
    mask[: npix // 2] = 1  # keep one hemisphere of pixels
    ra = jnp.array([10.0, 20.0, 30.0])
    dec = jnp.array([10.0, -10.0, 80.0])
    z = jnp.array([0.1, 0.5, 2.0])
    keep = sky_mask(ra, dec, z, healpix_mask=jnp.asarray(mask), nside=nside,
                    z_range=(0.0, 1.0))
    assert keep.shape == (3,)
    # the z=2.0 row is excluded by the redshift window regardless of pixel
    assert not bool(keep[2])


def test_add_sky_columns_inplace(tmp_path):
    """add_sky_columns appends RA/Dec/Redshift(/RSD) consistent with a direct
    cartesian_to_sky call on the catalogue."""
    import h5py
    dj = DiscoDJ(dim=3, res=8, boxsize=512.0)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=2, convert_to_numpy=True)
    observer = np.array([256.0, 256.0, 256.0])
    path = str(tmp_path / "lc.h5")
    dj.evaluate_lpt_lightcone_to_hdf5(
        path, a_far=0.3, a_near=1.0, n_shells=16, observer=observer,
        n_part_chunks=2, n_newton_iters=1, v_mode="radial")

    add_sky_columns(path, with_rsd=True, cosmo=dj.cosmo)
    with h5py.File(path, "r") as f:
        g = f["PartType1"]
        for c in ("RA", "Dec", "Redshift", "RedshiftRSD"):
            assert c in g
        x = g["Coordinates"][:]
        vr = g["RadialVelocity"][:]
        ra = g["RA"][:]
    ref = cartesian_to_sky(jnp.asarray(x), jnp.asarray(observer), dj.cosmo,
                           v_radial=jnp.asarray(vr), with_rsd=True)
    np.testing.assert_allclose(ra, np.asarray(ref["ra"]), atol=1e-3)
