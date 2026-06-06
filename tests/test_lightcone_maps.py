"""Tests for HEALPix shell maps and Born convergence."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from discodj import DiscoDJ
from discodj.lpt.lightcone_maps import (
    MapSpec, accumulate_shell_maps, shells_to_overdensity,
    density_shells_to_kappa)


def _make_dj(res=8, boxsize=512.0, n_order=2):
    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=n_order, convert_to_numpy=True)
    return dj


def test_shell_maps_conserve_mass():
    """Summing the weighted shell maps recovers the total catalogue mass, and
    every crossing falls in exactly one (shell, pixel)."""
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    out = dj.evaluate_lpt_lightcone(a_far=0.3, a_near=1.0, n_shells=16,
                                    observer=observer, streaming=True,
                                    radial_sort=True, n_part_chunks=2,
                                    n_newton_iters=1, v_mode="radial")
    M = out["x"].shape[0]
    assert M > 0
    a_edges = np.geomspace(0.3, 1.0, 17)
    spec = MapSpec(nside=4, a_edges=a_edges, weighted=True)
    mw = np.ones(M, dtype=np.float32)
    maps = accumulate_shell_maps(out["x"], out["a_cross"], observer, spec,
                                 mass_weight=mw)
    assert maps.shape == spec.shape
    # all crossings in [a_far, a_near] are binned -> total count preserved
    assert np.isclose(float(maps.sum()), float(M), rtol=1e-4)


def test_overdensity_zero_mean():
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    out = dj.evaluate_lpt_lightcone(a_far=0.3, a_near=1.0, n_shells=16,
                                    observer=observer, streaming=True,
                                    radial_sort=True, n_part_chunks=2,
                                    n_newton_iters=1, v_mode="radial")
    spec = MapSpec(nside=4, a_edges=np.geomspace(0.3, 1.0, 9), weighted=False)
    maps = accumulate_shell_maps(out["x"], out["a_cross"], observer, spec)
    delta = shells_to_overdensity(maps)
    # non-empty shells must have <delta> = 0 over the sky
    for b in range(spec.n_bins):
        if float(maps[b].sum()) > 0:
            assert abs(float(jnp.mean(delta[b]))) < 1e-4


def test_kappa_is_differentiable():
    """kappa must flow gradients back to the overdensity shells (Fisher-ready)."""
    from discodj.cosmology.cosmology import Cosmology
    from discodj.cosmology.predefined_cosmologies import get_cosmology_dict_from_name
    cosmo = Cosmology(**get_cosmology_dict_from_name("Planck18EEBAOSN")
                      ).compute_timetables(None)
    a_edges = np.geomspace(0.3, 1.0, 9)
    npix = 12 * 4 * 4
    rng = np.random.default_rng(0)
    delta = jnp.asarray(rng.normal(size=(8, npix)).astype(np.float32))

    def total_kappa(d):
        return jnp.sum(density_shells_to_kappa(d, a_edges, cosmo, z_source=2.0))

    g = jax.grad(total_kappa)(delta)
    assert g.shape == delta.shape
    assert np.all(np.isfinite(np.asarray(g)))
    # near shells (larger a, smaller chi) within the source plane have nonzero weight
    assert float(jnp.abs(g).sum()) > 0


def test_sheet_resampling_conserves_mass(tmp_path):
    """Phase-space-sheet over-sampling (n_resample>1) must conserve the total
    deposited mass and multiply the row count by n_resample^dim."""
    from discodj.core.io import read_lightcone_header
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    spec = MapSpec(nside=8, a_edges=np.geomspace(0.4, 1.0, 9), weighted=True)
    kw = dict(a_far=0.4, a_near=1.0, n_shells=16, observer=observer,
              n_part_chunks=2, n_newton_iters=1, v_mode="radial", map_spec=spec)
    s1 = dj.evaluate_lpt_lightcone_to_hdf5(str(tmp_path / "nr1.h5"),
                                           n_resample=1, **kw)
    s3 = dj.evaluate_lpt_lightcone_to_hdf5(str(tmp_path / "nr3.h5"),
                                           n_resample=3, **kw)
    # rows scale as n_resample^dim
    assert abs(s3["n_particles"] / s1["n_particles"] - 27) < 0.5
    # total deposited mass is invariant (mass split over the sub-particles)
    m1 = float(np.asarray(s1["maps"]).sum())
    m3 = float(np.asarray(s3["maps"]).sum())
    np.testing.assert_allclose(m3, m1, rtol=0.02)
    assert read_lightcone_header(str(tmp_path / "nr3.h5"))["NumResample"] == 3


def test_maps_only_skips_catalogue(tmp_path):
    """write_catalogue=False accumulates the shell maps but writes no PartType1
    rows — letting high n_resample populate fine maps without a huge file."""
    import h5py
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    spec = MapSpec(nside=8, a_edges=np.geomspace(0.4, 1.0, 9), weighted=True)
    kw = dict(a_far=0.4, a_near=1.0, n_shells=16, observer=observer,
              n_part_chunks=2, n_newton_iters=1, v_mode="radial",
              n_resample=2, map_spec=spec)
    full = dj.evaluate_lpt_lightcone_to_hdf5(str(tmp_path / "full.h5"),
                                             write_catalogue=True, **kw)
    mo = dj.evaluate_lpt_lightcone_to_hdf5(str(tmp_path / "mo.h5"),
                                           write_catalogue=False, **kw)
    assert mo["n_particles"] == 0
    with h5py.File(str(tmp_path / "mo.h5"), "r") as f:
        assert f["PartType1/Coordinates"].shape[0] == 0
        assert "Maps/ShellMaps" in f
    # the maps are identical whether or not the catalogue is written
    np.testing.assert_allclose(np.asarray(mo["maps"]),
                               np.asarray(full["maps"]), rtol=1e-5)


def test_on_the_fly_maps_match_posthoc(tmp_path):
    """Maps accumulated during HDF5 generation must match a post-hoc pass over
    the written catalogue."""
    import h5py
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    a_edges = np.geomspace(0.3, 1.0, 9)
    spec = MapSpec(nside=4, a_edges=a_edges, weighted=True)
    path = str(tmp_path / "lc_maps.h5")
    summary = dj.evaluate_lpt_lightcone_to_hdf5(
        path, a_far=0.3, a_near=1.0, n_shells=16, observer=observer,
        n_part_chunks=3, n_newton_iters=1, keep_particle_idx=False,
        v_mode="radial", map_spec=spec)
    assert "maps" in summary
    with h5py.File(path, "r") as f:
        assert "Maps/ShellMaps" in f
        maps_disk = f["Maps/ShellMaps"][:]
        x = f["PartType1/Coordinates"][:]
        a = f["PartType1/ScaleFactor"][:]
        mass = f["PartType1/Masses"][0]
    # post-hoc accumulation over the catalogue
    mw = np.full(x.shape[0], mass, dtype=np.float32)
    maps_post = np.asarray(accumulate_shell_maps(x, a, observer, spec,
                                                 mass_weight=mw))
    np.testing.assert_allclose(maps_disk, maps_post, rtol=1e-4, atol=1e-2)
    # and the returned maps match what's on disk
    np.testing.assert_allclose(np.asarray(summary["maps"]), maps_disk,
                               rtol=1e-4, atol=1e-2)
