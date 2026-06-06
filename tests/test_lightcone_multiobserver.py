"""Tests for multi-observer lightcone batching."""

from __future__ import annotations

import numpy as np

from discodj import DiscoDJ
from discodj.core.io import read_lightcone_header


def _make_dj(res=8, boxsize=512.0, n_order=2):
    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=n_order, convert_to_numpy=True)
    return dj


def test_multi_observer_subset_matches_single(tmp_path):
    """The ObserverIndex==0 subset of a multi-observer run must reproduce a
    single-observer run at the same observer (same iteration order -> identical
    coordinates), and the header/columns must advertise the extra sky."""
    import h5py
    dj = _make_dj()
    obs_a = np.array([256.0, 256.0, 256.0])
    obs_b = np.array([128.0, 384.0, 200.0])

    kw = dict(a_far=0.3, a_near=1.0, n_shells=16, n_part_chunks=2,
              n_newton_iters=1, v_mode="radial")

    single = str(tmp_path / "single.h5")
    s_summary = dj.evaluate_lpt_lightcone_to_hdf5(single, observer=obs_a, **kw)

    multi = str(tmp_path / "multi.h5")
    m_summary = dj.evaluate_lpt_lightcone_to_hdf5(
        multi, observer=np.stack([obs_a, obs_b]), **kw)

    # single-observer file: no ObserverIndex, no Observers header
    s_meta = read_lightcone_header(single)
    assert s_meta["has_observer_idx"] is False
    assert "NumObservers" not in s_meta

    # multi file advertises the batch
    m_meta = read_lightcone_header(multi)
    assert m_meta["has_observer_idx"] is True
    assert int(m_meta["NumObservers"]) == 2
    assert np.asarray(m_meta["Observers"]).shape == (2, 3)
    assert m_summary["n_observers"] == 2

    with h5py.File(single, "r") as f:
        x_s = f["PartType1/Coordinates"][:]
        a_s = f["PartType1/ScaleFactor"][:]
    with h5py.File(multi, "r") as f:
        x_m = f["PartType1/Coordinates"][:]
        a_m = f["PartType1/ScaleFactor"][:]
        obs_idx = f["PartType1/ObserverIndex"][:]

    # both observers produce crossings
    assert (obs_idx == 0).sum() > 0 and (obs_idx == 1).sum() > 0
    # observer-0 subset is bit-identical to the single-observer run
    sel0 = obs_idx == 0
    assert int(sel0.sum()) == x_s.shape[0]
    np.testing.assert_array_equal(x_m[sel0], x_s)
    np.testing.assert_array_equal(a_m[sel0], a_s)
    # total rows = sum over observers
    assert m_summary["n_particles"] == x_m.shape[0]
    assert m_summary["n_particles"] > s_summary["n_particles"]


def test_multi_observer_maps_stack(tmp_path):
    """With a MapSpec, multiple observers yield a (n_obs, n_bins, npix) stack."""
    from discodj.lpt.lightcone_maps import MapSpec
    dj = _make_dj()
    obs = np.stack([np.array([256.0, 256.0, 256.0]),
                    np.array([200.0, 300.0, 256.0])])
    spec = MapSpec(nside=4, a_edges=np.geomspace(0.3, 1.0, 9), weighted=True)
    summary = dj.evaluate_lpt_lightcone_to_hdf5(
        str(tmp_path / "m_maps.h5"), observer=obs, a_far=0.3, a_near=1.0,
        n_shells=16, n_part_chunks=2, n_newton_iters=1, v_mode="radial",
        map_spec=spec)
    maps = np.asarray(summary["maps"])
    assert maps.shape == (2,) + spec.shape
    # each observer sees a non-empty sky
    assert maps[0].sum() > 0 and maps[1].sum() > 0
