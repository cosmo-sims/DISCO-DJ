"""Tests for the interleaved N-body past-lightcone."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from discodj import DiscoDJ
from discodj.core.io import read_lightcone_header


def _make_dj(res=16, boxsize=512.0, n_order=2):
    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=n_order, convert_to_numpy=True)
    return dj


def test_nbody_lightcone_crossings_on_chi_curve(tmp_path):
    """Every emitted N-body crossing must satisfy |x - observer| ~ chi(a_cross),
    i.e. it lies on the past lightcone to within the bracket interpolation
    error."""
    import h5py
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    path = str(tmp_path / "lc_nbody.h5")
    summary = dj.run_nbody_lightcone(
        path, a_ini=0.3, a_end=1.0, n_steps=20, observer=observer,
        res_pm=32, v_mode="radial", keep_particle_idx=True, stepper="bullfrog")
    assert summary["n_particles"] > 0
    assert summary["n_steps"] == 20

    meta = read_lightcone_header(path)
    assert meta["LightconeMode"] == 1
    with h5py.File(path, "r") as f:
        x = f["PartType1/Coordinates"][:]
        a = f["PartType1/ScaleFactor"][:]
    d = np.linalg.norm(x - observer[None, :], axis=-1)
    chi = np.asarray(dj.cosmo.chi(jnp.asarray(a)))
    # residual relative to a shell width; linear-in-a bracket interp over 20
    # steps keeps |d - chi| well below the box scale.
    resid = np.abs(d - chi)
    assert np.median(resid) < 5.0           # Mpc/h
    assert np.percentile(resid, 95) < 30.0
    # a_cross within the integration window
    assert a.min() >= 0.3 - 1e-3 and a.max() <= 1.0 + 1e-3


def test_nbody_lightcone_approaches_lpt_in_linear_regime(tmp_path):
    """With ICs from LPT and a shallow lightcone, the N-body lightcone should
    produce a comparable number of crossings to the LPT lightcone (the two
    trajectories agree in the quasi-linear regime)."""
    import h5py
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])

    nbody_path = str(tmp_path / "nb.h5")
    nb = dj.run_nbody_lightcone(
        nbody_path, a_ini=0.5, a_end=1.0, n_steps=24, observer=observer,
        res_pm=32, v_mode="radial", stepper="bullfrog")

    lpt = dj.evaluate_lpt_lightcone_to_hdf5(
        str(tmp_path / "lpt.h5"), a_far=0.5, a_near=1.0, n_shells=24,
        observer=observer, n_part_chunks=2, n_newton_iters=1, v_mode="radial")

    n_nb, n_lpt = nb["n_particles"], lpt["n_particles"]
    assert n_nb > 0 and n_lpt > 0
    # counts within a factor ~1.5 (boundary + nonlinear differences)
    assert 0.5 < n_nb / n_lpt < 2.0, f"nbody {n_nb} vs lpt {n_lpt}"
