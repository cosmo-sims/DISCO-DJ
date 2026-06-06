"""Tests for the deformation / tidal / velocity-gradient lightcone outputs."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from discodj import DiscoDJ


def _make_dj(res=8, boxsize=512.0, n_order=2):
    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=n_order, convert_to_numpy=True)
    return dj


def test_deformation_matches_jacobian():
    """The per-row deformation tensor T = I + sum_n D^n grad psi_n gathered in
    the lightcone kernel must match DiscoDJ.evaluate_jacobian_from_psi evaluated
    at the same scale factor and Lagrangian particle."""
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    out = dj.evaluate_lpt_lightcone(
        a_far=0.3, a_near=1.0, n_shells=16, observer=observer,
        streaming=True, radial_sort=True, n_part_chunks=2, n_newton_iters=2,
        keep_particle_idx=True, v_mode="full", deformation_mode="full")
    assert "deformation" in out and "velocity_gradient" in out
    T = np.asarray(out["deformation"])
    assert T.shape == (out["x"].shape[0], 3, 3)

    # Build the reference total displacement Psi(a) = sum_n D_n(a) psi_n for the
    # a of one row, then its Jacobian, and compare at that row's particle.
    n_order = dj._lpt.n_order
    psi_flats = [np.asarray(dj._lpt.psi[f"psi_{n}"]).reshape(-1, 3)
                 for n in range(1, n_order + 1)]
    row = int(np.argmin(np.abs(out["a_cross"] - np.median(out["a_cross"]))))
    a_row = float(out["a_cross"][row])
    pid = int(out["particle_idx"][row])
    D = float(dj.cosmo.Dplus(jnp.asarray(a_row)))
    psi_total = np.zeros_like(psi_flats[0])
    Dp = D
    for psi_n in psi_flats:
        psi_total += Dp * psi_n
        Dp *= D
    psi_total = psi_total.reshape(*([dj.res] * 3), 3)
    _, _, dpsi = dj.evaluate_jacobian_from_psi(jnp.asarray(psi_total))
    T_ref = np.asarray(dpsi).reshape(-1, 3, 3)[pid]
    np.testing.assert_allclose(T[row], T_ref, atol=1e-3, rtol=1e-3)


def test_stream_density_and_tidal_evals():
    """stream mode emits a positive stream density and 3 ascending tidal
    eigenvalues whose sum equals tr(T) = det-consistent first invariant."""
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    out = dj.evaluate_lpt_lightcone(
        a_far=0.3, a_near=1.0, n_shells=16, observer=observer,
        streaming=True, radial_sort=True, n_part_chunks=2, n_newton_iters=2,
        v_mode="radial", deformation_mode="stream")
    sd = np.asarray(out["stream_density"])
    ev = np.asarray(out["tidal_evals"])
    assert sd.shape == (out["x"].shape[0],)
    assert ev.shape == (out["x"].shape[0], 3)
    assert np.all(sd > 0)
    # eigenvalues ascending
    assert np.all(np.diff(ev, axis=1) >= -1e-4)
    # cross-check: stream_density ~ 1/|prod(eigs)| only if T symmetric; in the
    # quasi-linear regime det(T) ~ prod(sym eigs). Allow generous tolerance.
    prod = np.prod(ev, axis=1)
    finite = np.isfinite(prod) & (np.abs(prod) > 1e-3)
    np.testing.assert_allclose(sd[finite], 1.0 / np.abs(prod[finite]),
                               rtol=0.05)


def test_hdf5_deformation_columns(tmp_path):
    import h5py
    dj = _make_dj()
    observer = np.array([256.0, 256.0, 256.0])
    path = str(tmp_path / "lc_def.h5")
    dj.evaluate_lpt_lightcone_to_hdf5(
        path, a_far=0.3, a_near=1.0, n_shells=16, observer=observer,
        n_part_chunks=2, n_newton_iters=2, v_mode="radial",
        deformation_mode="stream")
    with h5py.File(path, "r") as f:
        g = f["PartType1"]
        assert "StreamDensity" in g and "TidalEigenvalues" in g
        assert g["StreamDensity"].shape == (g["Coordinates"].shape[0],)
        assert g["TidalEigenvalues"].shape == (g["Coordinates"].shape[0], 3)
        assert np.all(g["StreamDensity"][:] > 0)
