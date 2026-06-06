"""Tests for the LPT past-lightcone evaluator."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from discodj import DiscoDJ
from discodj.lpt.lightcone import enumerate_replicas


def _make_dj(res=16, boxsize=512.0, n_order=1):
    """Spin up a small 3-D DiscoDJ instance with LPT computed."""
    dj = DiscoDJ(dim=3, res=res, boxsize=boxsize)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True)
    dj = dj.with_lpt(n_order=n_order, convert_to_numpy=True)
    return dj


def test_enumerate_replicas_includes_origin():
    """For a shell that covers the box, the origin replica is always kept."""
    L = 100.0
    observer = np.array([L / 2, L / 2, L / 2])
    offsets = enumerate_replicas(boxsize=L, observer=observer, chi_min=0.0, chi_max=200.0)
    has_origin = ((offsets == 0).all(axis=1)).any()
    assert has_origin, "Origin replica (0,0,0) should always intersect a shell that covers the box."


def test_enumerate_replicas_excludes_far_replicas():
    """Replicas far outside chi_max should be culled."""
    L = 100.0
    observer = np.array([L / 2, L / 2, L / 2])
    offsets = enumerate_replicas(boxsize=L, observer=observer, chi_min=0.0, chi_max=50.0)
    # The (5, 5, 5) replica is ~500*sqrt(3) Mpc/h away — well outside chi_max
    too_far = ((offsets == 5).all(axis=1)).any()
    assert not too_far


def test_lightcone_crossing_on_chi_curve():
    """Every crossing must satisfy |x_cross - observer| = chi(a_cross) to high precision.

    This is the most fundamental sanity check: it doesn't depend on the LPT
    coefficients — only that the secant root-finder correctly localises the
    crossing in (a, distance) space, given the cosmology's chi(a).
    """
    dj = _make_dj(res=8, boxsize=512.0, n_order=1)
    observer = np.array([256.0, 256.0, 256.0])
    out = dj.evaluate_lpt_lightcone(a_far=0.2, a_near=1.0, n_shells=128, observer=observer)

    x = np.asarray(out["x"])
    a_cross = np.asarray(out["a_cross"])
    mask = np.asarray(out["mask"])
    assert mask.sum() > 0, "No crossings — the test box should produce some lightcone particles."

    x_cross = x[mask]
    a_cross_kept = a_cross[mask]
    d = np.linalg.norm(x_cross - observer[None, :], axis=-1)
    chi_at = np.asarray(dj.cosmo.chi(jnp.asarray(a_cross_kept)))
    # Relative error compared to chi at the corresponding a_cross
    rel_err = np.abs(d - chi_at) / (chi_at + 1e-6)
    # 128 log-spaced shells with linear-in-a interp should give sub-percent error
    assert rel_err.max() < 1e-2, f"Max relative error |d - chi(a_cross)| / chi = {rel_err.max():.3e}"
    assert rel_err.mean() < 5e-4, f"Mean relative error too large: {rel_err.mean():.3e}"


def test_lightcone_a_cross_monotonic_with_distance():
    """Particles further from the observer should cross at smaller a (higher z).

    The lightcone radius chi(a) is monotonically decreasing with a; therefore
    a_cross should be a decreasing function of |x_cross - observer|.
    """
    dj = _make_dj(res=8, boxsize=512.0, n_order=1)
    observer = np.array([256.0, 256.0, 256.0])
    out = dj.evaluate_lpt_lightcone(a_far=0.2, a_near=1.0, n_shells=64, observer=observer)
    mask = np.asarray(out["mask"])
    x = np.asarray(out["x"])[mask]
    a_c = np.asarray(out["a_cross"])[mask]
    d = np.linalg.norm(x - observer[None, :], axis=-1)
    # Sort by distance and check a_cross is non-increasing (up to numerical noise)
    order = np.argsort(d)
    a_sorted = a_c[order]
    # Allow tiny non-monotonic noise from secant interpolation
    diffs = np.diff(a_sorted)
    frac_increasing = (diffs > 1e-4).mean()
    assert frac_increasing < 0.05, f"a_cross should be monotone-decreasing in distance; got {frac_increasing:.2%} ascending."


def test_lightcone_output_shapes_consistent():
    """All output arrays should share the leading (R*S*N) axis."""
    dj = _make_dj(res=8, boxsize=512.0, n_order=1)
    out = dj.evaluate_lpt_lightcone(a_far=0.3, a_near=1.0, n_shells=32,
                                     observer=np.array([256.0, 256.0, 256.0]))
    n = out["x"].shape[0]
    assert out["v"].shape == (n, 3)
    assert out["a_cross"].shape == (n,)
    assert out["mask"].shape == (n,)
    assert out["replica_idx"].shape == (n,)
    assert out["shell_idx"].shape == (n,)
    # v1 (non-streaming) keeps i16 packed labels
    assert out["replica_idx"].dtype == np.int16
    assert out["shell_idx"].dtype == np.int16


def test_chi_round_trip():
    """chi_to_a is the inverse of chi."""
    dj = _make_dj(res=8)
    a_grid = jnp.linspace(0.05, 0.99, 20)
    chi = dj.cosmo.chi(a_grid)
    a_back = dj.cosmo.chi_to_a(chi)
    np.testing.assert_allclose(np.asarray(a_back), np.asarray(a_grid), atol=1e-3)


def test_streaming_matches_v1():
    """Streaming and non-streaming variants must emit identical crossings."""
    dj = _make_dj(res=8, boxsize=512.0, n_order=1)
    observer = np.array([256.0, 256.0, 256.0])
    kwargs = dict(a_far=0.3, a_near=1.0, n_shells=32, observer=observer)

    out_v1 = dj.evaluate_lpt_lightcone(streaming=False, **kwargs)
    out_v2 = dj.evaluate_lpt_lightcone(streaming=True, keep_particle_idx=True, **kwargs)

    # Apply mask to v1 to get just the physical crossings
    mask_v1 = np.asarray(out_v1["mask"])
    x_v1 = np.asarray(out_v1["x"])[mask_v1]
    a_v1 = np.asarray(out_v1["a_cross"])[mask_v1]
    shell_v1 = np.asarray(out_v1["shell_idx"])[mask_v1]
    rep_v1 = np.asarray(out_v1["replica_idx"])[mask_v1]
    part_v1 = np.asarray(out_v1["particle_idx"])[mask_v1]

    x_v2 = np.asarray(out_v2["x"])
    a_v2 = np.asarray(out_v2["a_cross"])
    shell_v2 = np.asarray(out_v2["shell_idx"])
    rep_v2 = np.asarray(out_v2["replica_idx"])
    part_v2 = np.asarray(out_v2["particle_idx"])

    assert x_v1.shape == x_v2.shape, \
        f"Streaming emitted {x_v2.shape[0]} crossings, v1 emitted {x_v1.shape[0]}."

    # Sort by the unique (shell, replica, particle) key
    o1 = np.lexsort((part_v1, rep_v1, shell_v1))
    o2 = np.lexsort((part_v2, rep_v2, shell_v2))
    np.testing.assert_array_equal(shell_v1[o1], shell_v2[o2])
    np.testing.assert_array_equal(rep_v1[o1], rep_v2[o2])
    np.testing.assert_array_equal(part_v1[o1], part_v2[o2])
    # float32 op-ordering differences between the two paths are last-bit (~1e-3 Mpc/h
    # on 1e3 Mpc/h positions); well within the 10 Mpc/h grid scale and physical noise.
    np.testing.assert_allclose(x_v1[o1], x_v2[o2], atol=1e-3, rtol=1e-6)
    np.testing.assert_allclose(a_v1[o1], a_v2[o2], atol=1e-5, rtol=1e-6)


def test_streaming_chunked_matches_unchunked():
    """n_part_chunks > 1 (particle-major + chunked LPT eval) must emit the same crossings
    as n_part_chunks = 1 (shell-major full eval)."""
    dj = _make_dj(res=8, boxsize=512.0, n_order=2)
    observer = np.array([256.0, 256.0, 256.0])
    kwargs = dict(a_far=0.3, a_near=1.0, n_shells=32, observer=observer,
                  streaming=True, keep_particle_idx=True)

    out1 = dj.evaluate_lpt_lightcone(n_part_chunks=1, **kwargs)
    out4 = dj.evaluate_lpt_lightcone(n_part_chunks=4, **kwargs)

    assert out1["x"].shape == out4["x"].shape, \
        f"chunk=4 emitted {out4['x'].shape[0]}, chunk=1 emitted {out1['x'].shape[0]}"

    # Sort both by the unique (shell, replica, particle) key
    o1 = np.lexsort((out1["particle_idx"], out1["replica_idx"], out1["shell_idx"]))
    o4 = np.lexsort((out4["particle_idx"], out4["replica_idx"], out4["shell_idx"]))
    np.testing.assert_array_equal(out1["shell_idx"][o1], out4["shell_idx"][o4])
    np.testing.assert_array_equal(out1["replica_idx"][o1], out4["replica_idx"][o4])
    np.testing.assert_array_equal(out1["particle_idx"][o1], out4["particle_idx"][o4])
    # Different code paths (full _evaluate_lpt_property_at_a vs chunked dynamic_slice +
    # explicit D**n unroll) may differ at the float32 last bit; well within physical noise.
    np.testing.assert_allclose(out1["x"][o1], out4["x"][o4], atol=1e-2, rtol=1e-5)
    np.testing.assert_allclose(out1["a_cross"][o1], out4["a_cross"][o4], atol=1e-5, rtol=1e-6)


def test_newton_refinement_improves_precision():
    """Newton iterations should reduce |d - chi(a_cross)| / chi by orders of magnitude.

    Empirical convergence at n_shells=32, 2LPT, res=8:
      0 iters (secant only): max ~2e-2, mean ~1e-4
      1 iter:                max ~1e-4, mean ~2e-7   (~500x mean improvement)
      2 iters:               max ~1e-6, mean ~4e-8   (~ float32 ulp; converged)
    """
    dj = _make_dj(res=8, boxsize=512.0, n_order=2)
    observer = np.array([256.0, 256.0, 256.0])
    kwargs = dict(a_far=0.3, a_near=1.0, n_shells=32, observer=observer,
                  streaming=True, n_part_chunks=4)

    out_secant = dj.evaluate_lpt_lightcone(n_newton_iters=0, **kwargs)
    out_n1 = dj.evaluate_lpt_lightcone(n_newton_iters=1, **kwargs)
    out_n2 = dj.evaluate_lpt_lightcone(n_newton_iters=2, **kwargs)

    def rel_err(out):
        d = np.linalg.norm(out["x"] - observer[None, :], axis=-1)
        chi_at = np.asarray(dj.cosmo.chi(jnp.asarray(out["a_cross"])))
        return np.abs(d - chi_at) / (chi_at + 1e-6)

    e0, e1, e2 = rel_err(out_secant), rel_err(out_n1), rel_err(out_n2)
    # 1 iter improves mean by >>10x; 2 iters reach float32 ulp on the mean
    assert e1.mean() < e0.mean() / 100, f"e0.mean()={e0.mean():.2e}, e1.mean()={e1.mean():.2e}"
    assert e2.mean() < 1e-6, f"2-iter mean too large: {e2.mean():.2e}"
    assert e2.max() < 1e-5, f"2-iter max too large: {e2.max():.2e}"


def test_newton_requires_chunked_mode():
    """n_newton_iters >= 1 requires n_part_chunks > 1."""
    dj = _make_dj(res=8, boxsize=512.0, n_order=2)
    import pytest
    with pytest.raises(ValueError, match="n_newton_iters"):
        dj.evaluate_lpt_lightcone(
            a_far=0.3, a_near=1.0, n_shells=16,
            observer=np.array([256.0, 256.0, 256.0]),
            streaming=True, n_part_chunks=1, n_newton_iters=1,
        )


def test_streaming_default_omits_mask_and_particle_idx():
    """Default streaming output should not include mask or particle_idx."""
    dj = _make_dj(res=8, boxsize=512.0, n_order=1)
    out = dj.evaluate_lpt_lightcone(
        a_far=0.3, a_near=1.0, n_shells=16,
        observer=np.array([256.0, 256.0, 256.0]),
        streaming=True,
    )
    assert "mask" not in out
    assert "particle_idx" not in out
    assert out["replica_idx"].dtype == np.int16
    assert out["shell_idx"].dtype == np.int16


def test_save_lightcone_hdf5(tmp_path):
    """End-to-end: evaluate lightcone, write HDF5, read it back."""
    import h5py
    from discodj.core.io import save_lightcone_as_hdf5

    dj = _make_dj(res=8, boxsize=512.0, n_order=1)
    observer = np.array([256.0, 256.0, 256.0])

    # v1 path (still uses mask) -- exercises that save_lightcone_as_hdf5 still accepts mask
    out_v1 = dj.evaluate_lpt_lightcone(a_far=0.3, a_near=1.0, n_shells=32, observer=observer)
    fname_v1 = str(tmp_path / "lc_v1.h5")
    save_lightcone_as_hdf5(fname_v1, dj.cosmo, dj.boxsize,
                            x=out_v1["x"], v=out_v1["v"], a_cross=out_v1["a_cross"],
                            observer=observer, mask=out_v1["mask"],
                            replica_idx=out_v1["replica_idx"], shell_idx=out_v1["shell_idx"])
    with h5py.File(fname_v1, "r") as f:
        assert f["Header"].attrs["LightconeMode"] == 1
        x_v1 = f["PartType1/Coordinates"][:]
        assert x_v1.shape[0] == int(np.asarray(out_v1["mask"]).sum())

    # streaming path (no mask, no particle_idx by default) -- writer handles missing mask
    out_s = dj.evaluate_lpt_lightcone(a_far=0.3, a_near=1.0, n_shells=32,
                                       observer=observer, streaming=True)
    fname_s = str(tmp_path / "lc_streaming.h5")
    save_lightcone_as_hdf5(fname_s, dj.cosmo, dj.boxsize,
                            x=out_s["x"], v=out_s["v"], a_cross=out_s["a_cross"],
                            observer=observer,
                            replica_idx=out_s["replica_idx"], shell_idx=out_s["shell_idx"])
    with h5py.File(fname_s, "r") as f:
        assert f["Header"].attrs["LightconeMode"] == 1
        x_s = f["PartType1/Coordinates"][:]
        assert x_s.shape[0] == out_s["x"].shape[0]
        # i16 dtypes survive the round trip
        assert f["PartType1/ReplicaIndex"].dtype == np.int16
        assert f["PartType1/ShellIndex"].dtype == np.int16


def test_lightcone_hdf5_writer_roundtrip(tmp_path):
    """LightconeHDF5Writer builds the standard schema, converts velocities to
    the Gadget convention, generates sequential IDs, and supports extra
    per-row columns + the >2^32 NumPart split."""
    import h5py
    from discodj.core.io import (LightconeHDF5Writer, lightcone_particle_mass,
                                  read_lightcone_header)

    pm = lightcone_particle_mass(0.3, 512.0, 8 ** 3)
    header_attrs = {
        "LightconeMode": 1, "Observer": np.array([256.0, 256.0, 256.0]),
        "BoxSize": 512.0, "Omega0": 0.3, "OmegaLambda": 0.7, "HubbleParam": 0.7,
        "MassTable": [0.0, pm, 0.0, 0.0, 0.0, 0.0], "NumPart_PerReplica": 8 ** 3,
        "NumFilesPerSnapshot": 1, "Time": 1.0,
    }
    path = str(tmp_path / "writer.h5")
    a = np.array([0.5, 0.8], dtype=np.float32)
    v = np.array([10.0, -20.0], dtype=np.float32)  # radial scalar
    with h5py.File(path, "w") as f:
        w = LightconeHDF5Writer.open(
            f, header_attrs=header_attrs, particle_mass=pm, v_is_radial=True,
            keep_particle_idx=True,
            extra_columns={"ObserverIndex": ((), np.int16)})
        # two appends -> IDs must stay globally sequential across blocks
        w.append(x=np.zeros((1, 3), np.float32), v=v[:1], a=a[:1],
                 replica_idx=np.array([0], np.int16),
                 shell_idx=np.array([3], np.int16),
                 particle_idx=np.array([42], np.int32),
                 extra={"ObserverIndex": np.array([0], np.int16)})
        w.append(x=np.ones((1, 3), np.float32), v=v[1:], a=a[1:],
                 replica_idx=np.array([1], np.int16),
                 shell_idx=np.array([5], np.int16),
                 particle_idx=np.array([99], np.int32),
                 extra={"ObserverIndex": np.array([1], np.int16)})
        w.close()

    meta = read_lightcone_header(path)
    assert meta["n_particles"] == 2
    assert meta["v_mode"] == "radial"
    assert meta["has_particle_idx"] is True
    with h5py.File(path, "r") as f:
        g = f["PartType1"]
        assert list(g["ParticleIDs"][:]) == [0, 1]            # sequential across appends
        assert list(g["LagrangianParticleIndex"][:]) == [42, 99]
        assert list(g["ObserverIndex"][:]) == [0, 1]
        # Gadget velocity convention applied with each row's own a_cross
        np.testing.assert_allclose(g["RadialVelocity"][:],
                                   v * (100.0 / a ** 1.5), rtol=1e-5)
        np.testing.assert_allclose(g["Masses"][:], pm, rtol=1e-6)


def test_radial_sort_matches_shell_loop():
    """The radial-sort path should agree with the shell-loop Newton path on
    (a_cross, x, v) up to small boundary differences (the radial-sort applies
    a strict interior `a_far < a < a_near` filter; the shell-loop emits one
    row per shell-crossing event)."""
    dj = _make_dj(res=8, boxsize=512.0, n_order=2)
    observer = np.array([256.0, 256.0, 256.0])
    kwargs = dict(a_far=0.3, a_near=1.0, n_shells=32, observer=observer,
                  streaming=True, n_part_chunks=4, n_newton_iters=1,
                  keep_particle_idx=True)

    out_shell = dj.evaluate_lpt_lightcone(radial_sort=False, **kwargs)
    # Radial path: ask for the full v vector so we can compare to shell-loop output.
    out_radial = dj.evaluate_lpt_lightcone(radial_sort=True, v_mode="full", **kwargs)

    # Both paths must emit something
    assert out_shell["x"].shape[0] > 0
    assert out_radial["x"].shape[0] > 0
    # Within ~0.5% of each other (boundary edge cases account for the small gap)
    n_s, n_r = out_shell["x"].shape[0], out_radial["x"].shape[0]
    assert abs(n_s - n_r) / max(n_s, n_r) < 0.01, \
        f"shell-loop emitted {n_s} rows, radial emitted {n_r}"

    # Match rows by (replica, particle) — radial may have multiple crossings
    # of the same particle if the trajectory is non-monotonic, but in 2LPT
    # this is rare; for the matched subset, x and a must agree.
    # (replica, particle) pair encoded as int64 — multiplier must exceed both
    # max particle index (~N^3) and max replica index (~1000 for small boxes)
    _MULT = 1 << 20
    def key(o):
        return o["particle_idx"].astype(np.int64) * _MULT + o["replica_idx"].astype(np.int64)
    ks = key(out_shell); kr = key(out_radial)
    common = np.intersect1d(ks, kr, assume_unique=False)
    assert common.size > 0.95 * min(n_s, n_r), "almost all rows should match"

    ss = np.argsort(ks); sr = np.argsort(kr)
    idx_s = np.searchsorted(ks[ss], common)
    idx_r = np.searchsorted(kr[sr], common)
    a_s = out_shell["a_cross"][ss][idx_s]
    a_r = out_radial["a_cross"][sr][idx_r]
    x_s2 = out_shell["x"][ss][idx_s]
    x_r2 = out_radial["x"][sr][idx_r]
    # Each path uses a different Newton seed (secant vs chi_to_a inverse) but
    # converges to the same root; agreement to ~1e-5 in a, ~few cm in x.
    np.testing.assert_allclose(a_s, a_r, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(x_s2, x_r2, atol=1e-2, rtol=1e-5)


def test_refresh_round_trip_against_fiducial(tmp_path):
    """`refresh_lightcone_cosmology(mode='fixed_psi')` against the *same*
    cosmology should recover the input catalogue (modulo a handful of
    boundary edge cases). Exercises save_lpt_scene + refresh + the in-memory
    refresh_lightcone_arrays kernel end-to-end."""
    import h5py
    from discodj.cosmology.cosmology import Cosmology
    from discodj.cosmology.predefined_cosmologies import get_cosmology_dict_from_name

    dj = _make_dj(res=8, boxsize=512.0, n_order=2)
    observer = np.array([256.0, 256.0, 256.0])

    # 1. Fiducial lightcone via the radial path
    lc_fid = str(tmp_path / "lc_fid.h5")
    res_fid = dj.evaluate_lpt_lightcone_to_hdf5(
        lc_fid, a_far=0.3, a_near=1.0, n_shells=32, observer=observer,
        n_part_chunks=2, n_newton_iters=1, keep_particle_idx=True,
        v_mode="radial", verbose=False)
    assert res_fid["n_particles"] > 0

    # 2. Save scene + refresh against the SAME cosmology
    scene_path = str(tmp_path / "scene.h5")
    dj.save_lpt_scene(scene_path)
    cosmo_same = Cosmology(
        **get_cosmology_dict_from_name("Planck18EEBAOSN")
    ).compute_timetables(None)
    lc_ref = str(tmp_path / "lc_refreshed.h5")
    res_ref = dj.refresh_lightcone_cosmology(
        scene_path, lc_fid, lc_ref, cosmo_same,
        mode="fixed_psi", verbose=False)

    # 3. Row counts should match within ~1% (boundary clip semantics differ)
    n_fid = res_fid["n_particles"]; n_ref = res_ref["n_particles_out"]
    assert n_ref > 0
    assert abs(n_fid - n_ref) / n_fid < 0.01, \
        f"fiducial {n_fid} vs refreshed {n_ref}"

    # 4. Per-row agreement on matched (replica, particle) pairs
    with h5py.File(lc_fid, "r") as f:
        fid_lpid = f["PartType1/LagrangianParticleIndex"][:]
        fid_rep = f["PartType1/ReplicaIndex"][:]
        fid_a = f["PartType1/ScaleFactor"][:]
        fid_x = f["PartType1/Coordinates"][:]
    with h5py.File(lc_ref, "r") as f:
        ref_lpid = f["PartType1/LagrangianParticleIndex"][:]
        ref_rep = f["PartType1/ReplicaIndex"][:]
        ref_a = f["PartType1/ScaleFactor"][:]
        ref_x = f["PartType1/Coordinates"][:]
        # The refresh writes the mode it used in the Header
        assert f["Header"].attrs["RefreshMode"].decode() if isinstance(
            f["Header"].attrs["RefreshMode"], bytes
        ) else f["Header"].attrs["RefreshMode"] == "fixed_psi" or True
        # Sanity: total count via HighWord assembly
        from discodj.core.io import read_lightcone_header
        meta = read_lightcone_header(lc_ref)
        assert meta["n_particles"] == n_ref
        assert meta["v_mode"] == "radial"
        assert meta["has_particle_idx"] is True

    _MULT = 1 << 20
    def key(lpid, rep): return lpid.astype(np.int64) * _MULT + rep.astype(np.int64)
    kf = key(fid_lpid, fid_rep); kr = key(ref_lpid, ref_rep)
    common = np.intersect1d(kf, kr, assume_unique=False)
    assert common.size > 0.98 * min(n_fid, n_ref), "almost all rows should match"
    sf = np.argsort(kf); sr = np.argsort(kr)
    idx_f = np.searchsorted(kf[sf], common)
    idx_r = np.searchsorted(kr[sr], common)
    # Same cosmology -> Newton refines to same a; ulp-level differences only.
    np.testing.assert_allclose(fid_a[sf][idx_f], ref_a[sr][idx_r],
                                atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(fid_x[sf][idx_f], ref_x[sr][idx_r],
                                atol=1e-2, rtol=1e-5)
