"""Rapid cosmology-refresh of an existing lightcone catalogue.

Given a "scene" (Ψ₁, Ψ₂ fields + the random seed that produced them) and a
lightcone catalogue computed at one fiducial cosmology, this module rebuilds
the catalogue at a *different* cosmology without re-running the chi_to_a
inverse, the replica enumeration, or the residual-tol filter — we already
know which (particle, replica) pairs cross and what their old crossing times
were, so one Newton iteration from the saved ``a_cross_old`` gets us to fp32
ulp in the new cosmology.

Two modes:

  - ``mode='fixed_psi'``: keep Ψ₁, Ψ₂ as-is. Exact for changes that don't
    touch the transfer function T(k) — primarily ``w0``, ``wa``, growth-only
    Ω_de. A controlled approximation otherwise. Cheapest: ~50–70 s at N=512
    (no 2LPT recompute, no IC re-sample).
  - ``mode='exact'``: regenerate the white noise from the saved seed, dress
    it with the new T(k), recompute Ψ₁, Ψ₂ via ``with_lpt(n_order)``. Exact
    for any θ. Costs roughly the same as a fresh radial-sort lightcone minus
    the chi_to_a seed search.

A third strategy — closed-form Ψ rescaling for σ₈-only changes — lives
inside ``'fixed_psi'`` mode (``sigma8_rescale=True``).

The inner kernel ``_make_refresh_kernel`` is jit-traceable in the Cosmology
PyTree, so callers in autodiff workflows can ``jax.jvp`` or ``jax.grad``
through it directly.
"""
from __future__ import annotations

from functools import partial

import numpy as onp
import jax
import jax.numpy as jnp

from ..core.io import compression_kwargs as _compression_kwargs
from ..core.io import read_lightcone_header as _read_lightcone_header
from ..core.io import LightconeHDF5Writer, lightcone_particle_mass
from .lightcone import enumerate_replicas


__all__ = [
    "save_lpt_scene",
    "load_lpt_scene",
    "refresh_lightcone_cosmology",
    "refresh_lightcone_arrays",
    "load_refresh_inputs",
]


# Cosmology Header attrs we round-trip through the scene file.
_COSMO_KEYS = ("Omega_c", "Omega_b", "h", "sigma8", "n_s",
               "Omega_k", "w0", "wa")
# DiscoDJ IC re-generation params we round-trip so 'exact' mode is bit-stable.
_IC_KEYS = ("seed", "white_noise_space", "k_order_fourier",
            "enforce_mode_stability", "sphere_mode", "fix_std")


# ---------------------------------------------------------------------------
# Scene file I/O
# ---------------------------------------------------------------------------

def save_lpt_scene(dj, path: str, *,
                   include_psi: bool = True,
                   include_fphi: bool = True,
                   ic_params: dict | None = None,
                   transfer_function: str = "Eisenstein-Hu",
                   compression: str = "zstd",
                   storage_chunk_rows: int = 1 << 20) -> None:
    """Persist the LPT scene from a `DiscoDJ` to HDF5.

    Writes:
      /Header   attrs (res, boxsize, dim, n_order, ic params, fiducial cosmology)
      /Psi/psi_1, /Psi/psi_2     fp32 (if include_psi)
      /ICs/fphi_ini              complex64 (if include_fphi)

    :param dj: a `DiscoDJ` after ``with_lpt(n_order)`` has been called.
    :param path: output HDF5 file path.
    :param include_psi: persist the displacement fields. Required for
        ``mode='fixed_psi'`` refreshes. ~3 GB at N=512 / ~26 GB at N=1024.
    :param include_fphi: persist the initial potential. Useful as a fallback
        for ``mode='exact'`` if the seed-based regeneration ever drifts.
        Small: ~63 MB at N=512 / ~500 MB at N=1024.
    :param ic_params: dict of the kwargs that were passed to
        ``DiscoDJ.with_ics`` so 'exact' mode can re-run with bit-identical
        IC sampling. If None, falls back to defaults; the round-trip is
        still bit-stable as long as the user used those defaults.
    :param transfer_function: which transfer function the fiducial scene was
        built with (passed to ``with_linear_ps`` in 'exact' mode).
    """
    import h5py
    assert dj._lpt is not None, "with_lpt(...) must be called before save_lpt_scene."

    ds_kwargs = _compression_kwargs(compression)
    psi_1 = onp.asarray(dj._lpt.psi["psi_1"]).astype(onp.float32)
    psi_2 = onp.asarray(dj._lpt.psi.get("psi_2"))
    if psi_2.ndim == 0:
        psi_2 = None
    if psi_2 is not None:
        psi_2 = psi_2.astype(onp.float32)
    fphi = onp.asarray(dj._ics["fphi"])

    # IC params we record so 'exact' mode can reproduce the white noise
    # bit-for-bit. We don't try to peek into the prior ``with_ics`` call —
    # the caller passes the dict they used (defaults if unspecified).
    if ic_params is None:
        ic_params = {}
    ic_full = {k: ic_params.get(k, _IC_DEFAULTS[k]) for k in _IC_KEYS}

    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["res"] = int(dj.res)
        h.attrs["boxsize"] = float(dj.boxsize)
        h.attrs["dim"] = int(dj.dim)
        h.attrs["n_order"] = int(dj._lpt.n_order)
        h.attrs["dtype_num"] = 32 if dj.dtype == onp.float32 else 64
        h.attrs["transfer_function"] = transfer_function
        # IC params
        for k in _IC_KEYS:
            v = ic_full[k]
            h.attrs[f"ic_{k}"] = ("<none>" if v is None else v)
        # Fiducial cosmology
        cosmo = dj.cosmo
        for k in _COSMO_KEYS:
            h.attrs[f"cosmo_{k}"] = float(getattr(cosmo, k))
        h.attrs["has_psi"] = bool(include_psi)
        h.attrs["has_fphi"] = bool(include_fphi)

        if include_psi:
            g = f.create_group("Psi")
            psi1_chunks = (min(storage_chunk_rows, psi_1.shape[0]),) + psi_1.shape[1:]
            g.create_dataset("psi_1", data=psi_1,
                             chunks=psi1_chunks, **ds_kwargs)
            if psi_2 is not None:
                psi2_chunks = (min(storage_chunk_rows, psi_2.shape[0]),) + psi_2.shape[1:]
                g.create_dataset("psi_2", data=psi_2,
                                 chunks=psi2_chunks, **ds_kwargs)
        if include_fphi:
            g = f.create_group("ICs")
            g.create_dataset("fphi_ini", data=fphi, **ds_kwargs)


# Fallback values for IC params if save_lpt_scene caller didn't supply them.
_IC_DEFAULTS = {
    "seed": 0,
    "white_noise_space": "real",
    "k_order_fourier": "stable",
    "enforce_mode_stability": False,
    "sphere_mode": False,
    "fix_std": None,
}


def load_lpt_scene(path: str, *, load_psi: bool = True,
                   load_fphi: bool = False) -> dict:
    """Load the scene Header + (optionally) Ψ / fphi fields from HDF5.

    :return: dict with keys: ``res``, ``boxsize``, ``dim``, ``n_order``,
        ``dtype_num``, ``transfer_function``, ``ic_params``,
        ``cosmology_params`` (suitable to splat into ``Cosmology(**...)``),
        ``has_psi``, ``has_fphi``, ``psi_1``, ``psi_2``, ``fphi_ini``.
        Field arrays are numpy arrays (cheap to upload to JAX as needed).
    """
    import h5py
    out: dict = {}
    with h5py.File(path, "r") as f:
        h = f["Header"]
        out["res"] = int(h.attrs["res"])
        out["boxsize"] = float(h.attrs["boxsize"])
        out["dim"] = int(h.attrs["dim"])
        out["n_order"] = int(h.attrs["n_order"])
        out["dtype_num"] = int(h.attrs["dtype_num"])
        out["transfer_function"] = str(h.attrs["transfer_function"])
        ic = {}
        for k in _IC_KEYS:
            attr_key = f"ic_{k}"
            if attr_key in h.attrs:
                v = h.attrs[attr_key]
                if isinstance(v, bytes):
                    v = v.decode()
                if v == "<none>":
                    v = None
                ic[k] = v
        out["ic_params"] = ic
        cosmo_params = {}
        for k in _COSMO_KEYS:
            attr_key = f"cosmo_{k}"
            if attr_key in h.attrs:
                cosmo_params[k] = float(h.attrs[attr_key])
        out["cosmology_params"] = cosmo_params
        out["has_psi"] = bool(h.attrs.get("has_psi", False))
        out["has_fphi"] = bool(h.attrs.get("has_fphi", False))

        if load_psi and out["has_psi"]:
            g = f["Psi"]
            out["psi_1"] = onp.asarray(g["psi_1"][...])
            if "psi_2" in g:
                out["psi_2"] = onp.asarray(g["psi_2"][...])
        if load_fphi and out["has_fphi"]:
            out["fphi_ini"] = onp.asarray(f["ICs/fphi_ini"][...])
    return out


# ---------------------------------------------------------------------------
# JIT kernel
# ---------------------------------------------------------------------------

def _cosmo_kernel_args(cosmo) -> dict:
    """Extract the bits a refresh kernel needs into plain JAX arrays.

    We avoid passing the `Cosmology` PyTree itself into the JIT because its
    `tree_flatten` places `_timetables` (a dict of arrays) in the *aux_data*
    slot — JAX compares aux_data by Python equality across cache lookups,
    and ``array == array`` is ambiguous, raising on the second new cosmology
    seen by the same jit.

    Returns a dict suitable for ``**`` into ``kernel(...)``. Triggers
    ``compute_timetables`` if not already populated.
    """
    # Force timetables to be built before we read the dict.
    _ = cosmo.chi(jnp.asarray(0.5))
    tt = cosmo._timetables
    return dict(
        a_tab=jnp.asarray(tt["a"]),
        Dplus_tab=jnp.asarray(tt["Dplus"]),
        Dplusda_tab=jnp.asarray(tt["Dplusda"]),
        chi_tab=jnp.asarray(tt["chi"]),
        Omega_m=jnp.asarray(cosmo.Omega_m),
        Omega_k=jnp.asarray(cosmo.Omega_k),
        Omega_de=jnp.asarray(cosmo.Omega_de),
        w0=jnp.asarray(cosmo.w0),
        wa=jnp.asarray(cosmo.wa),
    )


def _make_refresh_kernel(n_order: int, n_newton_iters: int = 1,
                         v_radial: bool = True, dtype=jnp.float32):
    """Build a jit'd vectorised per-row refresh kernel.

    The kernel takes per-row (particle_idx, replica_idx, a_cross_old), plus
    scene tensors (q_flat, psi_flats), plus plain cosmology arrays (see
    `_cosmo_kernel_args`). Returns per-row (x_new, v_out, a_new, shell_idx,
    valid).

    `n_order` is closed over so the per-particle LPT polynomial is unrolled
    in Python (no lax.scan / lax.cond — they inhibit multithreading on CPU,
    as we found with `fmu2_sym`).
    """
    c_over_H0 = jnp.asarray(2997.92458, dtype=dtype)  # Mpc/h

    def _Dplus(a, a_tab, Dplus_tab):
        return jnp.interp(a, a_tab, Dplus_tab).astype(dtype)

    def _Dplusda(a, a_tab, Dplusda_tab):
        return jnp.interp(a, a_tab, Dplusda_tab).astype(dtype)

    def _chi(a, a_tab, chi_tab):
        return jnp.interp(a, a_tab, chi_tab).astype(dtype)

    def _E(a, Omega_m, Omega_k, Omega_de, w0, wa):
        # Mirrors Cosmology.E + Omega_de_of_a (cosmology.py:294-302).
        omega_de_a = Omega_de * (a ** (-3.0 * (1.0 + w0 + wa))
                                  * jnp.exp(-3.0 * wa * (1.0 - a)))
        return jnp.sqrt(Omega_m * a ** -3 + Omega_k * a ** -2 + omega_de_a).astype(dtype)

    @jax.jit
    def kernel(particle_idx, replica_idx, a_cross_old,
               q_flat, psi_flats,
               replica_offsets, observer, a_shells,
               residual_tol, a_far, a_near,
               a_tab, Dplus_tab, Dplusda_tab, chi_tab,
               Omega_m, Omega_k, Omega_de, w0, wa):
        # ---- Gather per row ----
        q = q_flat[particle_idx]                        # (batch, 3)
        psi_chunks = tuple(p[particle_idx] for p in psi_flats)
        r = replica_offsets[replica_idx]                # (batch, 3)

        a = a_cross_old.astype(dtype)
        # ---- Newton iter(s) ----
        for _ in range(n_newton_iters):
            D = _Dplus(a, a_tab, Dplus_tab)
            dDda = _Dplusda(a, a_tab, Dplusda_tab)
            E_a = _E(a, Omega_m, Omega_k, Omega_de, w0, wa)
            chi_a = _chi(a, a_tab, chi_tab)
            x = q + r
            dxda = jnp.zeros_like(q)
            D_pow_n = D
            D_pow_nm1 = jnp.ones_like(D)
            for n_idx, psi_n in enumerate(psi_chunks):
                n = n_idx + 1
                x = x + D_pow_n[:, None] * psi_n
                dxda = dxda + (n * D_pow_nm1 * dDda)[:, None] * psi_n
                D_pow_n = D_pow_n * D
                D_pow_nm1 = D_pow_nm1 * D
            diff = x - observer[None, :]
            d = jnp.linalg.norm(diff, axis=-1)
            f = d - chi_a
            d_safe = jnp.where(d > 1e-10, d, 1.0)
            d_d_da = jnp.sum(diff * dxda, axis=-1) / d_safe
            dchi_da = -c_over_H0 / (a ** 2 * E_a)
            df_da = d_d_da - dchi_da
            df_da_safe = jnp.where(jnp.abs(df_da) > 1e-30, df_da, 1.0)
            a = a - f / df_da_safe
            a = jnp.clip(a, a_far, a_near)

        # ---- Final x, v at refined a ----
        D = _Dplus(a, a_tab, Dplus_tab)
        dDda = _Dplusda(a, a_tab, Dplusda_tab)
        E_a = _E(a, Omega_m, Omega_k, Omega_de, w0, wa)
        # growth_rate(a) = dD/da · a / D ; dD_t_over_D = growth · E · a²
        D_safe = jnp.where(jnp.abs(D) > 1e-30, D, 1.0)
        growth = (dDda * a / D_safe).astype(dtype)
        dD_t_over_D = growth * E_a * a ** 2
        x_final = q + r
        v_final = jnp.zeros_like(q)
        D_pow_n = D
        for n_idx, psi_n in enumerate(psi_chunks):
            n = n_idx + 1
            x_final = x_final + D_pow_n[:, None] * psi_n
            v_final = v_final + (n * D_pow_n * dD_t_over_D)[:, None] * psi_n
            D_pow_n = D_pow_n * D

        # ---- Residual + valid ----
        chi_a = _chi(a, a_tab, chi_tab)
        diff_obs = x_final - observer[None, :]
        d_final = jnp.linalg.norm(diff_obs, axis=-1)
        residual = jnp.abs(d_final - chi_a)
        valid = (residual < residual_tol) & (a > a_far) & (a < a_near)

        # ---- Optional radial projection ----
        if v_radial:
            d_safe = jnp.where(d_final > 1e-10, d_final, 1.0)
            v_out = jnp.sum(v_final * diff_obs, axis=-1) / d_safe
        else:
            v_out = v_final

        # ---- Shell index ----
        shell_idx = jnp.searchsorted(a_shells, a) - 1
        shell_idx = jnp.clip(shell_idx, 0, a_shells.shape[0] - 2).astype(jnp.int16)

        return x_final, v_out, a, shell_idx, valid

    return kernel


# ---------------------------------------------------------------------------
# In-memory autodiff entry point
# ---------------------------------------------------------------------------

def refresh_lightcone_arrays(
    particle_idx,
    replica_idx,
    a_cross_old,
    q_flat,
    psi_flats,
    replica_offsets,
    observer,
    a_shells,
    new_cosmology,
    *,
    a_far: float,
    a_near: float,
    n_newton_iters: int = 1,
    residual_tol: float = 1e-1,
    v_mode: str = "radial",
    dtype=jnp.float32,
) -> dict:
    """Refresh a lightcone catalogue in memory at a new cosmology.

    All inputs and outputs are JAX arrays — no file I/O. Use this for
    autodiff workflows (Fisher, JVP, grad over cosmological parameters):
    build the new cosmology from traced parameters, call
    ``compute_timetables(None)`` on it, then call this function. Gradients
    flow cleanly through the cosmology PyTree because the kernel only sees
    plain JAX arrays (extracted via ``_cosmo_kernel_args``).

    :param particle_idx: (M,) int — gather index into q_flat and each psi_flat.
        Read from ``LagrangianParticleIndex`` of an existing lightcone.
    :param replica_idx: (M,) int — gather index into replica_offsets.
    :param a_cross_old: (M,) float — fiducial crossing scale factors. Used
        as the Newton seed.
    :param q_flat: (N_part, 3) float — Lagrangian-grid positions.
    :param psi_flats: tuple of (N_part, 3) float arrays, length n_order.
    :param replica_offsets: (R, 3) float — replica vectors r·L.
    :param observer: (3,) float — observer position in box coords.
    :param a_shells: (n_shells+1,) float — ascending shell-edge scale factors.
    :param new_cosmology: ``Cosmology`` instance with ``compute_timetables``
        already called (we don't call it inside in case the caller is in
        an autodiff context).
    :param a_far, a_near: clipping bounds for Newton-refined ``a``.

    :return: dict with keys
        - ``'x'``: (M, 3) float — crossing positions at the new cosmology.
        - ``'v_radial'`` or ``'v'``: (M,) or (M, 3) — velocity (radial scalar
          if ``v_mode='radial'``, else 3-vector).
        - ``'a_cross'``: (M,) — refined crossing scale factor.
        - ``'shell_idx'``: (M,) int16.
        - ``'valid'``: (M,) bool — rows that passed the residual + interior
          checks. Caller can ``jnp.where(valid, ...)`` or compact host-side.

    Example: compute ∂lightcone/∂w0 via JVP::

        cosmo_fid = Cosmology(..., w0=-1.0).compute_timetables(None)
        primal, tangent = jax.jvp(
            lambda c: refresh_lightcone_arrays(..., new_cosmology=c, ...),
            (cosmo_fid,),
            (cosmo_fid_tangent,),   # zero everywhere except .w0 = 1.0
        )
        # tangent['a_cross'] is ∂a_cross/∂w0 at the fiducial point
    """
    assert v_mode in ("radial", "full"), v_mode
    v_is_radial = (v_mode == "radial")
    n_order = len(psi_flats)
    cosmo_args = _cosmo_kernel_args(new_cosmology)
    kernel = _make_refresh_kernel(n_order, n_newton_iters=n_newton_iters,
                                  v_radial=v_is_radial, dtype=dtype)
    x, v_out, a_new, shell_idx, valid = kernel(
        particle_idx, replica_idx, a_cross_old,
        q_flat, psi_flats,
        replica_offsets, observer, a_shells,
        jnp.asarray(residual_tol, dtype=dtype),
        jnp.asarray(a_far, dtype=dtype),
        jnp.asarray(a_near, dtype=dtype),
        **cosmo_args,
    )
    out = {"x": x, "a_cross": a_new, "shell_idx": shell_idx, "valid": valid}
    out["v_radial" if v_is_radial else "v"] = v_out
    return out


def load_refresh_inputs(input_lightcone: str) -> dict:
    """Read the per-row arrays needed by ``refresh_lightcone_arrays`` from an
    existing lightcone HDF5: ``particle_idx``, ``replica_idx``, ``a_cross``.

    Returns numpy arrays (caller can ``jnp.asarray`` to push to device).
    Also returns the input ``Header`` dict for survey-geometry metadata.
    """
    import h5py
    out: dict = {"header": _read_lightcone_header(input_lightcone)}
    with h5py.File(input_lightcone, "r") as f:
        g = f["PartType1"]
        out["particle_idx"] = onp.asarray(g["LagrangianParticleIndex"][...])
        out["replica_idx"] = onp.asarray(g["ReplicaIndex"][...])
        out["a_cross"] = onp.asarray(g["ScaleFactor"][...])
    return out


# ---------------------------------------------------------------------------
# File-to-file refresh
# ---------------------------------------------------------------------------

def _build_dj_from_scene(scene: dict, new_cosmo) -> "DiscoDJ":
    """Construct a fresh `DiscoDJ` from a scene dict + new cosmology.

    Used by `mode='exact'` to regenerate Ψ₁, Ψ₂ at the new cosmology
    deterministically (same seed → same white noise → new transfer function
    → new fphi → new Ψ).
    """
    # Imported here to avoid a circular import at module top.
    from ..disco_dj import DiscoDJ

    res = scene["res"]
    boxsize = scene["boxsize"]
    dim = scene["dim"]
    n_order = scene["n_order"]
    ic = scene["ic_params"]

    dj = DiscoDJ(dim=dim, res=res, boxsize=boxsize, cosmo=new_cosmo,
                 precision="single" if scene["dtype_num"] == 32 else "double")
    dj = dj.with_timetables()
    dj = dj.with_linear_ps(transfer_function=scene["transfer_function"])
    # Only forward the IC kwargs that `with_ics` accepts.
    ic_kwargs = {k: v for k, v in ic.items() if v is not None or k == "fix_std"}
    dj = dj.with_ics(**ic_kwargs)
    dj = dj.with_lpt(n_order=n_order)
    return dj


def refresh_lightcone_cosmology(
    scene_path: str,
    input_lightcone: str,
    output_lightcone: str,
    new_cosmology,
    *,
    mode: str = "fixed_psi",
    sigma8_rescale: bool = True,
    observer=None,
    a_far: float | None = None,
    a_near: float | None = None,
    n_shells: int | None = None,
    batch_size: int = 1 << 21,
    v_mode: str | None = None,
    residual_tol: float = 1e-1,
    n_newton_iters: int = 1,
    compression: str = "zstd",
    storage_chunk_rows: int = 1 << 20,
    verbose: bool = False,
) -> dict:
    """Recompute a lightcone catalogue at a new cosmology.

    See module docstring for the algorithm. Reads the input HDF5 in row
    batches, runs the jit'd refresh kernel, appends valid rows to the output
    HDF5 with the same schema. Defaults for observer / a_far / a_near /
    n_shells are derived from the input Header.

    :param scene_path: path to a scene HDF5 written by `save_lpt_scene`.
    :param input_lightcone: existing lightcone HDF5 at the fiducial cosmology.
        Must contain ``particle_idx`` (= ``LagrangianParticleIndex``),
        ``ReplicaIndex``, and ``ScaleFactor`` datasets.
    :param output_lightcone: output HDF5 path. Schema matches the input.
    :param new_cosmology: a `Cosmology` instance (timetables get built on
        first interpolation call if not already computed).
    :param mode: ``'fixed_psi'`` reuses scene's Ψ₁, Ψ₂ verbatim (exact iff
        new cosmology only differs in growth-only params; cheap). ``'exact'``
        rebuilds Ψ₁, Ψ₂ in the new cosmology from the saved seed.
    :param sigma8_rescale: if True and mode='fixed_psi', rescale Ψ by
        ``σ₈_new / σ₈_fid`` (Ψ₂ by the square). Exact for σ₈-only changes.
    :param batch_size: row count per HDF5 read batch and per kernel call.
        2 M rows ≈ 0.27 GB of intermediate buffers — fits comfortably.
    :return: ``{"n_particles_in": int, "n_particles_out": int,
                 "n_replicas": int, "mode": str}``.
    """
    import h5py
    from ..cosmology.cosmology import Cosmology
    assert mode in ("fixed_psi", "exact"), (
        f"mode must be 'fixed_psi' or 'exact', got {mode!r}")

    # ---- Read scene + input headers ----
    scene = load_lpt_scene(scene_path, load_psi=(mode == "fixed_psi"),
                           load_fphi=False)
    in_hdr = _read_lightcone_header(input_lightcone)
    n_in = int(in_hdr["n_particles"])

    res = scene["res"]
    boxsize = scene["boxsize"]
    n_order = scene["n_order"]
    dtype = jnp.float32 if scene["dtype_num"] == 32 else jnp.float64

    # ---- Survey geometry: derive defaults from input lightcone ----
    if observer is None:
        observer_host = onp.asarray(in_hdr["Observer"], dtype=onp.float64)
    else:
        observer_host = onp.asarray(observer, dtype=onp.float64)

    # Read all per-row a_cross to derive shell range if not given.
    # If memory is a concern we could read in batches and accumulate min/max,
    # but for now reading a single float32 array of length n_in is cheap (4 GB
    # at N=1024 worst case). Done once before the kernel runs.
    with h5py.File(input_lightcone, "r") as f:
        a_min_in = float(f["PartType1/ScaleFactor"][...].min()) if n_in else 1.0
        a_max_in = float(f["PartType1/ScaleFactor"][...].max()) if n_in else 1.0
    if a_far is None:
        a_far = a_min_in
    if a_near is None:
        a_near = a_max_in
    if n_shells is None:
        # Derive shell count from input shell_idx if present, else default.
        with h5py.File(input_lightcone, "r") as f:
            if "PartType1/ShellIndex" in f:
                n_shells = int(f["PartType1/ShellIndex"][...].max()) + 1
            else:
                n_shells = 128
    a_shells_host = onp.geomspace(a_far, a_near, n_shells + 1).astype(onp.float64)
    chi_min = float(onp.asarray(jax.device_get(
        new_cosmology.chi(jnp.asarray(a_near))))[()])
    chi_max = float(onp.asarray(jax.device_get(
        new_cosmology.chi(jnp.asarray(a_far))))[()])

    if v_mode is None:
        v_mode = in_hdr.get("v_mode") or "radial"
    assert v_mode in ("radial", "full"), f"v_mode must be 'radial' or 'full', got {v_mode!r}"
    v_is_radial = (v_mode == "radial")
    keep_particle_idx = in_hdr.get("has_particle_idx", False)
    if not keep_particle_idx:
        raise RuntimeError(
            "Input lightcone does not contain LagrangianParticleIndex; "
            "refresh requires it. Re-run the fiducial lightcone with "
            "keep_particle_idx=True.")

    # ---- Build ψ for the new cosmology ----
    if mode == "fixed_psi":
        if not scene["has_psi"]:
            raise RuntimeError(
                "Scene does not contain Ψ fields (has_psi=False); "
                "use mode='exact' or rebuild the scene with include_psi=True.")
        psi_1 = jnp.asarray(scene["psi_1"], dtype=dtype).reshape(-1, 3)
        psi_2 = jnp.asarray(scene["psi_2"], dtype=dtype).reshape(-1, 3) \
                if "psi_2" in scene else None
        # σ₈ rescale: if only σ₈ changed, Ψ scales linearly and Ψ⁽²⁾ as σ₈².
        if sigma8_rescale:
            s_fid = scene["cosmology_params"].get("sigma8", None)
            s_new = float(new_cosmology.sigma8)
            if s_fid is not None and abs(s_new - s_fid) / max(s_fid, 1e-12) > 1e-12:
                ratio = s_new / s_fid
                psi_1 = psi_1 * jnp.asarray(ratio, dtype=dtype)
                if psi_2 is not None:
                    psi_2 = psi_2 * jnp.asarray(ratio * ratio, dtype=dtype)
        psi_flats = (psi_1,) if psi_2 is None else (psi_1, psi_2)
        n_order_eff = len(psi_flats)
    else:
        # 'exact' mode: re-instantiate a DiscoDJ at the new cosmology and
        # run the LPT chain. The seed-driven white noise is identical to
        # the fiducial run, so the only difference is sqrt(P_new(k)).
        dj_new = _build_dj_from_scene(scene, new_cosmology)
        psi_flats = tuple(
            jnp.asarray(dj_new._lpt.psi[f"psi_{n}"], dtype=dtype).reshape(-1, 3)
            for n in range(1, n_order + 1)
        )
        n_order_eff = n_order

    # q_flat is derived from res + boxsize alone — same code path as dj.q.
    q_flat = _build_q_flat(res, boxsize, dtype)

    # ---- Replica enumeration (host-side AABB filter) ----
    replicas = enumerate_replicas(boxsize, observer_host, chi_min, chi_max)
    R = int(replicas.shape[0])
    if R == 0:
        # No replicas intersect; write empty output with right schema and return.
        _write_empty_output(output_lightcone, output_path=output_lightcone,
                            scene=scene, new_cosmology=new_cosmology,
                            observer_host=observer_host,
                            v_is_radial=v_is_radial,
                            keep_particle_idx=keep_particle_idx,
                            compression=compression,
                            storage_chunk_rows=storage_chunk_rows)
        return {"n_particles_in": n_in, "n_particles_out": 0,
                "n_replicas": 0, "mode": mode}
    replica_offsets = jnp.asarray(replicas, dtype=dtype) * boxsize

    # ---- JIT kernel ----
    kernel = _make_refresh_kernel(n_order_eff,
                                  n_newton_iters=n_newton_iters,
                                  v_radial=v_is_radial,
                                  dtype=dtype)

    # Extract plain JAX arrays from the Cosmology so the JIT cache doesn't
    # try to compare aux_data dicts containing arrays (see _cosmo_kernel_args).
    cosmo_args = _cosmo_kernel_args(new_cosmology)

    observer_jax = jnp.asarray(observer_host, dtype=dtype)
    a_shells_d = jnp.asarray(a_shells_host, dtype=dtype)
    a_far_d = jnp.asarray(a_far, dtype=dtype)
    a_near_d = jnp.asarray(a_near, dtype=dtype)
    residual_tol_d = jnp.asarray(residual_tol, dtype=dtype)

    # ---- Output file setup (shared LightconeHDF5Writer) ----
    n_part_per_replica = int(res ** scene["dim"])
    particle_mass = lightcone_particle_mass(new_cosmology.Omega_m, boxsize,
                                            n_part_per_replica)

    header_attrs = {
        "LightconeMode": 1,
        "Observer": observer_host,
        "BoxSize": boxsize,
        "Omega0": float(new_cosmology.Omega_m),
        "OmegaLambda": float(new_cosmology.Omega_de),
        "HubbleParam": float(new_cosmology.h),
        "MassTable": [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0],
        "NumPart_PerReplica": n_part_per_replica,
        "NumFilesPerSnapshot": 1,
        "Time": 1.0,
        "RefreshMode": mode,
    }

    with h5py.File(output_lightcone, "w") as h5f, \
         h5py.File(input_lightcone, "r") as h5in:
        writer = LightconeHDF5Writer.open(
            h5f, header_attrs=header_attrs, particle_mass=particle_mass,
            v_is_radial=v_is_radial, keep_particle_idx=True,
            compression=compression, storage_chunk_rows=storage_chunk_rows)

        in_lpid = h5in["PartType1/LagrangianParticleIndex"]
        in_rep = h5in["PartType1/ReplicaIndex"]
        in_a = h5in["PartType1/ScaleFactor"]

        # ---- Main loop: batches of input rows ----
        for start in range(0, n_in, batch_size):
            end = min(start + batch_size, n_in)
            pid_batch = onp.asarray(in_lpid[start:end])
            rep_batch = onp.asarray(in_rep[start:end])
            a_batch = onp.asarray(in_a[start:end])

            x_new, v_out, a_new, shell_new, valid = kernel(
                jnp.asarray(pid_batch, dtype=jnp.int32),
                jnp.asarray(rep_batch, dtype=jnp.int32),
                jnp.asarray(a_batch, dtype=dtype),
                q_flat, psi_flats,
                replica_offsets, observer_jax, a_shells_d,
                residual_tol_d, a_far_d, a_near_d,
                **cosmo_args,
            )
            valid_np = onp.asarray(valid)
            if valid_np.any():
                sel = onp.where(valid_np)[0]
                x_c = onp.asarray(x_new)[sel, :].astype(onp.float32)
                if v_is_radial:
                    v_c = onp.asarray(v_out)[sel].astype(onp.float32)
                else:
                    v_c = onp.asarray(v_out)[sel, :].astype(onp.float32)
                a_c = onp.asarray(a_new)[sel].astype(onp.float32)
                shell_c = onp.asarray(shell_new)[sel].astype(onp.int16)
                rep_c = rep_batch[sel].astype(onp.int16)
                lpid_c = pid_batch[sel].astype(onp.int32)
                writer.append(x=x_c, v=v_c, a=a_c, replica_idx=rep_c,
                              shell_idx=shell_c, particle_idx=lpid_c)
            if verbose:
                print(f"  batch {start//batch_size + 1} "
                      f"({start}-{end}): {int(valid_np.sum())} valid, "
                      f"{writer.total} total", flush=True)

        writer.close()
        total_out = writer.total

    if verbose:
        print(f"Refresh done. Wrote {total_out} of {n_in} rows to "
              f"{output_lightcone}.", flush=True)
    return {"n_particles_in": n_in, "n_particles_out": int(total_out),
            "n_replicas": R, "mode": mode}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_q_flat(res: int, boxsize: float, dtype) -> jnp.ndarray:
    """Reconstruct the Lagrangian grid (res, res, res, 3) flattened to
    (N_part, 3). Same logic as `DiscoDJ.q` but standalone so refresh doesn't
    need a `DiscoDJ` instance."""
    dx = boxsize / res
    axes = [onp.arange(res, dtype=onp.float32) * dx for _ in range(3)]
    grids = onp.meshgrid(*axes, indexing="ij")  # 3 arrays of (res, res, res)
    q = onp.stack(grids, axis=-1).reshape(-1, 3)
    return jnp.asarray(q, dtype=dtype)


def _write_empty_output(*, output_path, scene, new_cosmology,
                        observer_host, v_is_radial, keep_particle_idx,
                        compression, storage_chunk_rows):
    """Write a well-formed empty lightcone HDF5 when no replicas intersect."""
    import h5py
    boxsize = scene["boxsize"]
    n_part_per_replica = int(scene["res"] ** scene["dim"])
    particle_mass = lightcone_particle_mass(new_cosmology.Omega_m, boxsize,
                                            n_part_per_replica)
    header_attrs = {
        "LightconeMode": 1,
        "Observer": observer_host,
        "BoxSize": boxsize,
        "Omega0": float(new_cosmology.Omega_m),
        "OmegaLambda": float(new_cosmology.Omega_de),
        "HubbleParam": float(new_cosmology.h),
        "MassTable": [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0],
        "NumPart_PerReplica": n_part_per_replica,
        "NumFilesPerSnapshot": 1,
        "Time": 1.0,
    }
    with h5py.File(output_path, "w") as f:
        writer = LightconeHDF5Writer.open(
            f, header_attrs=header_attrs, particle_mass=particle_mass,
            v_is_radial=v_is_radial, keep_particle_idx=keep_particle_idx,
            compression=compression, storage_chunk_rows=storage_chunk_rows)
        writer.close()
