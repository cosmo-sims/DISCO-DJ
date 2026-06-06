"""LPT past-lightcone evaluator.

For n-LPT particles the trajectory x(a) = q + sum_k D_k(a) Psi_k(q) is closed-form
in the scale factor a through the growth factors. The past-lightcone crossing
condition |x(a) - x_obs| = chi(a) thus reduces to a per-particle 1-D root.

This module evaluates that crossing across a user-provided shell grid in a
fully jittable, vmappable, differentiable way:
  - shell grid {a_k}, k = 0..S, log-spaced from a_far to a_near
  - replica offsets (R, 3) statically filtered to those that intersect the
    active shell range [chi(a_near), chi(a_far)]
  - per shell interval and replica, linear interpolation in a between
    precomputed LPT samples + a secant step on f(a) = |x(a)-x_obs| - chi(a)
  - fixed-shape masked dump: every (shell, replica) emits (N_part, 8) of
    [x_cross, v_cross, a_cross, mask].

The mask is filtered host-side prior to writing to disk. Emitted x_cross /
v_cross / a_cross are smooth in the active region so reverse-mode through the
lightcone evaluator is well-defined.

Memory scaling: the per-(shell, replica) buffers are static of shape
(N_part, ...), so peak memory is O(S * R * N_part * 8 floats). For deep
lightcones (small a_far -> many replicas R) this can exceed device memory.
Two reductions are available without changing semantics: (i) raise a_far so
chi_max shrinks and R drops cubically; (ii) sweep shells with a Python or
lax.scan loop instead of a single vmap to keep only one (R, N_part) buffer
live at a time. The second is a v2 follow-up.
"""

from __future__ import annotations

from functools import partial

import numpy as onp
import jax
import jax.numpy as jnp

__all__ = ["enumerate_replicas", "evaluate_lpt_lightcone",
           "evaluate_lpt_lightcone_streaming",
           "evaluate_lpt_lightcone_streaming_radial",
           "evaluate_lpt_lightcone_to_hdf5_radial"]


def _per_replica_crossing(r_offset, x_k, x_kp1, v_k, v_kp1, a_k, a_kp1, chi_k, chi_kp1, observer):
    """Linear-in-a secant crossing for one shell interval and one replica.

    Inputs are full (N_part, ...) fields; the replica offset is added to positions.
    Returns (x_cross, v_cross, a_cross, crosses) each of shape (N_part, ...).
    """
    x_k_r = x_k + r_offset[None, :]
    x_kp1_r = x_kp1 + r_offset[None, :]
    d_k = jnp.linalg.norm(x_k_r - observer[None, :], axis=-1)
    d_kp1 = jnp.linalg.norm(x_kp1_r - observer[None, :], axis=-1)
    f_k = d_k - chi_k
    f_kp1 = d_kp1 - chi_kp1
    # f goes from negative (particle inside shell at a_far edge) to positive
    # (particle outside shell at a_near edge) as a grows.
    crosses = (f_k < 0) & (f_kp1 > 0)
    denom = f_kp1 - f_k
    denom_safe = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    t = jnp.where(crosses, -f_k / denom_safe, 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    a_cross = a_k + t * (a_kp1 - a_k)
    x_cross = x_k_r + t[:, None] * (x_kp1_r - x_k_r)
    v_cross = v_k + t[:, None] * (v_kp1 - v_k)
    return x_cross, v_cross, a_cross, crosses


@jax.jit
def _per_shell_kernel(x_k, x_kp1, v_k, v_kp1, a_k, a_kp1, chi_k, chi_kp1,
                       replica_offsets, observer):
    """Vmap _per_replica_crossing over replicas. Returns (R, N, ...) arrays."""
    def fn(r):
        return _per_replica_crossing(r, x_k, x_kp1, v_k, v_kp1,
                                     a_k, a_kp1, chi_k, chi_kp1, observer)
    return jax.vmap(fn)(replica_offsets)


def enumerate_replicas(boxsize: float, observer: onp.ndarray, chi_min: float, chi_max: float) -> onp.ndarray:
    """Static host-side enumeration of periodic-box replicas that intersect the active shell.

    Each replica r corresponds to translating the box by r*boxsize. We keep r iff the
    AABB [r*L, (r+1)*L] intersects the spherical shell of radii [chi_min, chi_max]
    centred at the observer.

    Returns an (R, 3) int array of replica indices.
    """
    observer = onp.asarray(observer, dtype=onp.float64)
    n_rep = int(onp.ceil(chi_max / boxsize)) + 1
    kept = []
    rs = onp.arange(-n_rep, n_rep + 1)
    for i in rs:
        for j in rs:
            for k in rs:
                lo = onp.array([i, j, k]) * boxsize  # AABB low corner
                hi = lo + boxsize
                # distance from observer to AABB
                d_per_axis_lo = onp.maximum(lo - observer, 0.0)
                d_per_axis_hi = onp.maximum(observer - hi, 0.0)
                dmin = onp.linalg.norm(onp.maximum(d_per_axis_lo, d_per_axis_hi))
                # furthest distance from observer to AABB corners
                corners = onp.array([[lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]])
                # 8 corners; pick the one farthest from observer
                d2max = 0.0
                for cx in corners[0]:
                    for cy in corners[1]:
                        for cz in corners[2]:
                            d2 = (cx - observer[0]) ** 2 + (cy - observer[1]) ** 2 + (cz - observer[2]) ** 2
                            if d2 > d2max:
                                d2max = d2
                dmax = onp.sqrt(d2max)
                if dmax >= chi_min and dmin <= chi_max:
                    kept.append([i, j, k])
    if not kept:
        return onp.zeros((0, 3), dtype=onp.int32)
    return onp.array(kept, dtype=onp.int32)


def _parse_observers(observer, boxsize: float) -> onp.ndarray:
    """Normalise the ``observer`` argument to a host ``(n_obs, 3)`` float64
    array. ``None`` -> a single box-centre observer; ``(3,)`` -> one observer;
    ``(n_obs, 3)`` -> a batch of independent observers."""
    if observer is None:
        return (onp.array([0.5, 0.5, 0.5], dtype=onp.float64) * boxsize)[None, :]
    obs = onp.asarray(observer, dtype=onp.float64)
    if obs.ndim == 1:
        assert obs.shape == (3,), "observer must be (3,) or (n_obs, 3)."
        return obs[None, :]
    assert obs.ndim == 2 and obs.shape[1] == 3, \
        "observer must be (3,) or (n_obs, 3)."
    return obs


def evaluate_lpt_lightcone(dj, a_far: float, a_near: float, n_shells: int = 64,
                            observer=None, n_order: int | None = None,
                            exact_growth: bool = False):
    """Generate LPT past-lightcone particle records.

    :param dj: DiscoDJ instance with LPT computed.
    :param a_far: scale factor at the far edge of the lightcone (smaller a, higher z).
    :param a_near: scale factor at the near edge (larger a, lower z; typically 1.0).
    :param n_shells: number of log-spaced scale-factor bins between a_far and a_near.
    :param observer: (3,) array, observer position in box coords (Mpc/h). Default: box centre.
    :param n_order: LPT order to use (default: highest computed).
    :param exact_growth: forwarded to the LPT evaluator.

    :return: dict with keys 'x' (R*S*N, 3), 'v' (R*S*N, 3), 'a_cross' (R*S*N,),
             'mask' (R*S*N,), 'replica_idx' (R*S*N,), 'shell_idx' (R*S*N,).
             v is the superconformal-time derivative dPsi/d(tildet), matching the
             convention used by save_as_hdf5.
    """
    assert dj.dim == 3, "Lightcones are 3-D only."
    assert dj._lpt is not None, "Compute LPT first via with_lpt() / compute_lpt()."

    boxsize = float(dj.boxsize)
    if observer is None:
        observer_host = onp.array([0.5, 0.5, 0.5], dtype=onp.float64) * boxsize
    else:
        observer_host = onp.asarray(observer, dtype=onp.float64)
        assert observer_host.shape == (3,), "observer must be a (3,) vector."
    observer_jax = jnp.asarray(observer_host, dtype=dj.dtype)

    # Shell grid
    dtype = dj.dtype
    a_shells = jnp.geomspace(a_far, a_near, n_shells + 1, dtype=dtype)
    chi_shells = dj.cosmo.chi(a_shells)
    chi_min = float(chi_shells[-1])  # at a_near
    chi_max = float(chi_shells[0])   # at a_far

    # Static replica enumeration (host)
    replicas = enumerate_replicas(boxsize, observer_host, chi_min, chi_max)
    R = replicas.shape[0]
    if R == 0:
        # Nothing intersects (e.g. observer far outside, very small lightcone)
        N = dj.res ** dj.dim
        zeros = jnp.zeros((0, 3), dtype=dtype)
        return {
            "x": zeros, "v": zeros,
            "a_cross": jnp.zeros((0,), dtype=dtype),
            "mask": jnp.zeros((0,), dtype=bool),
            "replica_idx": jnp.zeros((0,), dtype=jnp.int32),
            "shell_idx": jnp.zeros((0,), dtype=jnp.int32),
        }
    replica_offsets = jnp.asarray(replicas, dtype=dtype) * boxsize  # (R, 3)

    # Precompute LPT positions and velocities at each shell boundary
    # Position field (unwrapped, includes q): (N_part, 3) per a
    def pos_at(a):
        return dj.ensure_flat_shape(
            dj._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=True,
                                          exact_growth=exact_growth)
        )

    # Velocity: d Psi / d tildet (superconformal time), matches save_as_hdf5 convention
    def vel_at(a):
        return dj.ensure_flat_shape(
            dj._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=False,
                                          time_derivative=True, exact_growth=exact_growth)
        )

    x_shells = jax.vmap(pos_at)(a_shells)  # (S+1, N, 3)
    v_shells = jax.vmap(vel_at)(a_shells)  # (S+1, N, 3)

    N = x_shells.shape[1]
    S = n_shells

    def per_shell_replica(k_and_r):
        k, r_idx = k_and_r
        return _per_replica_crossing(
            replica_offsets[r_idx],
            x_shells[k], x_shells[k + 1], v_shells[k], v_shells[k + 1],
            a_shells[k], a_shells[k + 1], chi_shells[k], chi_shells[k + 1],
            observer_jax,
        )

    # Cartesian product over (shell, replica) indices, vmap over both axes
    shell_idx_grid, replica_idx_grid = jnp.meshgrid(
        jnp.arange(S, dtype=jnp.int32), jnp.arange(R, dtype=jnp.int32), indexing="ij"
    )
    pairs = (shell_idx_grid.reshape(-1), replica_idx_grid.reshape(-1))  # both (S*R,)

    x_all, v_all, a_all, mask_all = jax.vmap(per_shell_replica)(pairs)
    # shapes: (S*R, N, 3), (S*R, N, 3), (S*R, N), (S*R, N)

    # Flatten to (S*R*N, ...) and attach shell/replica/particle labels.
    # Packed dtypes: i16 for shell/replica (max ~256 each in practice).
    x_flat = x_all.reshape(-1, 3)
    v_flat = v_all.reshape(-1, 3)
    a_flat = a_all.reshape(-1)
    mask_flat = mask_all.reshape(-1)
    shell_labels = jnp.repeat(pairs[0].astype(jnp.int16), N)
    replica_labels = jnp.repeat(pairs[1].astype(jnp.int16), N)
    particle_labels = jnp.tile(jnp.arange(N, dtype=jnp.int32), S * R)

    return {
        "x": x_flat,
        "v": v_flat,
        "a_cross": a_flat,
        "mask": mask_flat,
        "replica_idx": replica_labels,
        "shell_idx": shell_labels,
        "particle_idx": particle_labels,
    }


def _make_newton_kernel(dj, n_order: int, n_newton_iters: int):
    """Build a per-shell kernel that performs secant + n Newton iterations on
    f(a) = |x_LPT(a) - x_obs| - chi(a), entirely from chunked LPT data.

    Returns ``(kernel, init_x_at)`` where:

    - ``kernel(start, chunk_size, x_k_lpt, a_k, a_kp1, chi_k, chi_kp1,
      replica_offsets, observer)`` returns
      ``((x_cross, v_cross, a_cross, crosses), x_kp1_lpt)``.
      The per-replica outputs have leading axis R (replicas), inner axis
      chunk_size. The caller carries ``x_kp1_lpt`` into the next shell as the
      new ``x_k_lpt`` so x_at the left bracket is evaluated only once per
      shell (sliding window).

    - ``init_x_at(start, chunk_size, a)`` evaluates x_LPT(a) for the chunk to
      seed the sliding window at the far bracket of shell 0.

    The seed for Newton is the linear-in-a secant solution. Final x_cross and
    v_cross are evaluated from LPT (q + sum_n D^n psi_n) at the refined a,
    NOT by linear interpolation between bracket endpoints, so neighbouring
    particles' (x, v) are individually accurate rather than carrying a shared
    linear-interp error.
    """
    cosmo = dj.cosmo
    dtype = dj.dtype
    c_over_H0 = jnp.asarray(2997.92458, dtype=dtype)  # Mpc/h
    psi_flat_all = tuple(
        dj._lpt.psi[f"psi_{n}"].reshape(-1, 3) for n in range(1, n_order + 1)
    )
    q_flat = dj.q.reshape(-1, 3)

    def _x_at(q_chunk, psi_chunks, a_scalar):
        D = cosmo.Dplus(jnp.atleast_1d(a_scalar)).astype(dtype)[0]
        x = q_chunk + 0.0
        D_pow = D
        for psi_n in psi_chunks:
            x = x + D_pow * psi_n
            D_pow = D_pow * D
        return x

    @partial(jax.jit, static_argnames=("chunk_size",))
    def init_x_at(start, chunk_size, a):
        q_chunk = jax.lax.dynamic_slice_in_dim(q_flat, start, chunk_size, axis=0)
        psi_chunks = tuple(
            jax.lax.dynamic_slice_in_dim(psi_n, start, chunk_size, axis=0)
            for psi_n in psi_flat_all
        )
        return _x_at(q_chunk, psi_chunks, a)

    @partial(jax.jit, static_argnames=("chunk_size",))
    def kernel(start, chunk_size, x_k_lpt, a_k, a_kp1, chi_k, chi_kp1,
               replica_offsets, observer):
        # Slice LPT data for this chunk
        q_chunk = jax.lax.dynamic_slice_in_dim(q_flat, start, chunk_size, axis=0)
        psi_chunks = tuple(
            jax.lax.dynamic_slice_in_dim(psi_n, start, chunk_size, axis=0)
            for psi_n in psi_flat_all
        )

        # x at the right bracket (x at left bracket is carried from prev shell)
        x_kp1_lpt = _x_at(q_chunk, psi_chunks, a_kp1)

        def per_replica(r_offset):
            # ---- Secant seed ----
            x_k_r = x_k_lpt + r_offset[None, :]
            x_kp1_r = x_kp1_lpt + r_offset[None, :]
            d_k = jnp.linalg.norm(x_k_r - observer[None, :], axis=-1)
            d_kp1 = jnp.linalg.norm(x_kp1_r - observer[None, :], axis=-1)
            f_k = d_k - chi_k
            f_kp1 = d_kp1 - chi_kp1
            crosses = (f_k < 0) & (f_kp1 > 0)
            denom = f_kp1 - f_k
            denom_safe = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
            t = jnp.clip(jnp.where(crosses, -f_k / denom_safe, 0.0), 0.0, 1.0)
            a = a_k + t * (a_kp1 - a_k)

            # ---- Newton refinement ----
            for _ in range(n_newton_iters):
                D = cosmo.Dplus(a).astype(dtype)
                dDda = cosmo.Dplusda(a).astype(dtype)
                E_a = cosmo.E(a).astype(dtype)
                chi_a = cosmo.chi(a).astype(dtype)
                # x(a) = q + r + sum_n D^n psi_n;  dx/da = sum_n n D^{n-1} dD/da psi_n
                x = q_chunk + r_offset[None, :]
                dxda = jnp.zeros_like(q_chunk)
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
                a = a - jnp.where(crosses, f / df_da_safe, 0.0)
                a = jnp.clip(a, a_k, a_kp1)

            # ---- Final x, v from LPT at refined a ----
            D = cosmo.Dplus(a).astype(dtype)
            E_a = cosmo.E(a).astype(dtype)
            growth = cosmo.growth_rate(a).astype(dtype)
            dD_t_over_D = growth * E_a * a ** 2  # (dD/dsuperconft) / D
            x_final = q_chunk + r_offset[None, :]
            v_final = jnp.zeros_like(q_chunk)
            D_pow_n = D
            for n_idx, psi_n in enumerate(psi_chunks):
                n = n_idx + 1
                x_final = x_final + D_pow_n[:, None] * psi_n
                # d/dsuperconft of D^n psi_n = n D^{n-1} (dD/dsuperconft) psi_n
                #                            = n D^n  (dD/dsuperconft)/D psi_n
                v_final = v_final + (n * D_pow_n * dD_t_over_D)[:, None] * psi_n
                D_pow_n = D_pow_n * D

            return x_final, v_final, a, crosses

        return jax.vmap(per_replica)(replica_offsets), x_kp1_lpt

    return kernel, init_x_at


def _make_chunked_lpt_eval(dj, n_order: int):
    """Build jittable per-chunk LPT position / velocity evaluators.

    Returns ``(pos_at_chunk, vel_at_chunk)``, each callable as
    ``f(a, start, chunk_size)`` where ``chunk_size`` is a static int. The
    function slices into the pre-stored ``self._lpt.psi`` fields via
    ``jax.lax.dynamic_slice_in_dim`` so the full ``(N, 3)`` arrays are never
    materialised. Only the standard non-exact-growth path is supported here
    (exact_growth uses different growth keys and a different recurrence).
    """
    psi_flat = [dj._lpt.psi[f"psi_{n}"].reshape(-1, 3) for n in range(1, n_order + 1)]
    q_flat = dj.q.reshape(-1, 3)
    cosmo = dj.cosmo
    dtype = dj.dtype

    @partial(jax.jit, static_argnames=("chunk_size",))
    def pos_at_chunk(a, start, chunk_size):
        a_arr = jnp.atleast_1d(a)
        D = cosmo.Dplus(a_arr).astype(dtype)[0]
        # x(a, q) = q + sum_n D^n * psi_n
        out = jax.lax.dynamic_slice_in_dim(q_flat, start, chunk_size, axis=0)
        D_pow = D
        for psi_n in psi_flat:
            psi_chunk = jax.lax.dynamic_slice_in_dim(psi_n, start, chunk_size, axis=0)
            out = out + D_pow * psi_chunk
            D_pow = D_pow * D
        return out

    @partial(jax.jit, static_argnames=("chunk_size",))
    def vel_at_chunk(a, start, chunk_size):
        a_arr = jnp.atleast_1d(a)
        D = cosmo.Dplus(a_arr).astype(dtype)[0]
        # d/d(superconft) = (dD/dt) * (a^2) = H a^3 * (dD/da); equivalently
        # dD/d(superconft) / D = growth_rate(a) * E(a) * a^2.
        dDdt_over_D = (cosmo.growth_rate(a_arr) * cosmo.E(a_arr) * a_arr ** 2).astype(dtype)[0]
        out = jnp.zeros((chunk_size, 3), dtype=dtype)
        D_pow = D  # D^1, multiplied to give D^n after n-1 iterations
        for n_idx, psi_n in enumerate(psi_flat):
            n = n_idx + 1
            psi_chunk = jax.lax.dynamic_slice_in_dim(psi_n, start, chunk_size, axis=0)
            # fac = n * D^(n-1) * dD/dsuperconft = n * D^n * (dD/dt)/D
            fac = n * D_pow * dDdt_over_D
            out = out + fac * psi_chunk
            D_pow = D_pow * D
        return out

    return pos_at_chunk, vel_at_chunk


def _make_radial_kernel(dj, n_order: int, n_newton_iters: int,
                          v_radial: bool = True, deformation_mode: str = "none"):
    """Build a per-chunk per-replica kernel that finds each particle's single
    lightcone crossing in one shot — no shell loop.

    Seed: at the geometric midpoint a_mid, the LPT distance d_mid is a good
    estimate of chi(a_cross). cosmo.chi_to_a(d_mid) inverts the comoving
    distance to give a Newton seed; n Newton iterations on
    f(a) = |x_LPT(a) - x_obs| - chi(a) refine to fp32 ulp. A residual check
    filters particles whose true crossing falls outside [a_far, a_near] (the
    Newton seed gets clipped to the boundary and the residual stays large).

    :param v_radial: if True (default), output the line-of-sight velocity
        v · n̂ where n̂ = (x - x_obs)/|x - x_obs| as a (chunk,) array. If False,
        output the full 3-D velocity vector as (chunk, 3). Radial-only saves
        8 B/row in the catalogue; for most lightcone-cosmology pipelines only
        the radial component matters (redshift-space distortions, kSZ, etc.).

    :param deformation_mode: per-row deformation outputs evaluated at the
        refined ``a``. ``"none"`` (default) adds nothing. ``"stream"`` appends
        ``(stream_density, tidal_evals)`` where ``stream_density = 1/|det T|``
        and ``tidal_evals`` are the ascending eigenvalues of the symmetric part
        of the deformation tensor ``T = I + sum_n D^n grad psi_n`` (cosmic-web
        classification). ``"full"`` appends ``(T, velocity_gradient)`` as
        ``(chunk, 3, 3)`` tensors. When not ``"none"`` the kernel takes a
        trailing ``grad_psi_flats`` tuple of ``(N^3, 3, 3)`` gradient grids.

    Returns ``kernel(start, chunk_size, r_offset, a_mid, a_far, a_near,
    observer, a_shells, residual_tol, q_flat, psi_flats[, grad_psi_flats])
    -> (x, v_out, a, crosses, shell_idx[, *deformation outputs])``,
    where ``v_out`` is (chunk,) if v_radial else (chunk, 3).
    """
    cosmo = dj.cosmo
    dtype = dj.dtype
    c_over_H0 = jnp.asarray(2997.92458, dtype=dtype)  # Mpc/h
    assert deformation_mode in ("none", "stream", "full"), \
        f"deformation_mode must be 'none'|'stream'|'full', got {deformation_mode!r}"

    # NOTE: q_flat and the psi_n_flat fields are passed as *kernel arguments*
    # (not closure-captured), so JIT lowering treats them as inputs rather
    # than inlined constants. At N=1024 this drops the 38 GB compile-time
    # duplication that previously OOM'd during the "captured constants"
    # warning. n_order is fixed at make time so the tuple length is static.

    @partial(jax.jit, static_argnames=("chunk_size",))
    def kernel(start, chunk_size, r_offset, a_mid, a_far, a_near,
               observer, a_shells, residual_tol,
               q_flat, psi_flats, grad_psi_flats=()):
        # Slice LPT data for this chunk
        q_chunk = jax.lax.dynamic_slice_in_dim(q_flat, start, chunk_size, axis=0)
        psi_chunks = tuple(
            jax.lax.dynamic_slice_in_dim(psi_n, start, chunk_size, axis=0)
            for psi_n in psi_flats
        )

        # 1. Midpoint LPT position and distance to observer
        D_mid = cosmo.Dplus(jnp.atleast_1d(a_mid)).astype(dtype)[0]
        x_mid = q_chunk + r_offset[None, :]
        D_pow = D_mid
        for psi_n in psi_chunks:
            x_mid = x_mid + D_pow * psi_n
            D_pow = D_pow * D_mid
        d_mid = jnp.linalg.norm(x_mid - observer[None, :], axis=-1)

        # 2. Newton seed via inverse chi (clipped to survey range)
        a = cosmo.chi_to_a(d_mid).astype(dtype)
        a = jnp.clip(a, a_far, a_near)

        # 3. Newton refine
        for _ in range(n_newton_iters):
            D = cosmo.Dplus(a).astype(dtype)
            dDda = cosmo.Dplusda(a).astype(dtype)
            E_a = cosmo.E(a).astype(dtype)
            chi_a = cosmo.chi(a).astype(dtype)
            x = q_chunk + r_offset[None, :]
            dxda = jnp.zeros_like(q_chunk)
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

        # 4. Final x, v at refined a
        D = cosmo.Dplus(a).astype(dtype)
        E_a = cosmo.E(a).astype(dtype)
        growth = cosmo.growth_rate(a).astype(dtype)
        dD_t_over_D = growth * E_a * a ** 2  # (dD/dsuperconft) / D
        x_final = q_chunk + r_offset[None, :]
        v_final = jnp.zeros_like(q_chunk)
        D_pow_n = D
        for n_idx, psi_n in enumerate(psi_chunks):
            n = n_idx + 1
            x_final = x_final + D_pow_n[:, None] * psi_n
            v_final = v_final + (n * D_pow_n * dD_t_over_D)[:, None] * psi_n
            D_pow_n = D_pow_n * D

        # 5. Residual filter: particles whose true crossing is outside
        # [a_far, a_near] get clipped to a boundary and fail this check.
        chi_a = cosmo.chi(a).astype(dtype)
        diff_obs = x_final - observer[None, :]
        d_final = jnp.linalg.norm(diff_obs, axis=-1)
        residual = jnp.abs(d_final - chi_a)
        # strict interior excludes particles whose crossing sits exactly at the
        # survey boundary (where Newton clip would silently produce a false hit)
        crosses = (residual < residual_tol) & (a > a_far) & (a < a_near)

        # 6. Shell index from refined a (mirrors the shell-loop semantics)
        # a_shells is ascending of length n_shells+1. searchsorted returns the
        # insertion index k+1 such that a_shells[k] <= a < a_shells[k+1].
        shell_idx = jnp.searchsorted(a_shells, a) - 1
        shell_idx = jnp.clip(shell_idx, 0, a_shells.shape[0] - 2).astype(jnp.int16)

        # 7. Optional projection onto line-of-sight n̂ = (x - x_obs)/|x - x_obs|.
        # Sign convention: v_radial > 0 = receding (redshift).
        if v_radial:
            d_safe = jnp.where(d_final > 1e-10, d_final, 1.0)
            v_out = jnp.sum(v_final * diff_obs, axis=-1) / d_safe
        else:
            v_out = v_final

        # 8. Optional deformation / tidal / velocity-gradient at the refined a.
        # Each row is a Lagrangian particle, so gathering ∂_j ψ_n is exact (no
        # spatial interpolation). T_ij = δ_ij + Σ_n D^n ∂_j ψ_n,i;
        # ∂v_i/∂q_j = Σ_n n D^n (dD_t/D) ∂_j ψ_n,i.
        if deformation_mode != "none":
            grad_chunks = tuple(
                jax.lax.dynamic_slice_in_dim(g, start, chunk_size, axis=0)
                for g in grad_psi_flats
            )
            eye = jnp.broadcast_to(jnp.eye(3, dtype=dtype),
                                   (chunk_size, 3, 3))
            T = eye
            vel_grad = jnp.zeros((chunk_size, 3, 3), dtype=dtype)
            D_pow_n = D
            for n_idx, g in enumerate(grad_chunks):
                n = n_idx + 1
                T = T + D_pow_n[:, None, None] * g
                vel_grad = vel_grad + (n * D_pow_n * dD_t_over_D)[:, None, None] * g
                D_pow_n = D_pow_n * D
            if deformation_mode == "stream":
                detT = jnp.linalg.det(T)
                detT_safe = jnp.where(jnp.abs(detT) > 1e-30, detT, 1.0)
                stream_density = 1.0 / jnp.abs(detT_safe)
                T_sym = 0.5 * (T + jnp.swapaxes(T, -1, -2))
                tidal_evals = jnp.linalg.eigvalsh(T_sym)  # ascending (chunk, 3)
                return (x_final, v_out, a, crosses, shell_idx,
                        stream_density, tidal_evals)
            # full
            return x_final, v_out, a, crosses, shell_idx, T, vel_grad

        return x_final, v_out, a, crosses, shell_idx

    return kernel


def _compute_grad_psi_mesh(dj, n_order: int):
    """Per-order displacement-gradient grids in mesh shape ``(res,..,res,3,3)``
    with ``grid[.., i, j] = d psi_n,i / d q_j`` (exact Fourier gradients)."""
    from ..core.kernels import gradient_kernel
    from einops import rearrange
    grad_kernels = [gradient_kernel(dj.k_vecs, d, order=0, with_jax=True)
                    for d in range(dj.dim)]
    out = []
    for n in range(1, n_order + 1):
        psi_mesh = dj.ensure_mesh_shape(dj._lpt.psi[f"psi_{n}"])
        dpsi = jnp.asarray([
            jax.vmap(jnp.fft.rfftn, in_axes=-1, out_axes=-1)(psi_mesh) * g[..., None]
            for g in grad_kernels])
        dpsi = rearrange(dpsi, "g ... d -> ... (d g)")
        dpsi = jax.vmap(jnp.fft.irfftn, in_axes=-1, out_axes=-1)(dpsi)
        # (..., i, j) = d_j psi_i  (component i, gradient direction j)
        dpsi = rearrange(dpsi, "... (d g) -> ... d g", d=dj.dim, g=dj.dim)
        out.append(dpsi.astype(dj.dtype))
    return out


def _compute_grad_psi_flats(dj, n_order: int):
    """Per-order displacement-gradient grids flattened to ``(N^3, 3, 3)``.

    Uses exact Fourier gradients (``order=0``), the same construction as
    ``DiscoDJ.evaluate_jacobian_from_psi`` but kept per-order so the lightcone
    kernel can weight each by ``D_n(a)`` at the refined crossing.
    """
    return tuple(g.reshape(-1, dj.dim, dj.dim)
                 for g in _compute_grad_psi_mesh(dj, n_order))


def _sheet_resampled_fields(dj, n_order: int, shift, need_grad: bool):
    """Lagrangian positions and displacement (+ gradient) fields for one
    phase-space-sheet sub-sample offset ``shift`` (in grid-cell units).

    Sub-particles sit at ``q + shift * cell``; their displacement (and its
    gradient) is the band-limited (Fourier) interpolation of the ``psi`` fields
    to that offset — i.e. a point *inside* the Lagrangian cube / Kuhn tetrahedra
    rather than only at the grid vertices. With ``n_resample^dim`` such offsets,
    each carrying ``1/n_resample^dim`` of the mass, the deposited field
    approaches the smooth fixed-mass-per-tetrahedron sheet density and the
    grid-vertex aliasing of a one-point-per-cell deposit disappears.

    Reuses ``core.scatter_and_gather.fourier_interpolate_field`` — the same
    sheet interpolation the PM force and ``compute_field_quantity_from_particles``
    use via ``n_resample``.
    """
    from ..core.scatter_and_gather import fourier_interpolate_field
    dim = dj.dim
    boxsize = float(dj.boxsize)
    cell = boxsize / dj.res
    q_flat = jnp.asarray(dj.q.reshape(-1, dim))
    if onp.allclose(shift, 0.0):
        psi_flats = tuple(dj._lpt.psi[f"psi_{n}"].reshape(-1, dim)
                          for n in range(1, n_order + 1))
        grad = _compute_grad_psi_flats(dj, n_order) if need_grad else ()
        return q_flat, psi_flats, grad
    shift = list(onp.asarray(shift, dtype=onp.float64))
    q_flat = q_flat + jnp.asarray(shift, dtype=dj.dtype) * cell
    psi_flats = tuple(
        fourier_interpolate_field(dim, dj._lpt.psi[f"psi_{n}"], shift, boxsize,
                                  dj.dtype_num).reshape(-1, dim)
        for n in range(1, n_order + 1))
    if need_grad:
        grad = tuple(
            fourier_interpolate_field(
                dim, g.reshape(g.shape[:dim] + (dim * dim,)), shift, boxsize,
                dj.dtype_num).reshape(-1, dim, dim)
            for g in _compute_grad_psi_mesh(dj, n_order))
    else:
        grad = ()
    return q_flat, psi_flats, grad


def _sheet_shift_vectors(dim: int, n_resample: int) -> onp.ndarray:
    """``(n_resample^dim, dim)`` sub-cell offsets in grid-cell units."""
    d = onp.linspace(0.0, 1.0, n_resample, endpoint=False)
    grid = onp.meshgrid(*((d,) * dim), indexing="ij")
    return onp.stack([g.reshape(-1) for g in grid], axis=-1)


# Extra-column schema emitted for each deformation_mode (name -> (shape_tail, dtype)).
def _deformation_columns(deformation_mode: str):
    if deformation_mode == "stream":
        return {"StreamDensity": ((), onp.float32),
                "TidalEigenvalues": ((3,), onp.float32)}
    if deformation_mode == "full":
        return {"DeformationTensor": ((3, 3), onp.float32),
                "VelocityGradient": ((3, 3), onp.float32)}
    return {}


def evaluate_lpt_lightcone_streaming_radial(
    dj, a_far: float, a_near: float = 1.0, n_shells: int = 64,
    observer=None, n_order: int | None = None,
    n_part_chunks: int = 1, n_newton_iters: int = 1,
    keep_particle_idx: bool = False,
    residual_tol: float = 1e-1,  # Mpc/h
    v_mode: str = "radial",
    deformation_mode: str = "none",
    verbose: bool = False,
):
    """Radial-sort variant: solve each (particle, replica) for its single
    lightcone crossing in one shot, then assign a shell index post-hoc.

    Eliminates the per-shell loop. Each (chunk, replica) pair becomes one
    jitted kernel call (~8 * R calls instead of n_shells * n_part_chunks).
    Total memory traffic drops O(n_shells)-fold for the dominant LPT-eval
    terms.

    :param residual_tol: max allowed |d_final - chi(a_cross)| in Mpc/h for a
        row to be emitted. Doubles as the filter for particles whose true
        crossing lies outside [a_far, a_near] (they end up clipped to a
        boundary with large residual). Default 0.1 Mpc/h is comfortably above
        fp32 ulp at typical chi but well below a shell width.
    :param v_mode: ``"radial"`` (default) emits the line-of-sight velocity
        v · n̂ as ``out["v_radial"]`` (M,). ``"full"`` emits the 3-D vector as
        ``out["v"]`` (M, 3). Radial-only saves 8 B/row.
    :param deformation_mode: ``"none"`` (default), ``"stream"`` (adds
        ``out["stream_density"]`` (M,) and ``out["tidal_evals"]`` (M, 3)), or
        ``"full"`` (adds ``out["deformation"]`` and ``out["velocity_gradient"]``,
        both (M, 3, 3)). Evaluated from LPT at each crossing's refined a.
    """
    assert dj.dim == 3, "Lightcones are 3-D only."
    assert dj._lpt is not None, "Compute LPT first via with_lpt() / compute_lpt()."
    assert n_newton_iters >= 1, "radial_sort requires n_newton_iters >= 1"
    assert n_part_chunks >= 1, "n_part_chunks must be >= 1"
    assert v_mode in ("radial", "full"), f"v_mode must be 'radial' or 'full', got {v_mode!r}"
    assert deformation_mode in ("none", "stream", "full"), \
        f"deformation_mode must be 'none'|'stream'|'full', got {deformation_mode!r}"
    v_is_radial = (v_mode == "radial")
    # in-memory output keys for the extra deformation arrays, in kernel order
    _DEF_KEYS = {"stream": ["stream_density", "tidal_evals"],
                 "full": ["deformation", "velocity_gradient"]}.get(deformation_mode, [])

    boxsize = float(dj.boxsize)
    if observer is None:
        observer_host = onp.array([0.5, 0.5, 0.5], dtype=onp.float64) * boxsize
    else:
        observer_host = onp.asarray(observer, dtype=onp.float64)
        assert observer_host.shape == (3,), "observer must be a (3,) vector."

    dtype = dj.dtype
    # geometric mean — close to the chi midpoint of a log-spaced shell grid
    a_mid = float(onp.sqrt(a_far * a_near))
    a_shells_host = onp.geomspace(a_far, a_near, n_shells + 1).astype(onp.float64)
    chi_shells_host = onp.asarray(jax.device_get(dj.cosmo.chi(jnp.asarray(a_shells_host))))
    chi_min = float(chi_shells_host[-1])
    chi_max = float(chi_shells_host[0])

    replicas = enumerate_replicas(boxsize, observer_host, chi_min, chi_max)
    R = replicas.shape[0]
    if R == 0:
        return _empty_streaming_result(keep_particle_idx, v_is_radial=v_is_radial)
    replica_offsets = jnp.asarray(replicas, dtype=dtype) * boxsize  # (R, 3)
    observer_jax = jnp.asarray(observer_host, dtype=dtype)
    a_shells_d = jnp.asarray(a_shells_host, dtype=dtype)
    a_far_d = jnp.asarray(a_far, dtype=dtype)
    a_near_d = jnp.asarray(a_near, dtype=dtype)
    a_mid_d = jnp.asarray(a_mid, dtype=dtype)
    residual_tol_d = jnp.asarray(residual_tol, dtype=dtype)

    n_order_eff = n_order if n_order is not None else dj._lpt.n_order
    N_part = dj.res ** dj.dim
    base = N_part // n_part_chunks
    chunk_starts = [c * base for c in range(n_part_chunks)] + [N_part]
    kernel = _make_radial_kernel(dj, n_order_eff, n_newton_iters,
                                 v_radial=v_is_radial,
                                 deformation_mode=deformation_mode)
    # Pass ψ and q as kernel args (not closure) so they aren't duplicated as
    # captured constants during JIT lowering. Materialise once here.
    q_flat_d = jnp.asarray(dj.q.reshape(-1, 3))
    psi_flats_d = tuple(dj._lpt.psi[f"psi_{n}"].reshape(-1, 3)
                        for n in range(1, n_order_eff + 1))
    grad_psi_flats_d = (_compute_grad_psi_flats(dj, n_order_eff)
                        if deformation_mode != "none" else ())

    buf_x, buf_v, buf_a, buf_rep, buf_shell, buf_part = [], [], [], [], [], []
    buf_def = {k: [] for k in _DEF_KEYS}
    total = 0

    def _collect_radial(x, v, a, mask, shell_idx, extras, lo, r_idx):
        nonlocal total
        mask_np = onp.asarray(mask)
        if not mask_np.any():
            return
        sel = onp.where(mask_np)[0]
        buf_x.append(onp.asarray(x)[sel, :].astype(onp.float32))
        # v is (chunk,) when radial, (chunk, 3) when full -- same slicing works
        v_np = onp.asarray(v)
        buf_v.append((v_np[sel, :] if v_np.ndim == 2 else v_np[sel]).astype(onp.float32))
        buf_a.append(onp.asarray(a)[sel].astype(onp.float32))
        buf_rep.append(onp.full(sel.size, r_idx, dtype=onp.int16))
        buf_shell.append(onp.asarray(shell_idx)[sel].astype(onp.int16))
        if keep_particle_idx:
            buf_part.append((sel + lo).astype(onp.int32))
        for k, arr in zip(_DEF_KEYS, extras):
            buf_def[k].append(onp.asarray(arr)[sel].astype(onp.float32))
        total += int(sel.size)

    for c in range(n_part_chunks):
        lo = chunk_starts[c]
        hi = chunk_starts[c + 1]
        chunk_size = hi - lo
        if chunk_size <= 0:
            continue
        start = jnp.asarray(lo, dtype=jnp.int32)
        chunk_before = total
        pending = None  # (x, v, a, mask, shell_idx, extras, r_idx) awaiting _collect
        for r_idx in range(R):
            r_off = replica_offsets[r_idx]
            res = kernel(
                start, chunk_size, r_off, a_mid_d, a_far_d, a_near_d,
                observer_jax, a_shells_d, residual_tol_d,
                q_flat_d, psi_flats_d, grad_psi_flats_d,
            )
            x, v, a, mask, shell_idx = res[:5]
            extras = res[5:]
            if pending is not None:
                _collect_radial(*pending[:6], lo=lo, r_idx=pending[6])
            pending = (x, v, a, mask, shell_idx, extras, r_idx)
        if pending is not None:
            _collect_radial(*pending[:6], lo=lo, r_idx=pending[6])
        if verbose:
            print(f"  chunk {c+1}/{n_part_chunks} (particles {lo}-{hi}): "
                  f"{total - chunk_before} new, {total} total")

    if total == 0:
        return _empty_streaming_result(keep_particle_idx, v_is_radial=v_is_radial)

    # Concatenate-and-discard: clear each buf list immediately after building
    # its concatenated array so per-chunk slices are released before the next
    # concat allocates its big buffer. Cuts transient RAM noticeably at large N.
    out = {}
    out["x"] = onp.concatenate(buf_x, axis=0);     buf_x.clear()
    # v_radial: (M,) under "v_radial";  full: (M, 3) under "v".
    v_key = "v_radial" if v_is_radial else "v"
    out[v_key] = onp.concatenate(buf_v, axis=0);   buf_v.clear()
    out["a_cross"] = onp.concatenate(buf_a, axis=0); buf_a.clear()
    out["replica_idx"] = onp.concatenate(buf_rep, axis=0); buf_rep.clear()
    out["shell_idx"] = onp.concatenate(buf_shell, axis=0); buf_shell.clear()
    if keep_particle_idx:
        out["particle_idx"] = onp.concatenate(buf_part, axis=0); buf_part.clear()
    for k in _DEF_KEYS:
        out[k] = onp.concatenate(buf_def[k], axis=0); buf_def[k].clear()
    return out


def evaluate_lpt_lightcone_to_hdf5_radial(
    dj, output_path: str,
    a_far: float, a_near: float = 1.0, n_shells: int = 64,
    observer=None, n_order: int | None = None,
    n_part_chunks: int = 1, n_newton_iters: int = 1,
    keep_particle_idx: bool = False,
    residual_tol: float = 1e-1,
    v_mode: str = "radial",
    deformation_mode: str = "none",
    n_resample: int = 1,
    write_catalogue: bool = True,
    compression="zstd",
    storage_chunk_rows: int = 1 << 20,
    map_spec=None,
    verbose: bool = False,
) -> dict:
    """Radial-sort lightcone with incremental HDF5 writes.

    Same algorithm as ``evaluate_lpt_lightcone_streaming_radial`` but per-chunk
    crossings are appended to resizable HDF5 datasets as soon as the chunk
    finishes. Peak RAM holds at most one chunk's worth of crossings, never the
    full catalogue. Enables N >> 512 lightcones on 128 GB hosts.

    HDF5 layout matches ``save_lightcone_as_hdf5`` (Gadget-like): a Header
    group with cosmology + box + observer attrs, and a PartType1 group with
    extendable Coordinates / (Velocities or RadialVelocity) / ScaleFactor /
    ParticleIDs / Masses / ReplicaIndex / ShellIndex datasets, plus
    LagrangianParticleIndex if ``keep_particle_idx=True``.

    :param output_path: HDF5 file to write.
    :param observer: ``(3,)`` for one observer (default box centre) or
        ``(n_obs, 3)`` to write many independent mock skies into one file. With
        more than one observer an ``ObserverIndex`` column is added and the
        Header gains ``Observers``/``NumObservers``; each observer gets its own
        periodic-replica set.
    :param v_mode: ``"radial"`` (default) writes a 1-D ``RadialVelocity``
        dataset (line-of-sight component, redshift sign convention). ``"full"``
        writes the 3-D ``Velocities`` dataset. Radial saves 8 B/row.
    :param n_resample: phase-space-sheet over-sampling factor per dimension.
        ``1`` (default) deposits one particle per Lagrangian cell at its grid
        vertex — which aliases as a grid pattern at low resolution. ``>1``
        spawns ``n_resample^dim`` sub-particles *inside* each cell by Fourier-
        interpolating ψ (reusing ``core.scatter_and_gather``), each carrying
        ``1/n_resample^dim`` of the mass, so the deposited density approaches
        the smooth fixed-mass-per-tetrahedron sheet field. Multiplies the row
        count (and runtime) by ``n_resample^dim``; the Header records
        ``NumResample``.
    :param write_catalogue: if ``False``, the per-particle ``PartType1`` rows
        are *not* written (datasets stay empty) — only the ``map_spec`` shell
        maps are accumulated to ``/Maps``. Lets you crank ``n_resample`` for
        smooth high-``nside`` maps without the disk / row cost of a billion-row
        catalogue.
    :param storage_chunk_rows: HDF5 dataset chunk size (rows per compressed
        block). 1<<20 (1M particles) balances compression ratio against
        partial-read efficiency.
    :param map_spec: optional :class:`discodj.lpt.lightcone_maps.MapSpec`. When
        given, the crossings are also binned into a ``(n_bins, npix)`` HEALPix
        shell-map stack *during* generation (no second pass), written to a
        ``/Maps`` group and returned under ``"maps"`` in the summary.
    :return: ``{"n_particles": int, "n_replicas": int}`` summary (plus
        ``"maps"`` if ``map_spec`` is given).
    """
    import h5py
    from ..core.io import LightconeHDF5Writer, lightcone_particle_mass

    assert dj.dim == 3, "Lightcones are 3-D only."
    assert dj._lpt is not None, "Compute LPT first via with_lpt() / compute_lpt()."
    assert n_newton_iters >= 1, "to_hdf5_radial requires n_newton_iters >= 1"
    assert n_part_chunks >= 1, "n_part_chunks must be >= 1"
    assert v_mode in ("radial", "full"), f"v_mode must be 'radial' or 'full', got {v_mode!r}"
    assert deformation_mode in ("none", "stream", "full"), \
        f"deformation_mode must be 'none'|'stream'|'full', got {deformation_mode!r}"
    assert n_resample >= 1, "n_resample must be >= 1"
    v_is_radial = (v_mode == "radial")
    # Phase-space-sheet over-sampling: n_resample^dim sub-particles per cell,
    # each 1/n_resample^dim of the mass, sampled *inside* the Lagrangian cube via
    # Fourier interpolation of psi -> smooth fixed-mass-per-tetrahedron deposit.
    shift_vecs = _sheet_shift_vectors(dj.dim, n_resample)
    n_sub = shift_vecs.shape[0]
    # extra HDF5 columns + in-kernel output keys for the deformation outputs
    def_columns = _deformation_columns(deformation_mode)
    def_keys = {"stream": ["StreamDensity", "TidalEigenvalues"],
                "full": ["DeformationTensor", "VelocityGradient"]}.get(deformation_mode, [])

    # Release the with_lpt JIT compile cache before we allocate lightcone
    # buffers. At N=512 this frees ~28 GB that XLA was holding for compute_core
    # lowering; the lightcone radial kernel re-jits cleanly afterwards. Worth
    # ~3.5x headroom; the to_hdf5 path is the memory-conscious entry point,
    # so we do this unconditionally here. If callers have other JIT'd state
    # they want to preserve, use evaluate_lpt_lightcone_streaming_radial.
    jax.clear_caches()

    boxsize = float(dj.boxsize)
    observers_host = _parse_observers(observer, boxsize)
    n_obs = observers_host.shape[0]
    multi_obs = n_obs > 1

    dtype = dj.dtype
    a_mid = float(onp.sqrt(a_far * a_near))
    a_shells_host = onp.geomspace(a_far, a_near, n_shells + 1).astype(onp.float64)
    chi_shells_host = onp.asarray(jax.device_get(dj.cosmo.chi(jnp.asarray(a_shells_host))))
    chi_min = float(chi_shells_host[-1])
    chi_max = float(chi_shells_host[0])

    # Per-observer periodic-box replica sets (host-side, cheap).
    replicas_list = [enumerate_replicas(boxsize, observers_host[i], chi_min, chi_max)
                     for i in range(n_obs)]
    R_list = [int(r.shape[0]) for r in replicas_list]

    n_order_eff = n_order if n_order is not None else dj._lpt.n_order
    N_part = dj.res ** dj.dim
    # With sheet over-sampling there are n_sub sub-particles per Lagrangian cell.
    n_part_per_replica = N_part * n_sub

    # Per-(sub-)particle mass for the Gadget-like header (mass conserved: the
    # box mass is split over N_part * n_sub sub-particles).
    particle_mass = lightcone_particle_mass(dj.cosmo.Omega_m, boxsize,
                                            n_part_per_replica)

    header_attrs = {
        "LightconeMode": 1,
        "Observer": observers_host[0],
        "BoxSize": boxsize,
        "Omega0": float(dj.cosmo.Omega_m),
        "OmegaLambda": float(dj.cosmo.Omega_de),
        "HubbleParam": float(dj.cosmo.h),
        "MassTable": [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0],
        "NumPart_PerReplica": n_part_per_replica,
        "NumResample": n_resample,
        "NumFilesPerSnapshot": 1,
        "Time": 1.0,  # not meaningful for a lightcone
    }
    if multi_obs:
        header_attrs["Observers"] = observers_host
        header_attrs["NumObservers"] = n_obs

    # Per-observer ObserverIndex column only when there is more than one sky.
    extra_columns = dict(def_columns)
    if multi_obs:
        extra_columns["ObserverIndex"] = ((), onp.int16)

    # Optional on-the-fly HEALPix shell-map accumulation. For multiple observers
    # the maps stack as (n_obs, n_bins, npix) for sample-covariance estimation.
    maps_accum = None
    if map_spec is not None:
        maps_accum = onp.zeros(((n_obs,) + map_spec.shape) if multi_obs
                               else map_spec.shape, dtype=onp.float64)

    def _write_maps(h5f):
        if map_spec is None:
            return
        grp = h5f.create_group("Maps")
        grp.attrs["nside"] = map_spec.nside
        grp.attrs["a_edges"] = map_spec.a_edges
        grp.attrs["weighted"] = int(map_spec.weighted)
        grp.create_dataset("ShellMaps", data=maps_accum.astype(onp.float32))

    a_shells_d = jnp.asarray(a_shells_host, dtype=dtype)
    a_far_d = jnp.asarray(a_far, dtype=dtype)
    a_near_d = jnp.asarray(a_near, dtype=dtype)
    a_mid_d = jnp.asarray(a_mid, dtype=dtype)
    residual_tol_d = jnp.asarray(residual_tol, dtype=dtype)

    with h5py.File(output_path, "w") as h5f:
        writer = LightconeHDF5Writer.open(
            h5f, header_attrs=header_attrs, particle_mass=particle_mass,
            v_is_radial=v_is_radial, keep_particle_idx=keep_particle_idx,
            extra_columns=extra_columns,
            compression=compression, storage_chunk_rows=storage_chunk_rows)

        if sum(R_list) == 0:
            writer.close()
            _write_maps(h5f)
            if verbose:
                print(f"Lightcone written to {output_path} (0 particles; no "
                      f"replicas intersect the active shell).", flush=True)
            out = {"n_particles": 0, "n_replicas": 0}
            if multi_obs:
                out["n_observers"] = n_obs
            if map_spec is not None:
                out["maps"] = maps_accum
            return out

        # Kernel is observer- and sheet-independent (q, psi passed as runtime
        # args), so build it once and reuse across observers and sub-samples.
        kernel = _make_radial_kernel(dj, n_order_eff, n_newton_iters,
                                     v_radial=v_is_radial,
                                     deformation_mode=deformation_mode)
        need_grad = deformation_mode != "none"

        base = N_part // n_part_chunks
        chunk_starts = [c * base for c in range(n_part_chunks)] + [N_part]

        # Outer loop over phase-space-sheet sub-samples (n_sub = n_resample^dim;
        # n_sub == 1 reproduces the one-point-per-cell deposit exactly). The
        # interpolated (q, psi) for a sheet are built once and reused across
        # observers; ψ and q go in as kernel args, not captured constants —
        # avoids the double-allocation during JIT lowering that OOM'd at N=1024.
        for s_idx in range(n_sub):
            q_flat_d, psi_flats_d, grad_psi_flats_d = _sheet_resampled_fields(
                dj, n_order_eff, shift_vecs[s_idx], need_grad)
            lpid_base = s_idx * N_part

            for obs_idx in range(n_obs):
                R = R_list[obs_idx]
                if R == 0:
                    continue
                observer_host = observers_host[obs_idx]
                observer_jax = jnp.asarray(observer_host, dtype=dtype)
                replica_offsets = jnp.asarray(replicas_list[obs_idx], dtype=dtype) * boxsize

                for c in range(n_part_chunks):
                    lo = chunk_starts[c]
                    hi = chunk_starts[c + 1]
                    chunk_size = hi - lo
                    if chunk_size <= 0:
                        continue
                    start = jnp.asarray(lo, dtype=jnp.int32)
                    chunk_before = writer.total

                    # Per-chunk staging across replicas (R typically 1-27): one
                    # contiguous HDF5 append per chunk keeps the per-chunk compress
                    # cost reasonable. Pending-1 ahead lets the next kernel dispatch
                    # overlap with the host gather, same trick as the streaming path.
                    stage_x, stage_v, stage_a = [], [], []
                    stage_rep, stage_shell, stage_lpid = [], [], []
                    stage_def = {k: [] for k in def_keys}
                    pending = None

                    def _drain_pending():
                        nonlocal pending
                        if pending is None:
                            return
                        xp, vp, ap, mp, sp, extras_p, rp = pending
                        mask_np = onp.asarray(mp)
                        if mask_np.any():
                            sel = onp.where(mask_np)[0]
                            stage_x.append(onp.asarray(xp)[sel, :].astype(onp.float32))
                            vp_np = onp.asarray(vp)
                            stage_v.append((vp_np[sel, :] if vp_np.ndim == 2 else vp_np[sel])
                                           .astype(onp.float32))
                            stage_a.append(onp.asarray(ap)[sel].astype(onp.float32))
                            stage_rep.append(onp.full(sel.size, rp, dtype=onp.int16))
                            stage_shell.append(onp.asarray(sp)[sel].astype(onp.int16))
                            if keep_particle_idx:
                                stage_lpid.append((sel + lo + lpid_base).astype(onp.int32))
                            for k, arr in zip(def_keys, extras_p):
                                stage_def[k].append(onp.asarray(arr)[sel].astype(onp.float32))
                        pending = None

                    for r_idx in range(R):
                        r_off = replica_offsets[r_idx]
                        res = kernel(
                            start, chunk_size, r_off, a_mid_d, a_far_d, a_near_d,
                            observer_jax, a_shells_d, residual_tol_d,
                            q_flat_d, psi_flats_d, grad_psi_flats_d,
                        )
                        x, v, a_arr, mask, shell_idx = res[:5]
                        extras = res[5:]
                        _drain_pending()
                        pending = (x, v, a_arr, mask, shell_idx, extras, r_idx)
                    _drain_pending()

                    # Concatenate this chunk's staging across replicas and flush.
                    if stage_x:
                        x_c = onp.concatenate(stage_x);     stage_x.clear()
                        v_c = onp.concatenate(stage_v);     stage_v.clear()
                        a_c = onp.concatenate(stage_a);     stage_a.clear()
                        rep_c = onp.concatenate(stage_rep); stage_rep.clear()
                        shell_c = onp.concatenate(stage_shell); stage_shell.clear()
                        lpid_c = onp.concatenate(stage_lpid) if stage_lpid else None
                        if stage_lpid:
                            stage_lpid.clear()
                        extra_c = {k: onp.concatenate(stage_def[k]) for k in def_keys}
                        for k in def_keys:
                            stage_def[k].clear()
                        if multi_obs:
                            extra_c["ObserverIndex"] = onp.full(
                                x_c.shape[0], obs_idx, dtype=onp.int16)
                        if write_catalogue:
                            writer.append(x=x_c, v=v_c, a=a_c, replica_idx=rep_c,
                                          shell_idx=shell_c, particle_idx=lpid_c,
                                          extra=(extra_c or None))
                        if map_spec is not None:
                            from .lightcone_maps import accumulate_shell_maps
                            mw = (onp.full(x_c.shape[0], particle_mass, dtype=onp.float32)
                                  if map_spec.weighted else None)
                            partial = onp.asarray(accumulate_shell_maps(
                                x_c, a_c, observer_host, map_spec, mass_weight=mw),
                                dtype=onp.float64)
                            if multi_obs:
                                maps_accum[obs_idx] += partial
                            else:
                                maps_accum += partial
                        del x_c, v_c, a_c, rep_c, shell_c, lpid_c

                    if verbose:
                        print(f"  sheet {s_idx+1}/{n_sub} observer {obs_idx+1}/"
                              f"{n_obs} chunk {c+1}/{n_part_chunks} "
                              f"(particles {lo}-{hi}): {writer.total - chunk_before}"
                              f" new, {writer.total} total", flush=True)

        writer.close()
        _write_maps(h5f)
        total = writer.total

    if verbose:
        print(f"Lightcone written to {output_path} ({total} particles).",
              flush=True)
    out = {"n_particles": total, "n_replicas": sum(R_list)}
    if multi_obs:
        out["n_observers"] = n_obs
    if map_spec is not None:
        out["maps"] = maps_accum
    return out


def evaluate_lpt_lightcone_streaming(dj, a_far: float, a_near: float = 1.0, n_shells: int = 64,
                                       observer=None, n_order: int | None = None,
                                       exact_growth: bool = False,
                                       n_part_chunks: int = 1,
                                       n_newton_iters: int = 0,
                                       keep_particle_idx: bool = False,
                                       verbose: bool = False):
    """Streaming variant of evaluate_lpt_lightcone for large problems.

    Two iteration modes, both fully chunked, both fully using JAX/XLA
    parallelism inside each jitted kernel call:

      * ``n_part_chunks == 1`` (shell-major, fast path): single Python loop over
        shells with a full-N sliding position/velocity window. Cheapest when
        the full ``(N, 3)`` x/v arrays fit comfortably.

      * ``n_part_chunks > 1`` (particle-major, big-N path): outer Python loop
        over particle chunks, inner loop over shells. Each chunk has its own
        ``(chunk, 3)`` sliding x/v window. LPT evaluation is chunked via
        ``jax.lax.dynamic_slice_in_dim`` into the stored ``psi_n`` fields, so
        the full ``(N, 3)`` x/v arrays are never materialised. Use this when
        N_part is so large that the full-N sliding window does not fit. Only
        the default (non-exact-growth) LPT path is supported in this mode.

    Output schema (packed):
      x (f32, M, 3) | v (f32, M, 3) | a_cross (f32, M) | replica_idx (i16, M)
      shell_idx (i16, M)  [+ particle_idx (i32, M) if keep_particle_idx=True]

    ``mask`` is omitted entirely (all returned rows are physical crossings).

    The host transfer per chunk breaks JAX tracing; this variant is *not*
    differentiable. Use the non-streaming evaluator for gradient workflows.

    :param n_part_chunks: number of sequential particle chunks. 1 (default) =
        shell-major fast path. >1 = particle-major path with chunked LPT eval.
    :param n_newton_iters: number of Newton iterations on f(a) = |x_LPT(a) - x_obs| - chi(a)
        applied after the secant seed. 0 (default) = secant only (~1e-5 a_cross error per
        shell). >= 1 = Newton refinement using the analytic LPT trajectory and dchi/da;
        each iter gets us roughly to ulp-precision. Final x_cross / v_cross are evaluated
        directly from LPT at the refined a (not linearly interpolated), so neighbouring
        particles' (x, v) are individually accurate. Requires n_part_chunks > 1
        (uses the chunked-eval path).
    :param keep_particle_idx: include the Lagrangian particle index per row.
        Adds 4 B/record; default False since most downstream use does not need it.
        Set to True for AHK-style phase-space sheet reconstruction.
    """
    assert dj.dim == 3, "Lightcones are 3-D only."
    assert dj._lpt is not None, "Compute LPT first via with_lpt() / compute_lpt()."

    boxsize = float(dj.boxsize)
    if observer is None:
        observer_host = onp.array([0.5, 0.5, 0.5], dtype=onp.float64) * boxsize
    else:
        observer_host = onp.asarray(observer, dtype=onp.float64)
        assert observer_host.shape == (3,), "observer must be a (3,) vector."

    dtype = dj.dtype
    a_shells_d = jnp.geomspace(a_far, a_near, n_shells + 1, dtype=dtype)
    chi_shells_d = dj.cosmo.chi(a_shells_d)
    a_shells = onp.asarray(jax.device_get(a_shells_d))
    chi_shells = onp.asarray(jax.device_get(chi_shells_d))
    chi_min = float(chi_shells[-1])
    chi_max = float(chi_shells[0])

    replicas = enumerate_replicas(boxsize, observer_host, chi_min, chi_max)
    R = replicas.shape[0]
    if R == 0:
        return _empty_streaming_result(keep_particle_idx)
    replica_offsets = jnp.asarray(replicas, dtype=dtype) * boxsize  # (R, 3)
    observer_jax = jnp.asarray(observer_host, dtype=dtype)

    assert n_part_chunks >= 1, "n_part_chunks must be >= 1"
    if n_newton_iters >= 1 and n_part_chunks == 1:
        raise ValueError(
            "n_newton_iters >= 1 requires n_part_chunks > 1 (uses the chunked-eval path). "
            "Set n_part_chunks to a value > 1 (e.g. 8 or 16)."
        )
    n_order_eff = n_order if n_order is not None else dj._lpt.n_order
    N_part = dj.res ** dj.dim

    buf_x, buf_v, buf_a, buf_rep, buf_shell, buf_part = [], [], [], [], [], []
    total = 0

    def _collect(x_c, v_c, a_c, mask, lo, k):
        nonlocal total
        mask_np = onp.asarray(mask)  # (R, chunk) bool
        if not mask_np.any():
            return
        r_idx, n_idx = onp.where(mask_np)
        buf_x.append(onp.asarray(x_c)[r_idx, n_idx, :].astype(onp.float32))
        buf_v.append(onp.asarray(v_c)[r_idx, n_idx, :].astype(onp.float32))
        buf_a.append(onp.asarray(a_c)[r_idx, n_idx].astype(onp.float32))
        buf_rep.append(r_idx.astype(onp.int16))
        buf_shell.append(onp.full(r_idx.size, k, dtype=onp.int16))
        if keep_particle_idx:
            buf_part.append((n_idx + lo).astype(onp.int32))
        total += int(r_idx.size)

    # ------------------------------------------------------------------
    # Mode A: shell-major (n_part_chunks == 1). Full sliding window.
    # ------------------------------------------------------------------
    if n_part_chunks == 1:
        @jax.jit
        def pos_at(a):
            return dj.ensure_flat_shape(
                dj._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=True,
                                              exact_growth=exact_growth)
            )

        @jax.jit
        def vel_at(a):
            return dj.ensure_flat_shape(
                dj._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=False,
                                              time_derivative=True, exact_growth=exact_growth)
            )

        a_prev_d = jnp.asarray(a_shells[0], dtype=dtype)
        x_prev = pos_at(a_prev_d)
        v_prev = vel_at(a_prev_d)

        for k in range(n_shells):
            a_k_d = jnp.asarray(a_shells[k], dtype=dtype)
            a_next_d = jnp.asarray(a_shells[k + 1], dtype=dtype)
            chi_k = jnp.asarray(chi_shells[k], dtype=dtype)
            chi_kp1 = jnp.asarray(chi_shells[k + 1], dtype=dtype)
            x_next = pos_at(a_next_d)
            v_next = vel_at(a_next_d)
            x_c, v_c, a_c, mask = _per_shell_kernel(
                x_prev, x_next, v_prev, v_next,
                a_k_d, a_next_d, chi_k, chi_kp1,
                replica_offsets, observer_jax,
            )
            before = total
            _collect(x_c, v_c, a_c, mask, lo=0, k=k)
            if verbose and (k % max(1, n_shells // 10) == 0 or k == n_shells - 1):
                print(f"  shell {k+1}/{n_shells}: {total - before} new, {total} total")
            x_prev, v_prev = x_next, v_next

    # ------------------------------------------------------------------
    # Mode B: particle-major (n_part_chunks > 1). Chunked LPT eval, per-chunk
    # sliding window. The full-N x/v arrays are never materialised.
    # ------------------------------------------------------------------
    else:
        if exact_growth:
            raise NotImplementedError(
                "Chunked LPT evaluation (n_part_chunks > 1) does not yet support exact_growth=True; "
                "use n_part_chunks=1 to fall back to full-N evaluation."
            )

        # Uniform chunk size (last chunk may be smaller, triggering one extra JIT compile).
        base = N_part // n_part_chunks
        chunk_starts = [c * base for c in range(n_part_chunks)] + [N_part]
        a_shells_d = [jnp.asarray(a_shells[k], dtype=dtype) for k in range(n_shells + 1)]
        chi_shells_d = [jnp.asarray(chi_shells[k], dtype=dtype) for k in range(n_shells + 1)]

        if n_newton_iters >= 1:
            # Newton path: kernel refines a per particle. Sliding window across
            # shells: x_kp1_lpt from shell k becomes x_k_lpt for shell k+1, so
            # the left-bracket LPT evaluation is only paid once per shell.
            # Async pipeline: kernel for shell k+1 is dispatched before _collect
            # syncs on shell k's outputs, hiding the host-side gather behind
            # the next kernel's compute.
            newton_kernel, init_x_at = _make_newton_kernel(dj, n_order_eff, n_newton_iters)
            for c in range(n_part_chunks):
                lo = chunk_starts[c]
                hi = chunk_starts[c + 1]
                chunk_size = hi - lo
                if chunk_size <= 0:
                    continue
                start = jnp.asarray(lo, dtype=jnp.int32)
                chunk_before = total
                x_prev_lpt = init_x_at(start, chunk_size, a_shells_d[0])
                pending = None  # (x_c, v_c, a_c, mask, k) awaiting _collect
                for k in range(n_shells):
                    (x_c, v_c, a_c, mask), x_next_lpt = newton_kernel(
                        start, chunk_size, x_prev_lpt,
                        a_shells_d[k], a_shells_d[k + 1],
                        chi_shells_d[k], chi_shells_d[k + 1],
                        replica_offsets, observer_jax,
                    )
                    if pending is not None:
                        _collect(*pending[:4], lo=lo, k=pending[4])
                    pending = (x_c, v_c, a_c, mask, k)
                    x_prev_lpt = x_next_lpt
                if pending is not None:
                    _collect(*pending[:4], lo=lo, k=pending[4])
                if verbose:
                    print(f"  chunk {c+1}/{n_part_chunks} (particles {lo}-{hi}): "
                          f"{total - chunk_before} new, {total} total")
        else:
            # Secant-only path: precompute per-chunk sliding x/v across shells.
            pos_at_chunk, vel_at_chunk = _make_chunked_lpt_eval(dj, n_order_eff)
            for c in range(n_part_chunks):
                lo = chunk_starts[c]
                hi = chunk_starts[c + 1]
                chunk_size = hi - lo
                if chunk_size <= 0:
                    continue
                start = jnp.asarray(lo, dtype=jnp.int32)
                x_prev = pos_at_chunk(a_shells_d[0], start, chunk_size)
                v_prev = vel_at_chunk(a_shells_d[0], start, chunk_size)
                chunk_before = total
                for k in range(n_shells):
                    x_next = pos_at_chunk(a_shells_d[k + 1], start, chunk_size)
                    v_next = vel_at_chunk(a_shells_d[k + 1], start, chunk_size)
                    x_c, v_c, a_c, mask = _per_shell_kernel(
                        x_prev, x_next, v_prev, v_next,
                        a_shells_d[k], a_shells_d[k + 1],
                        chi_shells_d[k], chi_shells_d[k + 1],
                        replica_offsets, observer_jax,
                    )
                    _collect(x_c, v_c, a_c, mask, lo=lo, k=k)
                    x_prev, v_prev = x_next, v_next
                if verbose:
                    print(f"  chunk {c+1}/{n_part_chunks} (particles {lo}-{hi}): "
                          f"{total - chunk_before} new, {total} total")

    if total == 0:
        return _empty_streaming_result(keep_particle_idx)

    # Concatenate-and-discard: clear each buf list immediately after building
    # its concatenated array so the per-chunk slices are released before the
    # next concat allocates a 13+ GB buffer. Cuts transient RAM by ~13 GB at
    # N=512.
    out = {}
    out["x"] = onp.concatenate(buf_x, axis=0);     buf_x.clear()
    out["v"] = onp.concatenate(buf_v, axis=0);     buf_v.clear()
    out["a_cross"] = onp.concatenate(buf_a, axis=0); buf_a.clear()
    out["replica_idx"] = onp.concatenate(buf_rep, axis=0); buf_rep.clear()
    out["shell_idx"] = onp.concatenate(buf_shell, axis=0); buf_shell.clear()
    if keep_particle_idx:
        out["particle_idx"] = onp.concatenate(buf_part, axis=0); buf_part.clear()
    return out


def _empty_streaming_result(keep_particle_idx: bool, v_is_radial: bool = False):
    out = {
        "x": onp.zeros((0, 3), dtype=onp.float32),
        "a_cross": onp.zeros((0,), dtype=onp.float32),
        "replica_idx": onp.zeros((0,), dtype=onp.int16),
        "shell_idx": onp.zeros((0,), dtype=onp.int16),
    }
    if v_is_radial:
        out["v_radial"] = onp.zeros((0,), dtype=onp.float32)
    else:
        out["v"] = onp.zeros((0, 3), dtype=onp.float32)
    if keep_particle_idx:
        out["particle_idx"] = onp.zeros((0,), dtype=onp.int32)
    return out
