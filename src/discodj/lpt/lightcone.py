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
                          v_radial: bool = True):
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

    Returns ``kernel(start, chunk_size, r_offset, a_mid, a_far, a_near,
    observer, a_shells, residual_tol) -> (x, v_out, a, crosses, shell_idx)``,
    where ``v_out`` is (chunk,) if v_radial else (chunk, 3).
    """
    cosmo = dj.cosmo
    dtype = dj.dtype
    c_over_H0 = jnp.asarray(2997.92458, dtype=dtype)  # Mpc/h

    # NOTE: q_flat and the psi_n_flat fields are passed as *kernel arguments*
    # (not closure-captured), so JIT lowering treats them as inputs rather
    # than inlined constants. At N=1024 this drops the 38 GB compile-time
    # duplication that previously OOM'd during the "captured constants"
    # warning. n_order is fixed at make time so the tuple length is static.

    @partial(jax.jit, static_argnames=("chunk_size",))
    def kernel(start, chunk_size, r_offset, a_mid, a_far, a_near,
               observer, a_shells, residual_tol,
               q_flat, psi_flats):
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

        return x_final, v_out, a, crosses, shell_idx

    return kernel


def evaluate_lpt_lightcone_streaming_radial(
    dj, a_far: float, a_near: float = 1.0, n_shells: int = 64,
    observer=None, n_order: int | None = None,
    n_part_chunks: int = 1, n_newton_iters: int = 1,
    keep_particle_idx: bool = False,
    residual_tol: float = 1e-1,  # Mpc/h
    v_mode: str = "radial",
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
    """
    assert dj.dim == 3, "Lightcones are 3-D only."
    assert dj._lpt is not None, "Compute LPT first via with_lpt() / compute_lpt()."
    assert n_newton_iters >= 1, "radial_sort requires n_newton_iters >= 1"
    assert n_part_chunks >= 1, "n_part_chunks must be >= 1"
    assert v_mode in ("radial", "full"), f"v_mode must be 'radial' or 'full', got {v_mode!r}"
    v_is_radial = (v_mode == "radial")

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
    kernel = _make_radial_kernel(dj, n_order_eff, n_newton_iters, v_radial=v_is_radial)
    # Pass ψ and q as kernel args (not closure) so they aren't duplicated as
    # captured constants during JIT lowering. Materialise once here.
    q_flat_d = jnp.asarray(dj.q.reshape(-1, 3))
    psi_flats_d = tuple(dj._lpt.psi[f"psi_{n}"].reshape(-1, 3)
                        for n in range(1, n_order_eff + 1))

    buf_x, buf_v, buf_a, buf_rep, buf_shell, buf_part = [], [], [], [], [], []
    total = 0

    def _collect_radial(x, v, a, mask, shell_idx, lo, r_idx):
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
        total += int(sel.size)

    for c in range(n_part_chunks):
        lo = chunk_starts[c]
        hi = chunk_starts[c + 1]
        chunk_size = hi - lo
        if chunk_size <= 0:
            continue
        start = jnp.asarray(lo, dtype=jnp.int32)
        chunk_before = total
        pending = None  # (x, v, a, mask, shell_idx, r_idx) awaiting _collect
        for r_idx in range(R):
            r_off = replica_offsets[r_idx]
            x, v, a, mask, shell_idx = kernel(
                start, chunk_size, r_off, a_mid_d, a_far_d, a_near_d,
                observer_jax, a_shells_d, residual_tol_d,
                q_flat_d, psi_flats_d,
            )
            if pending is not None:
                _collect_radial(*pending[:5], lo=lo, r_idx=pending[5])
            pending = (x, v, a, mask, shell_idx, r_idx)
        if pending is not None:
            _collect_radial(*pending[:5], lo=lo, r_idx=pending[5])
        if verbose:
            print(f"  chunk {c+1}/{n_part_chunks} (particles {lo}-{hi}): "
                  f"{total - chunk_before} new, {total} total")

    if total == 0:
        return _empty_streaming_result(keep_particle_idx)

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
    return out


def evaluate_lpt_lightcone_to_hdf5_radial(
    dj, output_path: str,
    a_far: float, a_near: float = 1.0, n_shells: int = 64,
    observer=None, n_order: int | None = None,
    n_part_chunks: int = 1, n_newton_iters: int = 1,
    keep_particle_idx: bool = False,
    residual_tol: float = 1e-1,
    v_mode: str = "radial",
    compression="zstd",
    storage_chunk_rows: int = 1 << 20,
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
    :param v_mode: ``"radial"`` (default) writes a 1-D ``RadialVelocity``
        dataset (line-of-sight component, redshift sign convention). ``"full"``
        writes the 3-D ``Velocities`` dataset. Radial saves 8 B/row.
    :param storage_chunk_rows: HDF5 dataset chunk size (rows per compressed
        block). 1<<20 (1M particles) balances compression ratio against
        partial-read efficiency.
    :return: ``{"n_particles": int, "n_replicas": int}`` summary.
    """
    import h5py
    from ..core.io import compression_kwargs as _compression_kwargs

    assert dj.dim == 3, "Lightcones are 3-D only."
    assert dj._lpt is not None, "Compute LPT first via with_lpt() / compute_lpt()."
    assert n_newton_iters >= 1, "to_hdf5_radial requires n_newton_iters >= 1"
    assert n_part_chunks >= 1, "n_part_chunks must be >= 1"
    assert v_mode in ("radial", "full"), f"v_mode must be 'radial' or 'full', got {v_mode!r}"
    v_is_radial = (v_mode == "radial")

    # Release the with_lpt JIT compile cache before we allocate lightcone
    # buffers. At N=512 this frees ~28 GB that XLA was holding for compute_core
    # lowering; the lightcone radial kernel re-jits cleanly afterwards. Worth
    # ~3.5x headroom; the to_hdf5 path is the memory-conscious entry point,
    # so we do this unconditionally here. If callers have other JIT'd state
    # they want to preserve, use evaluate_lpt_lightcone_streaming_radial.
    jax.clear_caches()

    boxsize = float(dj.boxsize)
    if observer is None:
        observer_host = onp.array([0.5, 0.5, 0.5], dtype=onp.float64) * boxsize
    else:
        observer_host = onp.asarray(observer, dtype=onp.float64)
        assert observer_host.shape == (3,), "observer must be a (3,) vector."

    dtype = dj.dtype
    a_mid = float(onp.sqrt(a_far * a_near))
    a_shells_host = onp.geomspace(a_far, a_near, n_shells + 1).astype(onp.float64)
    chi_shells_host = onp.asarray(jax.device_get(dj.cosmo.chi(jnp.asarray(a_shells_host))))
    chi_min = float(chi_shells_host[-1])
    chi_max = float(chi_shells_host[0])

    replicas = enumerate_replicas(boxsize, observer_host, chi_min, chi_max)
    R = replicas.shape[0]

    n_order_eff = n_order if n_order is not None else dj._lpt.n_order
    N_part = dj.res ** dj.dim
    n_part_per_replica = N_part

    # Particle mass for the Gadget-like header (one box's worth of mass).
    G = 43.007105731706317
    Hubble = 100.0
    particle_mass = float(
        dj.cosmo.Omega_m * 3.0 * Hubble * Hubble / (8.0 * onp.pi * G)
        * boxsize ** 3 / n_part_per_replica
    )

    ds_kwargs = _compression_kwargs(compression)

    with h5py.File(output_path, "w") as h5f:
        header = h5f.create_group("Header")
        header.attrs["LightconeMode"] = 1
        header.attrs["Observer"] = observer_host
        header.attrs["BoxSize"] = boxsize
        header.attrs["Omega0"] = float(dj.cosmo.Omega_m)
        header.attrs["OmegaLambda"] = float(dj.cosmo.Omega_de)
        header.attrs["HubbleParam"] = float(dj.cosmo.h)
        header.attrs["MassTable"] = [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0]
        header.attrs["NumPart_PerReplica"] = n_part_per_replica
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 1.0  # not meaningful for a lightcone

        prefix = "PartType1/"

        def _mkds(name, shape_tail, dt):
            shape = (0,) + shape_tail
            maxshape = (None,) + shape_tail
            chunks = (storage_chunk_rows,) + shape_tail
            return h5f.create_dataset(prefix + name, shape=shape, maxshape=maxshape,
                                       chunks=chunks, dtype=dt, **ds_kwargs)

        d_coords = _mkds("Coordinates", (3,), onp.float32)
        if v_is_radial:
            d_vel = _mkds("RadialVelocity", (), onp.float32)
        else:
            d_vel = _mkds("Velocities",     (3,), onp.float32)
        d_a      = _mkds("ScaleFactor", (),   onp.float32)
        # uint64 because lightcone catalogues at N >= 1024 exceed 2^32 rows.
        d_pid    = _mkds("ParticleIDs", (),   onp.uint64)
        d_mass   = _mkds("Masses",      (),   onp.float32)
        d_rep    = _mkds("ReplicaIndex",(),   onp.int16)
        d_shell  = _mkds("ShellIndex",  (),   onp.int16)
        d_lpid   = (_mkds("LagrangianParticleIndex", (), onp.int32)
                    if keep_particle_idx else None)

        if R == 0:
            header.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 0]
            header.attrs["NumPart_Total"] = [0, 0, 0, 0, 0, 0]
            header.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
            if verbose:
                print(f"Lightcone written to {output_path} (0 particles; no "
                      f"replicas intersect the active shell).", flush=True)
            return {"n_particles": 0, "n_replicas": 0}

        replica_offsets = jnp.asarray(replicas, dtype=dtype) * boxsize  # (R, 3)
        observer_jax = jnp.asarray(observer_host, dtype=dtype)
        a_shells_d = jnp.asarray(a_shells_host, dtype=dtype)
        a_far_d = jnp.asarray(a_far, dtype=dtype)
        a_near_d = jnp.asarray(a_near, dtype=dtype)
        a_mid_d = jnp.asarray(a_mid, dtype=dtype)
        residual_tol_d = jnp.asarray(residual_tol, dtype=dtype)
        kernel = _make_radial_kernel(dj, n_order_eff, n_newton_iters,
                                     v_radial=v_is_radial)
        # ψ and q as kernel args, not captured constants — avoids the
        # double-allocation during JIT lowering that OOM'd at N=1024.
        q_flat_d = jnp.asarray(dj.q.reshape(-1, 3))
        psi_flats_d = tuple(dj._lpt.psi[f"psi_{n}"].reshape(-1, 3)
                            for n in range(1, n_order_eff + 1))

        base = N_part // n_part_chunks
        chunk_starts = [c * base for c in range(n_part_chunks)] + [N_part]
        total = 0

        def _flush(x, v, a_arr, rep_idx, shell_idx, lpid):
            """Append a contiguous crossings block to the HDF5 datasets."""
            nonlocal total
            n_new = x.shape[0]
            if n_new == 0:
                return
            new_total = total + n_new
            # Gadget velocity convention: v_g = v * 100 / a^1.5 per-particle
            # (same scalar factor for radial or full vector components)
            a_safe = onp.where(a_arr > 0, a_arr, 1.0)
            if v_is_radial:
                v_gadget = v * (100.0 / a_safe ** 1.5)
                d_vel.resize((new_total,));     d_vel[total:new_total] = v_gadget
            else:
                v_gadget = v * (100.0 / a_safe[:, None] ** 1.5)
                d_vel.resize((new_total, 3));   d_vel[total:new_total, :] = v_gadget

            d_coords.resize((new_total, 3));  d_coords[total:new_total, :] = x
            d_a.resize((new_total,));         d_a[total:new_total] = a_arr
            d_rep.resize((new_total,));       d_rep[total:new_total] = rep_idx
            d_shell.resize((new_total,));     d_shell[total:new_total] = shell_idx
            d_pid.resize((new_total,))
            d_pid[total:new_total] = onp.arange(total, new_total, dtype=onp.uint64)
            d_mass.resize((new_total,))
            d_mass[total:new_total] = particle_mass
            if d_lpid is not None and lpid is not None:
                d_lpid.resize((new_total,))
                d_lpid[total:new_total] = lpid
            total = new_total

        for c in range(n_part_chunks):
            lo = chunk_starts[c]
            hi = chunk_starts[c + 1]
            chunk_size = hi - lo
            if chunk_size <= 0:
                continue
            start = jnp.asarray(lo, dtype=jnp.int32)
            chunk_before = total

            # Per-chunk staging across replicas (R typically 1-27): one
            # contiguous HDF5 append per chunk keeps the per-chunk compress
            # cost reasonable. Pending-1 ahead lets the next kernel dispatch
            # overlap with the host gather, same trick as the streaming path.
            stage_x, stage_v, stage_a = [], [], []
            stage_rep, stage_shell, stage_lpid = [], [], []
            pending = None

            def _drain_pending():
                nonlocal pending
                if pending is None:
                    return
                xp, vp, ap, mp, sp, rp = pending
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
                        stage_lpid.append((sel + lo).astype(onp.int32))
                pending = None

            for r_idx in range(R):
                r_off = replica_offsets[r_idx]
                x, v, a_arr, mask, shell_idx = kernel(
                    start, chunk_size, r_off, a_mid_d, a_far_d, a_near_d,
                    observer_jax, a_shells_d, residual_tol_d,
                    q_flat_d, psi_flats_d,
                )
                _drain_pending()
                pending = (x, v, a_arr, mask, shell_idx, r_idx)
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
                _flush(x_c, v_c, a_c, rep_c, shell_c, lpid_c)
                del x_c, v_c, a_c, rep_c, shell_c, lpid_c

            if verbose:
                print(f"  chunk {c+1}/{n_part_chunks} (particles {lo}-{hi}): "
                      f"{total - chunk_before} new, {total} total", flush=True)

        # Gadget convention: NumPart_Total / NumPart_ThisFile are 32-bit per
        # particle type; the upper 32 bits go to NumPart_Total_HighWord for
        # catalogues with > 2^32 rows. Required for N >= 1024.
        low = total & 0xFFFFFFFF
        high = total >> 32
        header.attrs["NumPart_ThisFile"] = [0, low, 0, 0, 0, 0]
        header.attrs["NumPart_Total"] = [0, low, 0, 0, 0, 0]
        header.attrs["NumPart_Total_HighWord"] = [0, high, 0, 0, 0, 0]

    if verbose:
        print(f"Lightcone written to {output_path} ({total} particles).",
              flush=True)
    return {"n_particles": total, "n_replicas": R}


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
