"""Interleaved past-lightcone from the PM / Tree-PM N-body trajectory.

The LPT lightcone uses the closed-form ``x(a) = q + sum_n D_n(a) psi_n``. The
N-body trajectory has no closed form, so here consecutive integrator steps are
treated as crossing brackets: as the host-driven stepping loop advances
(``DiscoDJ.run_nbody(step_callback=...)``), each new ``(X, P, a)`` snapshot is
paired with the previous one and the per-particle/per-replica lightcone crossing
``|x(a) - x_obs| = chi(a)`` is solved by linear-in-``a`` interpolation across the
bracket — reusing the exact same vmapped kernel as the LPT shell loop
(``_per_shell_kernel``). Crossings stream straight to HDF5, so only two
snapshots are ever held in memory (peak RAM ~ a normal N-body run), which is
what lets it scale to large N.

Output schema is identical to the LPT lightcone (Gadget-like, via
``LightconeHDF5Writer``), so all the sky-projection / map / export tooling
applies unchanged. Velocities are the canonical ``a^2 dx/dt`` that the writer
converts to the Gadget ``sqrt(a)`` convention, matching the LPT path.
"""

from __future__ import annotations

import numpy as onp
import jax
import jax.numpy as jnp

from ..lpt.lightcone import enumerate_replicas, _per_shell_kernel, _parse_observers
from ..core.io import LightconeHDF5Writer, lightcone_particle_mass

__all__ = ["run_nbody_lightcone"]


def run_nbody_lightcone(dj, output_path: str, a_ini: float, a_end: float,
                        n_steps: int, observer=None, *,
                        res_pm: int | None = None,
                        v_mode: str = "radial",
                        keep_particle_idx: bool = False,
                        compression="zstd",
                        storage_chunk_rows: int = 1 << 20,
                        verbose: bool = False,
                        **nbody_kwargs) -> dict:
    """Generate a past-lightcone from an N-body run, streaming to HDF5.

    :param dj: a ``DiscoDJ`` with LPT computed (used for the N-body ICs).
    :param a_ini, a_end: integration range (far -> near; ``a_end`` typically 1).
    :param n_steps: number of integrator steps; each step boundary is a bracket.
    :param observer: ``(3,)`` observer position (default box centre). A single
        observer is supported.
    :param res_pm: PM grid resolution (required for the ``"pm"`` force).
    :param v_mode: ``"radial"`` (default) or ``"full"`` velocity output.
    :param nbody_kwargs: forwarded to ``DiscoDJ.run_nbody`` (e.g. ``stepper``,
        ``method``, ``grad_kernel_order``).
    :return: ``{"n_particles": int, "n_replicas": int, "n_steps": int}``.
    """
    import h5py

    assert dj.dim == 3, "Lightcones are 3-D only."
    assert v_mode in ("radial", "full"), \
        f"v_mode must be 'radial' or 'full', got {v_mode!r}"
    v_is_radial = (v_mode == "radial")

    boxsize = float(dj.boxsize)
    observers = _parse_observers(observer, boxsize)
    assert observers.shape[0] == 1, \
        "run_nbody_lightcone currently supports a single observer."
    observer_host = observers[0]
    observer_jax = jnp.asarray(observer_host, dtype=dj.dtype)
    dtype = dj.dtype

    # Replica set spans the full simulation comoving-distance range.
    chi_a = lambda a: float(dj.cosmo.chi(jnp.asarray(a)))
    chi_far, chi_near = chi_a(a_ini), chi_a(a_end)
    chi_min, chi_max = min(chi_far, chi_near), max(chi_far, chi_near)
    replicas = enumerate_replicas(boxsize, observer_host, chi_min, chi_max)
    R = int(replicas.shape[0])
    replica_offsets = jnp.asarray(replicas, dtype=dtype) * boxsize

    N_part = dj.res ** dj.dim
    particle_mass = lightcone_particle_mass(dj.cosmo.Omega_m, boxsize, N_part)
    header_attrs = {
        "LightconeMode": 1,
        "Observer": observer_host,
        "BoxSize": boxsize,
        "Omega0": float(dj.cosmo.Omega_m),
        "OmegaLambda": float(dj.cosmo.Omega_de),
        "HubbleParam": float(dj.cosmo.h),
        "MassTable": [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0],
        "NumPart_PerReplica": N_part,
        "NumFilesPerSnapshot": 1,
        "Time": 1.0,
        "LightconeSource": "nbody",
    }

    with h5py.File(output_path, "w") as h5f:
        writer = LightconeHDF5Writer.open(
            h5f, header_attrs=header_attrs, particle_mass=particle_mass,
            v_is_radial=v_is_radial, keep_particle_idx=keep_particle_idx,
            compression=compression, storage_chunk_rows=storage_chunk_rows)

        prev = {"state": None}  # (X_prev, P_prev, a_prev)

        def _callback(k, a_k, X_flat, P_flat):
            if R == 0:
                return
            X_cur = jnp.asarray(X_flat, dtype=dtype)
            P_cur = jnp.asarray(P_flat, dtype=dtype)
            if prev["state"] is None:
                prev["state"] = (X_cur, P_cur, a_k)
                return
            X_prev, P_prev, a_prev = prev["state"]
            chi_prev = jnp.asarray(chi_a(a_prev), dtype=dtype)
            chi_cur = jnp.asarray(chi_a(a_k), dtype=dtype)
            x_c, v_c, a_c, mask = _per_shell_kernel(
                X_prev, X_cur, P_prev, P_cur,
                jnp.asarray(a_prev, dtype=dtype), jnp.asarray(a_k, dtype=dtype),
                chi_prev, chi_cur, replica_offsets, observer_jax)
            # shapes: (R, N, 3), (R, N, 3), (R, N), (R, N)
            mask_np = onp.asarray(mask)
            if mask_np.any():
                r_idx, n_idx = onp.where(mask_np)
                x_sel = onp.asarray(x_c)[r_idx, n_idx, :].astype(onp.float32)
                v_sel = onp.asarray(v_c)[r_idx, n_idx, :].astype(onp.float32)
                a_sel = onp.asarray(a_c)[r_idx, n_idx].astype(onp.float32)
                rep_sel = r_idx.astype(onp.int16)
                shell_sel = onp.full(r_idx.size, k - 1, dtype=onp.int16)
                if v_is_radial:
                    diff = x_sel - observer_host[None, :]
                    d = onp.linalg.norm(diff, axis=-1)
                    d_safe = onp.where(d > 1e-10, d, 1.0)
                    v_out = (v_sel * diff).sum(-1) / d_safe
                else:
                    v_out = v_sel
                lpid = n_idx.astype(onp.int32) if keep_particle_idx else None
                writer.append(x=x_sel, v=v_out.astype(onp.float32), a=a_sel,
                              replica_idx=rep_sel, shell_idx=shell_sel,
                              particle_idx=lpid)
            prev["state"] = (X_cur, P_cur, a_k)
            if verbose:
                print(f"  step {k}/{n_steps} (a={a_k:.4f}): "
                      f"{writer.total} crossings total", flush=True)

        dj.run_nbody(a_ini=a_ini, a_end=a_end, n_steps=n_steps,
                     res_pm=res_pm, step_callback=_callback,
                     return_displacement=True, **nbody_kwargs)

        writer.close()
        total = writer.total

    if verbose:
        print(f"N-body lightcone written to {output_path} "
              f"({total} crossings).", flush=True)
    return {"n_particles": total, "n_replicas": R, "n_steps": n_steps}
