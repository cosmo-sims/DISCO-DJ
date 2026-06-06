#!/usr/bin/env python
"""Small-grid lightcone diagnostics: isolate the source of large-scale
anisotropy (periodic replication vs sampling vs a geometry bug).

Renders a 2x2 panel of equatorial-slice wedges:
   rows: shallow (chi_max < L -> single box, R=1) vs deep (many replicas)
   cols: vertex sampling vs uniform-in-cell (random sub-cell jitter)
plus printed checks (replica count, crossing-count vs volume).

Fast on purpose (small res); run:  python docs/diag_lightcone.py
"""
from __future__ import annotations
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discodj import DiscoDJ
from discodj.core.healpix import vec2ang
from discodj.lpt.lightcone import enumerate_replicas

HERE = os.path.dirname(os.path.abspath(__file__))
RES = int(os.environ.get("DIAG_RES", "64"))
BOX = float(os.environ.get("DIAG_BOX", "1000.0"))

plt.rcParams.update({"figure.facecolor": "#0d1117", "savefig.facecolor": "#0d1117",
                     "axes.facecolor": "#05070a", "text.color": "#e6edf3",
                     "axes.labelcolor": "#9da7b3", "axes.titlecolor": "#e6edf3"})


def make_dj():
    dj = (DiscoDJ(dim=3, res=RES, boxsize=BOX, cosmo="Planck18EEBAOSN")
          .with_timetables().with_linear_ps().with_ics(seed=42).with_lpt(n_order=2))
    return dj


def wedge(dj, a_far, jitter, dec_hw=3.0, randomize=False):
    obs = np.array([BOX / 2] * 3)
    out = dj.evaluate_lpt_lightcone(a_far=a_far, a_near=1.0, n_shells=32,
                                    observer=obs, streaming=True, radial_sort=True,
                                    n_part_chunks=2, n_newton_iters=1, v_mode="radial",
                                    randomize_replicas=randomize, replica_seed=7)
    x = np.asarray(out["x"], dtype=np.float64)
    rel = x - obs[None, :]
    if jitter:
        h = BOX / RES
        rng = np.random.default_rng(0)
        rel = rel + rng.uniform(-h / 2, h / 2, size=rel.shape)
    d = np.linalg.norm(rel, axis=-1)
    th, ph = vec2ang(rel)
    dec = 90.0 - np.degrees(np.asarray(th))
    sl = np.abs(dec) < dec_hw
    chi_max = float(dj.cosmo.chi(jnp.asarray(a_far)))
    reps = enumerate_replicas(BOX, obs, float(dj.cosmo.chi(jnp.asarray(1.0))), chi_max)
    expected = (4 / 3) * np.pi * chi_max ** 3 / (BOX / RES) ** 3
    print(f"  a_far={a_far}: chi_max={chi_max:.0f}  R={reps.shape[0]}  "
          f"crossings={x.shape[0]:,}  (ball-volume estimate ~{expected:,.0f})")
    return rel[sl, 0], rel[sl, 1], chi_max


def main():
    dj = make_dj()
    # shallow: single box (chi_max < L/2 -> R=1); deep: many replicas
    a_shallow = float(dj.cosmo.chi_to_a(jnp.asarray(0.45 * BOX)))  # chi ~ 0.45 L
    a_deep = 0.4
    cases = [("deep — vertex", a_deep, False, False),
             ("deep — uniform-in-cell", a_deep, True, False),
             ("deep — randomized replicas", a_deep, False, True),
             ("deep — randomized + jitter", a_deep, True, True)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (label, a_far, jit, rnd) in zip(axes.ravel(), cases):
        print(f"{label}:")
        gx, gy, R = wedge(dj, a_far, jit, randomize=rnd)
        nb = 320
        H, _, _ = np.histogram2d(gx, gy, bins=nb, range=[[-R, R], [-R, R]],
                                 weights=1.0 / np.clip(np.hypot(gx, gy), 1.0, None))
        pos = H[H > 0]
        ax.imshow(np.log10(np.where(H > 0, H, pos.min() * .5)).T, origin="lower",
                  extent=[-R, R, -R, R], cmap="magma", interpolation="nearest",
                  vmin=np.percentile(np.log10(pos), 40),
                  vmax=np.percentile(np.log10(pos), 99))
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(HERE, "figures", "diag_wedges.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
