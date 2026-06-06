#!/usr/bin/env python
"""Generate the DiscoDJ lightcone visual gallery.

Runs one real 2-LPT past-lightcone and renders a set of figures that showcase
the breadth of products the pipeline yields — from *smooth fields* (projected
dark-matter density, weak-lensing convergence) to *point catalogues* (the cosmic
web on the lightcone, redshift-space distortions, n(z)) — plus an interactive
3-D fly-through. The figures are written to ``docs/figures/`` and assembled into
a self-contained ``docs/lightcone_gallery.html``.

Run from the repo root:  ``python docs/make_lightcone_gallery.py``
Requires the ``sky`` + plotting extras: ``matplotlib healpy plotly`` (and discodj).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import plotly.graph_objects as go

from discodj import DiscoDJ
from discodj.lpt.lightcone_maps import (MapSpec, shells_to_overdensity,
                                        density_shells_to_kappa)
from discodj.core.healpix import vec2ang

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- simulation / survey configuration (kept modest so this runs in minutes) ----
# Box chosen LARGER than the lightcone (L > 2*chi_max) so the whole past-light
# ball fits in a SINGLE box (no periodic replication). Replicating one periodic
# box tiles identical structures every L -> a cubic lattice of repeated
# superclusters (cubic anisotropy); and there is no artifact-free way to
# decorrelate replicas of a periodic box (any per-replica rotation/translation
# breaks the seamless face-matching -> overdense boundary seams). A box bigger
# than the survey is the clean fix.
RES = int(os.environ.get("GALLERY_RES", "256"))
BOXSIZE = float(os.environ.get("GALLERY_BOX", "4000.0"))   # Mpc/h
A_FAR, A_NEAR = 0.55, 1.0  # z = 0.82 -> z = 0;  chi(0.55) ~ 1977 < BOXSIZE/2
N_SHELLS = 32
NSIDE = int(os.environ.get("GALLERY_NSIDE", "256"))  # HEALPix map resolution
NSIDE_GAL = 128            # galaxy angular map (sparser tracer)
SMOOTH_DEG = 0.7           # map smoothing FWHM
Z_SOURCE = 1.0             # weak-lensing source plane at the far edge
C_KM_S = 299792.458
# Phase-space-sheet over-sampling: n_resample^3 sub-particles per Lagrangian
# cell (each 1/n^3 of the mass), sampled inside the cell by Fourier-interpolating
# psi -> smooth fixed-mass-per-tetrahedron deposit instead of grid-vertex aliasing.
N_RESAMPLE = int(os.environ.get("GALLERY_NRESAMPLE", "2"))
# Galaxies via basic *linear Lagrangian bias*: each base-sheet cell (one per
# Lagrangian grid point) hosts a galaxy with probability
#   p(q) = GAL_NBAR * (1 + GAL_BIAS * delta_L(q)),   clipped to [0, 1],
# where delta_L is the linear (initial) overdensity = -div(psi_1). The mean
# probability GAL_NBAR is the same for every cell; the linear density modulates
# it (GAL_BIAS = 0 -> unbiased, every cell equally likely). Galaxies are then
# advected to their Eulerian lightcone positions like any other particle.
GAL_NBAR = float(os.environ.get("GALLERY_GAL_NBAR", "0.04"))
GAL_BIAS = float(os.environ.get("GALLERY_GAL_BIAS", "1.5"))
GAL_SEED = 1234

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "savefig.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#9da7b3",
    "ytick.color": "#9da7b3", "axes.edgecolor": "#30363d",
    "font.size": 11,
})


def generate_lightcone(path):
    """Build the scene and write the lightcone (+ on-the-fly density maps)."""
    print(f"Building {RES}^3 scene (L={BOXSIZE} Mpc/h) ...", flush=True)
    dj = (DiscoDJ(dim=3, res=RES, boxsize=BOXSIZE, cosmo="Planck18EEBAOSN")
          .with_timetables().with_linear_ps().with_ics(seed=42)
          .with_lpt(n_order=2))
    observer = np.array([BOXSIZE / 2] * 3)
    spec = MapSpec(nside=NSIDE,
                   a_edges=np.geomspace(A_FAR, A_NEAR, N_SHELLS + 1),
                   weighted=True)
    print(f"Generating lightcone (+ shell maps), n_resample={N_RESAMPLE} "
          f"-> {N_RESAMPLE**3} sub-particles/cell, nside={NSIDE} ...", flush=True)
    summary = dj.evaluate_lpt_lightcone_to_hdf5(
        path, a_far=A_FAR, a_near=A_NEAR, n_shells=N_SHELLS, observer=observer,
        n_part_chunks=4, n_newton_iters=1, v_mode="radial",
        n_resample=N_RESAMPLE,
        keep_particle_idx=True,   # to map base-sheet crossings -> Lagrangian cell
        map_spec=spec, verbose=True)
    print(f"  -> {summary['n_particles']:,} crossings, "
          f"{summary['n_replicas']} replicas", flush=True)
    return dj, observer, spec, summary


def _save_mollview(out):
    fig = plt.gcf()
    fig.set_facecolor("#0d1117")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {out}", flush=True)


def fig_sky_density(spec, density_shells, out):
    """Mollweide of the projected DM density in a thin redshift slab.

    A full-depth projection averages many independent line-of-sight structures
    and washes out angular contrast, so we integrate a single slab (z ~ 0.2-0.5)
    and lightly smooth it — the coherent cosmic web on the sky.
    """
    a_mid = 0.5 * (spec.a_edges[1:] + spec.a_edges[:-1])
    z_mid = 1.0 / a_mid - 1.0
    slab = (z_mid >= 0.2) & (z_mid <= 0.5)
    m = np.asarray(density_shells)[slab].sum(axis=0)
    ratio = m / max(m[m > 0].mean(), 1e-30)            # 1 + delta
    sm = hp.smoothing(ratio, fwhm=np.radians(SMOOTH_DEG))
    plt.close("all")
    hp.mollview(np.log10(np.clip(sm, 1e-2, None)), cmap="magma", cbar=True,
                norm="hist",
                title="Projected dark-matter density (z ~ 0.2-0.5 slab)",
                unit=r"$\log_{10}(1+\delta)$", bgcolor="#0d1117")
    hp.graticule(color="#30363d", dpar=30, dmer=30)
    _save_mollview(out)


def fig_convergence(dj, spec, density_shells, out):
    """Mollweide of the Born weak-lensing convergence kappa (1.5 deg smoothing)."""
    delta = shells_to_overdensity(density_shells)
    kappa = np.asarray(density_shells_to_kappa(delta, spec.a_edges, dj.cosmo,
                                               z_source=Z_SOURCE))
    kappa = hp.smoothing(kappa, fwhm=np.radians(SMOOTH_DEG))
    lim = np.percentile(np.abs(kappa), 99)
    plt.close("all")
    hp.mollview(kappa, cmap="RdBu_r", cbar=True, min=-lim, max=lim,
                title=fr"Weak-lensing convergence $\kappa$  ($z_s={Z_SOURCE}$)",
                unit=r"$\kappa$", bgcolor="#0d1117")
    hp.graticule(color="#30363d", dpar=30, dmer=30)
    _save_mollview(out)


def linear_overdensity_flat(dj):
    """Linear (initial) overdensity field delta_L = -div(psi_1), flattened to
    the Lagrangian-cell order (length res^dim). psi_1 is the Zel'dovich (D=1)
    displacement, so this is the linear density extrapolated to a = 1."""
    psi1 = np.asarray(dj._lpt.psi["psi_1"]).reshape(dj.res, dj.res, dj.res, 3)
    res, L = dj.res, float(dj.boxsize)
    kx = 2.0 * np.pi * np.fft.fftfreq(res, d=L / res)
    kr = 2.0 * np.pi * np.fft.rfftfreq(res, d=L / res)
    KX, KY, KZ = np.meshgrid(kx, kx, kr, indexing="ij")
    div_hat = (1j * KX * np.fft.rfftn(psi1[..., 0])
               + 1j * KY * np.fft.rfftn(psi1[..., 1])
               + 1j * KZ * np.fft.rfftn(psi1[..., 2]))
    delta = -np.fft.irfftn(div_hat, s=(res, res, res), axes=(0, 1, 2))
    return delta.reshape(-1).astype(np.float32)


def galaxy_cell_mask(delta_L):
    """Bernoulli galaxy occupation per Lagrangian cell under linear Lagrangian
    bias: p = clip(GAL_NBAR * (1 + GAL_BIAS * delta_L), 0, 1). Deterministic
    (fixed seed) so a cell's occupation is consistent across its replicas."""
    p = np.clip(GAL_NBAR * (1.0 + GAL_BIAS * delta_L), 0.0, 1.0)
    rng = np.random.default_rng(GAL_SEED)
    return rng.random(delta_L.shape[0]) < p


def read_catalogue(path, observer, dj, gal_cell, dec_halfwidth=2.0, n_3d=40000):
    """Stream the catalogue once, gathering: the thin-Dec wedge slice (all mass
    + galaxies), the n(z) histograms, a 3-D subsample, and a galaxy HEALPix
    count map. Galaxies are the base-sheet cells flagged by ``gal_cell`` (one
    per Lagrangian cell, linear-Lagrangian-bias Bernoulli occupation), advected
    to their Eulerian lightcone positions. Returns a dict of arrays."""
    import h5py
    from discodj.core.healpix import ang2pix_ring, nside2npix
    obs = np.asarray(observer)
    a_obs_of_z = lambda z: 1.0 / (1.0 + z)
    n_part_base = gal_cell.shape[0]
    hcell = float(dj.boxsize) / dj.res          # comoving cell size, for sub-cell jitter
    rng_jit = np.random.default_rng(2025)

    wedge = {k: [] for k in ("ra", "chi_real", "chi_rsd")}
    gwedge = {k: [] for k in ("ra", "chi_real")}     # galaxies in the slice
    z_real_all, z_rsd_all, zg_all = [], [], []
    sub = {k: [] for k in ("x", "y", "z", "redshift")}
    gal_map = np.zeros(nside2npix(NSIDE_GAL), dtype=np.float64)
    n_tot = n_gal = n_base = 0
    batch = 1 << 22
    with h5py.File(path, "r") as f:
        g = f["PartType1"]
        M = g["Coordinates"].shape[0]
        stride = max(M // n_3d, 1)
        has_gal = "LagrangianParticleIndex" in g
        for s in range(0, M, batch):
            e = min(s + batch, M)
            x = np.asarray(g["Coordinates"][s:e], dtype=np.float64)
            vr = np.asarray(g["RadialVelocity"][s:e], dtype=np.float64)
            a = np.asarray(g["ScaleFactor"][s:e], dtype=np.float64)
            rel = x - obs[None, :]
            d = np.linalg.norm(rel, axis=-1)
            theta, phi = vec2ang(rel)
            dec = 90.0 - np.degrees(np.asarray(theta))
            ra = np.degrees(np.asarray(phi))
            z_cosmo = 1.0 / a - 1.0
            v_pec = vr / np.sqrt(a)                  # km/s peculiar
            z_obs = (1.0 + z_cosmo) * (1.0 + v_pec / C_KM_S) - 1.0
            z_real_all.append(z_cosmo)
            z_rsd_all.append(z_obs)
            n_tot += e - s
            # galaxies = base-sheet cells (lpid < n_part_base) flagged by the
            # linear-Lagrangian-bias occupation mask -> one tracer per Lagrangian
            # cell, advected to its Eulerian lightcone position.
            if has_gal:
                lpid = np.asarray(g["LagrangianParticleIndex"][s:e])
                base_mask = lpid < n_part_base
                n_base += int(base_mask.sum())
                is_gal = base_mask & gal_cell[np.clip(lpid, 0, n_part_base - 1)]
            else:
                is_gal = np.zeros(e - s, dtype=bool)
            n_gal += int(is_gal.sum())
            if is_gal.any():
                # Sample each galaxy UNIFORMLY within its cell rather than at the
                # base vertex: a perfect lattice sampled at vertices is NOT
                # homogeneous on the lightcone (radial sphere-shell rings + cubic
                # axis spokes); a uniform sub-cell offset removes those geometric
                # artifacts (independent per replica image).
                relg = rel[is_gal] + rng_jit.uniform(-hcell / 2, hcell / 2,
                                                     size=(int(is_gal.sum()), 3))
                dg = np.linalg.norm(relg, axis=-1)
                thg, phg = vec2ang(relg)
                decg = 90.0 - np.degrees(np.asarray(thg))
                rag = np.degrees(np.asarray(phg))
                gp = ang2pix_ring(NSIDE_GAL, thg, phg)
                np.add.at(gal_map, np.asarray(gp), 1.0)
                zg_all.append(z_cosmo[is_gal])
                slg = np.abs(decg) < dec_halfwidth
                if slg.any():
                    gwedge["ra"].append(np.radians(rag[slg]))
                    gwedge["chi_real"].append(dg[slg])
            # thin equatorial slice for the wedge (all mass)
            sl = np.abs(dec) < dec_halfwidth
            if sl.any():
                a_obs = a_obs_of_z(z_obs[sl])
                chi_rsd = np.asarray(dj.cosmo.chi(jnp.asarray(a_obs)))
                wedge["ra"].append(np.radians(ra[sl]))
                wedge["chi_real"].append(d[sl])
                wedge["chi_rsd"].append(chi_rsd)
            # 3-D subsample (strided)
            idx = np.arange(0, e - s, stride)
            if idx.size:
                sub["x"].append(rel[idx, 0]); sub["y"].append(rel[idx, 1])
                sub["z"].append(rel[idx, 2]); sub["redshift"].append(z_cosmo[idx])
    cat = {k: (np.concatenate(v) if v else np.zeros(0))
           for d_ in (wedge, sub) for k, v in d_.items()}
    cat["gal_ra"] = np.concatenate(gwedge["ra"]) if gwedge["ra"] else np.zeros(0)
    cat["gal_chi"] = np.concatenate(gwedge["chi_real"]) if gwedge["chi_real"] else np.zeros(0)
    cat["z_real"] = np.concatenate(z_real_all)
    cat["z_rsd"] = np.concatenate(z_rsd_all)
    cat["z_gal"] = np.concatenate(zg_all) if zg_all else np.zeros(0)
    cat["gal_map"] = gal_map
    cat["gal_frac"] = n_gal / max(n_base, 1)   # fraction of base-sheet cells
    cat["n_gal"] = n_gal
    return cat


def fig_cosmic_web(cat, out):
    """Full-disk view of the equatorial slice as a 2-D density image — the
    classic 'slice through the universe'. A binned image avoids the overplot
    saturation of a scatter and shows filaments, knots and voids crisply."""
    ra, chi = cat["ra"], cat["chi_real"]
    x = chi * np.cos(ra)
    y = chi * np.sin(ra)
    R = chi.max() * 1.02
    nb = 460
    # The 4-deg slab is physically thicker at larger radius (count ~ chi), so
    # weight by 1/chi to flatten the radial gradient and show uniform contrast.
    w = 1.0 / np.clip(chi, 1.0, None)
    H, _, _ = np.histogram2d(x, y, bins=nb, range=[[-R, R], [-R, R]], weights=w)
    H = H.T
    pos = H[H > 0]
    floor = pos.min()
    img = np.log10(np.where(H > 0, H, floor * 0.5))
    vmin = np.percentile(np.log10(pos), 35)
    vmax = np.percentile(np.log10(pos), 99.5)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.set_facecolor("#05070a")
    ax.imshow(img, origin="lower", extent=[-R, R, -R, R], cmap="magma",
              interpolation="bilinear", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("comoving x [Mpc/h]")
    ax.set_ylabel("comoving y [Mpc/h]")
    ax.set_title("A 4-degree-thick slice through the lightcone — the cosmic web\n"
                 "traced by 2LPT particles (observer at centre)", pad=14)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {out}", flush=True)


def fig_rsd(cat, out):
    """Real-space vs redshift-space sector as smooth 2-D density images."""
    ra = cat["ra"]
    sel = (ra > np.radians(20)) & (ra < np.radians(110))
    sel &= (cat["chi_real"] > 400) & (cat["chi_real"] < 1500)
    th = ra[sel]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6),
                             subplot_kw={"projection": "polar"})
    # shared colour scale across the two panels
    nb_r, nb_t = 260, 260
    t_edges = np.linspace(np.radians(20), np.radians(110), nb_t)
    r_edges = np.linspace(400, 1500, nb_r)
    imgs = {}
    for key in ("chi_real", "chi_rsd"):
        H, _, _ = np.histogram2d(th, cat[key][sel], bins=[t_edges, r_edges])
        imgs[key] = np.log10(H + 1.0)
    vmax = max(np.percentile(im[im > 0], 99.5) for im in imgs.values())
    T, Rr = np.meshgrid(t_edges, r_edges, indexing="ij")
    for ax, key, ttl in ((axes[0], "chi_real", "Real space"),
                         (axes[1], "chi_rsd", "Redshift space (with RSD)")):
        ax.set_facecolor("#05070a")
        ax.pcolormesh(T, Rr, imgs[key], cmap="magma", vmin=0, vmax=vmax,
                      shading="auto")
        ax.set_thetamin(20); ax.set_thetamax(110)
        ax.set_ylim(400, 1500); ax.set_yticklabels([])
        ax.set_title(ttl, pad=18)
        ax.grid(color="#30363d", alpha=0.25)
    fig.suptitle("Redshift-space distortions: peculiar velocities stretch "
                 "structures along the line of sight", y=1.02)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {out}", flush=True)


def fig_galaxy_sky(cat, out):
    """Angular clustering of the sheet-selected galaxies (smoothed HEALPix)."""
    gmap = hp.smoothing(cat["gal_map"], fwhm=np.radians(1.0))
    plt.close("all")
    hp.mollview(gmap, cmap="cividis", cbar=True, norm="hist",
                title=f"Galaxy angular density (linear Lagrangian bias b={GAL_BIAS:g})",
                unit="galaxies / pixel (smoothed)", bgcolor="#0d1117")
    hp.graticule(color="#30363d", dpar=30, dmer=30)
    _save_mollview(out)


def fig_galaxies_web(cat, out):
    """Galaxies (cyan) over the dark-matter density slice (Cartesian, no polar
    ring artifacts) — they trace the knots and filaments, biased."""
    ra, chi = cat["ra"], cat["chi_real"]
    x = chi * np.cos(ra); y = chi * np.sin(ra)
    R = chi.max() * 1.02
    nb = 460
    w = 1.0 / np.clip(chi, 1.0, None)
    H, _, _ = np.histogram2d(x, y, bins=nb, range=[[-R, R], [-R, R]], weights=w)
    H = H.T
    pos = H[H > 0]
    img = np.log10(np.where(H > 0, H, pos.min() * 0.5))
    gx = cat["gal_chi"] * np.cos(cat["gal_ra"])
    gy = cat["gal_chi"] * np.sin(cat["gal_ra"])
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.set_facecolor("#05070a")
    ax.imshow(img, origin="lower", extent=[-R, R, -R, R], cmap="bone",
              interpolation="bilinear",
              vmin=np.percentile(np.log10(pos), 45),
              vmax=np.percentile(np.log10(pos), 99))
    ax.scatter(gx, gy, s=0.3, c="#39d0d8", alpha=0.5, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("comoving x [Mpc/h]"); ax.set_ylabel("comoving y [Mpc/h]")
    ax.set_title("Galaxies (cyan) tracing the cosmic web over the smooth "
                 "dark-matter density\n"
                 f"(linear Lagrangian bias b={GAL_BIAS:g}, "
                 f"{cat['n_gal']:,} galaxies)", pad=14)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {out}", flush=True)


def fig_nz(cat, out):
    """Redshift distribution, cosmological vs observed (RSD) + galaxies."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0, 1.0, 60)
    ax.hist(cat["z_real"], bins=bins, histtype="step", color="#58a6ff",
            lw=2, label="all mass (cosmological z)")
    ax.hist(cat["z_rsd"], bins=bins, histtype="step", color="#f0883e",
            lw=2, label="all mass (observed z, RSD)")
    if cat["z_gal"].size:
        scale = cat["z_real"].size / max(cat["z_gal"].size, 1)
        ax.hist(cat["z_gal"], bins=bins, histtype="step", color="#39d0d8",
                lw=2, weights=np.full(cat["z_gal"].size, scale),
                label="galaxies (linear bias, rescaled)")
    ax.set_xlabel("redshift $z$"); ax.set_ylabel("crossings / bin")
    ax.set_title("Lightcone redshift distribution $n(z)$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#30363d", alpha=0.3)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {out}", flush=True)


def interactive_3d_html(cat):
    """A small interactive 3-D point cloud of the lightcone, coloured by z."""
    fig = go.Figure(go.Scatter3d(
        x=cat["x"], y=cat["y"], z=cat["z"], mode="markers",
        marker=dict(size=1.4, color=cat["redshift"], colorscale="Magma",
                    opacity=0.7, colorbar=dict(title="z")),
        hoverinfo="skip"))
    fig.update_layout(
        template="plotly_dark", height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Interactive lightcone — drag to rotate, scroll to zoom",
        scene=dict(xaxis_title="x [Mpc/h]", yaxis_title="y [Mpc/h]",
                   zaxis_title="z [Mpc/h]", aspectmode="data"),
        paper_bgcolor="#0d1117", font=dict(color="#e6edf3"))
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       div_id="lc3d")


def build_html(figs, plotly_div, summary, out):
    cards = []
    for title, desc, png in figs:
        rel = "figures/" + os.path.basename(png)   # PNGs live beside this HTML
        cards.append(f"""
      <figure class="card">
        <img src="{rel}" alt="{title}" loading="lazy"/>
        <figcaption><h3>{title}</h3><p>{desc}</p></figcaption>
      </figure>""")
    n = summary["n_particles"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DiscoDJ lightcone gallery</title>
<style>
  :root {{ --bg:#0d1117; --fg:#e6edf3; --mut:#9da7b3; --card:#161b22; --line:#30363d; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:54px 24px 26px; text-align:center; max-width:980px; margin:0 auto; }}
  header h1 {{ font-size:2.1rem; margin:0 0 .4em; letter-spacing:-.02em; }}
  header p {{ color:var(--mut); max-width:760px; margin:.4em auto; }}
  .stat {{ display:inline-block; margin:14px 10px 0; padding:6px 14px;
           background:var(--card); border:1px solid var(--line); border-radius:999px;
           font-size:.85rem; color:var(--mut); }}
  .stat b {{ color:var(--fg); }}
  main {{ max-width:1180px; margin:0 auto; padding:18px 18px 80px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:22px; }}
  .card {{ margin:0; background:var(--card); border:1px solid var(--line);
           border-radius:14px; overflow:hidden; }}
  .card img {{ width:100%; display:block; background:#05070a; }}
  figcaption {{ padding:14px 18px 18px; }}
  figcaption h3 {{ margin:.1em 0 .3em; font-size:1.05rem; }}
  figcaption p {{ margin:0; color:var(--mut); font-size:.92rem; }}
  .wide {{ grid-column:1/-1; }}
  .interactive {{ margin-top:34px; background:var(--card); border:1px solid var(--line);
                  border-radius:14px; padding:8px; }}
  h2.section {{ margin:48px 0 6px; font-size:1.35rem; }}
  a {{ color:#58a6ff; }}
  footer {{ text-align:center; color:var(--mut); padding:30px; font-size:.85rem; }}
</style></head>
<body>
<header>
  <h1>DiscoDJ past-lightcones — from smooth fields to catalogues</h1>
  <p>Every panel below comes from a single differentiable 2-LPT past-lightcone
     ({RES}³ particles, {BOXSIZE:.0f} Mpc/h box, Planck18), generated and rendered by
     <code>docs/make_lightcone_gallery.py</code>. Mass is deposited with
     phase-space-sheet over-sampling ({N_RESAMPLE}³ sub-particles per cell) for a
     smooth, alias-free field. The same pipeline scales to 1024³ and refreshes
     across cosmologies for Fisher / inference work.</p>
  <div>
    <span class="stat"><b>{n:,}</b> crossings</span>
    <span class="stat"><b>{N_RESAMPLE}³</b> sub-particles/cell</span>
    <span class="stat"><b>{summary['n_replicas']}</b> box replicas</span>
    <span class="stat">z = <b>{1/A_FAR-1:.0f}</b> &rarr; <b>0</b></span>
    <span class="stat">pure-JAX, differentiable</span>
  </div>
</header>
<main>
  <h2 class="section">Smooth fields</h2>
  <div class="grid">
    {cards[0]}
    {cards[1]}
  </div>
  <h2 class="section">The cosmic web</h2>
  <div class="grid">
    <div class="wide">{cards[2]}</div>
  </div>
  <h2 class="section">Galaxies (linear Lagrangian bias)</h2>
  <div class="grid">
    <div class="wide">{cards[3]}</div>
    {cards[4]}
  </div>
  <h2 class="section">Redshift-space &amp; n(z)</h2>
  <div class="grid">
    <div class="wide">{cards[5]}</div>
    {cards[6]}
  </div>
  <h2 class="section">Interactive</h2>
  <div class="interactive">{plotly_div}</div>
</main>
<footer>Generated by DiscoDJ · see
  <a href="lightcone_integration.md">lightcone_integration.md</a></footer>
</body></html>"""
    with open(out, "w") as fh:
        fh.write(html)
    print(f"  wrote {out}", flush=True)


def main():
    import json
    tmp = os.path.join(tempfile.gettempdir(), "discodj_gallery_lc.h5")
    meta_path = tmp + ".meta.npz"
    reuse = os.environ.get("GALLERY_REUSE") and os.path.exists(tmp) \
        and os.path.exists(meta_path)
    if reuse:
        print("Reusing cached lightcone ...", flush=True)
        dj = (DiscoDJ(dim=3, res=RES, boxsize=BOXSIZE, cosmo="Planck18EEBAOSN")
              .with_timetables())
        d = np.load(meta_path, allow_pickle=True)
        observer = d["observer"]
        spec = MapSpec(nside=NSIDE,
                       a_edges=np.geomspace(A_FAR, A_NEAR, N_SHELLS + 1),
                       weighted=True)
        summary = {"n_particles": int(d["n_particles"]),
                   "n_replicas": int(d["n_replicas"]), "maps": d["maps"]}
        density_shells = summary["maps"]
        delta_L = d["delta_L"]
    else:
        dj, observer, spec, summary = generate_lightcone(tmp)
        density_shells = summary["maps"]
        delta_L = linear_overdensity_flat(dj)   # for the linear galaxy bias
        np.savez(meta_path, observer=observer, maps=density_shells,
                 n_particles=summary["n_particles"],
                 n_replicas=summary["n_replicas"], delta_L=delta_L)

    gal_cell = galaxy_cell_mask(delta_L)
    print(f"Galaxy linear bias: n_bar={GAL_NBAR}, b={GAL_BIAS} -> "
          f"{gal_cell.mean()*100:.2f}% of cells occupied", flush=True)

    p_density = os.path.join(FIGDIR, "sky_density.png")
    p_kappa = os.path.join(FIGDIR, "sky_convergence.png")
    p_web = os.path.join(FIGDIR, "cosmic_web_wedge.png")
    p_rsd = os.path.join(FIGDIR, "rsd_comparison.png")
    p_nz = os.path.join(FIGDIR, "nz_distribution.png")
    p_gsky = os.path.join(FIGDIR, "galaxy_sky.png")
    p_gweb = os.path.join(FIGDIR, "galaxies_web.png")

    print(f"Rendering smooth-field maps (nside={NSIDE}) ...", flush=True)
    fig_sky_density(spec, density_shells, p_density)
    fig_convergence(dj, spec, density_shells, p_kappa)

    print("Reading catalogue for the web / RSD / n(z) / galaxies / 3-D ...",
          flush=True)
    cat = read_catalogue(tmp, observer, dj, gal_cell)
    fig_cosmic_web(cat, p_web)
    fig_galaxy_sky(cat, p_gsky)
    fig_galaxies_web(cat, p_gweb)
    fig_rsd(cat, p_rsd)
    fig_nz(cat, p_nz)

    print("Building interactive 3-D + HTML gallery ...", flush=True)
    plotly_div = interactive_3d_html(cat)
    figs = [
        ("Projected dark-matter density",
         f"Mass painted onto a HEALPix sky (nside={NSIDE}) from the lightcone "
         "shells — the smooth large-scale-structure field a survey integrates "
         "through. Deposited with phase-space-sheet over-sampling, so it is the "
         "correct fixed-mass-per-tetrahedron density, not a grid of points.",
         p_density),
        ("Weak-lensing convergence &kappa;",
         "Born-approximation convergence from the overdensity shells. Fully "
         "differentiable, so &part;C<sub>&ell;</sub>/&part;&theta; flows back to cosmology.",
         p_kappa),
        ("The cosmic web on the lightcone",
         "A thin equatorial slice of the catalogue as a smooth 2-D density — "
         "filaments, knots and voids. Phase-space-sheet over-sampling deposits "
         "mass <em>inside</em> each Lagrangian cell, so there is no grid-vertex "
         "aliasing.",
         p_web),
        ("Galaxies from linear Lagrangian bias",
         f"Each Lagrangian cell hosts a galaxy with probability "
         f"p = n&#772;(1 + b&middot;&delta;<sub>L</sub>), b={GAL_BIAS:g} "
         "(&delta;<sub>L</sub> = the linear initial overdensity). The galaxies "
         "are then advected to their Eulerian lightcone positions — a clean "
         "linearly-biased mock that clusters more strongly than the mass.",
         p_gweb),
        ("Galaxy angular clustering",
         "The same sheet-selected galaxies on the sky — the angular galaxy "
         "density a survey would measure.",
         p_gsky),
        ("Redshift-space distortions",
         "The same structures in real space vs redshift space: peculiar "
         "velocities (from the stored radial velocity) stretch them along the "
         "line of sight — the signal RSD analyses target.",
         p_rsd),
        ("Redshift distribution n(z)",
         "Cosmological vs observed (RSD) redshifts for all mass, plus the "
         "sheet-selected galaxies (rescaled) — the galaxy n(z) a survey sees.",
         p_nz),
    ]
    out_html = os.path.join(HERE, "lightcone_gallery.html")
    build_html(figs, plotly_div, summary, out_html)

    if not os.environ.get("GALLERY_REUSE"):
        for p in (tmp, meta_path):
            try:
                os.remove(p)
            except OSError:
                pass
    print("\nGallery complete:", out_html, flush=True)


if __name__ == "__main__":
    main()
