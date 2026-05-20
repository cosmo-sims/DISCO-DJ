import os as _os
import numpy as onp
from ..core.types import AnyArray
from ..cosmology.cosmology import Cosmology

# Tell Blosc / Blosc2 how many threads to use *before* any write happens; the
# library reads this env var lazily during compress(), so setting it now is
# enough. Without this, Blosc defaults to 1 thread and writes go ~5x slower.
# We only set it if the user hasn't already chosen a value.
_os.environ.setdefault("BLOSC_NTHREADS", str(_os.cpu_count() or 1))

# Eager import so the HDF5 plugin path is registered before any h5py.File is
# opened by anyone importing this module. Lazy import inside the codec helper
# is too late for *reads* — h5py looks up the plugin path on first file open.
try:
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:
    _hdf5plugin = None

__all__ = ["save_as_hdf5", "save_lightcone_as_hdf5",
           "compression_kwargs", "read_lightcone_header"]


def read_lightcone_header(path: str) -> dict:
    """Read the ``/Header`` group attributes of a lightcone HDF5 file.

    Returns a plain dict of attribute name -> python value (np arrays become
    np arrays, scalars are unwrapped). Also adds:
      - ``"n_particles"``: total row count assembled from NumPart_Total +
        NumPart_Total_HighWord (handles >2^32 row catalogues).
      - ``"v_mode"``: ``"radial"`` if /PartType1 has ``RadialVelocity``,
        ``"full"`` if it has ``Velocities``, else ``None``.
      - ``"has_particle_idx"``: True if /PartType1 has
        ``LagrangianParticleIndex``.
    """
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        if "Header" in f:
            for k, v in f["Header"].attrs.items():
                # unwrap zero-d arrays/np scalars for ergonomic access
                if hasattr(v, "shape") and v.shape == ():
                    out[k] = v.item()
                else:
                    out[k] = onp.asarray(v) if hasattr(v, "shape") else v
        # Particle-count assembly
        low_arr = onp.asarray(out.get("NumPart_Total", [0] * 6))
        high_arr = onp.asarray(out.get("NumPart_Total_HighWord", [0] * 6))
        out["n_particles"] = int(low_arr[1]) + (int(high_arr[1]) << 32)
        # v_mode detection
        ds = list(f["PartType1"].keys()) if "PartType1" in f else []
        if "RadialVelocity" in ds:
            out["v_mode"] = "radial"
        elif "Velocities" in ds:
            out["v_mode"] = "full"
        else:
            out["v_mode"] = None
        out["has_particle_idx"] = "LagrangianParticleIndex" in ds
    return out


def compression_kwargs(compression):
    """Resolve the user-facing ``compression`` argument to a dict of
    ``h5py.create_dataset`` kwargs.

    Codecs (require the ``hdf5plugin`` package unless noted):
      - True / "zstd"        -> Blosc(cname='zstd', clevel=3, SHUFFLE).
        Multi-threaded (BLOSC_NTHREADS); ~5x faster than the reference zstd
        at the cost of ~10% larger files. Best default for big catalogues.
      - "blosc2"             -> Blosc2(cname='zstd', clevel=3, SHUFFLE).
        Also multi-threaded; ~7% smaller files than "zstd" but ~50% slower.
      - "zstd_serial"        -> reference single-thread zstd + shuffle.
        Smallest files but write throughput capped at one core.
      - "lz4"                -> Blosc(cname='lz4', SHUFFLE). Multi-threaded;
        fastest with compression, biggest of the lossless options.
      - False / None / "none" -> no HDF5 compression (raw write)
      - "gzip"               -> gzip + shuffle + fletcher32 (slowest; legacy)

    BLOSC_NTHREADS is set automatically at module import (= os.cpu_count()).
    """
    codec = compression
    if codec is True:
        codec = "zstd"
    if codec in (None, False, "none"):
        return {}
    if codec == "gzip":
        return {"compression": "gzip", "shuffle": True, "fletcher32": True}
    if codec in ("zstd", "blosc2", "lz4", "zstd_serial"):
        import hdf5plugin
        if codec == "zstd":
            # Blosc v1 + zstd + SHUFFLE: fastest multi-threaded compressed write
            # in the benches at chunk_size >= 1M rows. 4.1s / 13 GB at 512^3 vs
            # 22.9s for zstd_serial (single thread).
            return dict(hdf5plugin.Blosc(cname="zstd", clevel=3,
                                         shuffle=hdf5plugin.Blosc.SHUFFLE))
        if codec == "blosc2":
            return dict(hdf5plugin.Blosc2(cname="zstd", clevel=3,
                                          filters=hdf5plugin.Blosc2.SHUFFLE))
        if codec == "zstd_serial":
            return {**hdf5plugin.Zstd(clevel=3), "shuffle": True}
        # lz4
        return dict(hdf5plugin.Blosc(cname="lz4", clevel=3,
                                     shuffle=hdf5plugin.Blosc.SHUFFLE))
    raise ValueError(f"unknown compression {compression!r}; expected one of "
                     "True/False/None/'gzip'/'lz4'/'zstd'/'blosc2'/'zstd_serial'/'none'")


def save_as_hdf5(filename: str, cosmo: Cosmology, boxsize: float, x: AnyArray, p: AnyArray, a: AnyArray | float,
                 format_str: str = "gadget123", compressed: bool = True):
    """Save the particle positions and velocities to an HDF5 file in the Gadget (or similar) format.
    NOTE: this file uses the standard Gadget unit system, i.e. masses in 10^10 M_sun, velocities in km/s
    (note that Gadget sqrt(a) convention!), coordinates in Mpc/h.

    :param filename: path to the file to be written
    :param cosmo: cosmology object
    :param boxsize: size of the box in Mpc/h
    :param x: positions of the particles
    :param p: velocities of the particles
    :param a: scale factor at which the particles are saved
    :param format_str: format of the file, either "gadget123", "gadget4" or "swift"
    :param compressed: whether to use compression for the HDF5 file
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py needs to be installed for saving to HDF5!")

    dim = x.shape[-1]
    assert dim == 3, "Only 3D supported at the moment!"

    # First: write header
    with h5py.File(filename, 'w') as f:
        header_group = f.create_group('Header')

        # Write attributes
        npart = x.shape[0]
        header_group.attrs[u'Time'] = a
        header_group.attrs[u'Redshift'] = 1.0 / a - 1.0
        header_group.attrs[u'NumPart_ThisFile'] = [0, npart, 0, 0, 0, 0]  # only DM particles
        header_group.attrs[u'NumPart_Total_HighWord'] = [0, 0, 0, 0, 0, 0]
        header_group.attrs[u'NumPart_Total'] = header_group.attrs[u'NumPart_ThisFile']  # only single file supported
        header_group.attrs[u'NumFilesPerSnapshot'] = 1
        # Constants:
        G = 43.007105731706317  # in Mpc / 10^10 Msun (km/s)^2
        Hubble = 100.0
        boxsize_in_Mpc_per_h = boxsize  # we're using Mpc/h
        # particle mass in 10^10 Msun / h
        particle_mass = cosmo.Omega_m * 3 * Hubble * Hubble / (8 * onp.pi * G) \
                        * boxsize_in_Mpc_per_h ** 3 / npart
        header_group.attrs[u'MassTable'] = [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0]

        if format_str == "swift":
            header_group.attrs[u'BoxSize'] = boxsize_in_Mpc_per_h / cosmo.h

            # Write SWIFT snapshot attributes
            cosmology_group = f.create_group('Cosmology')
            cosmology_group.attrs[u'Omega_m'] = cosmo.Omega_m
            cosmology_group.attrs[u'Omega_lambda'] = cosmo.Omega_de
            cosmology_group.attrs[u'h'] = cosmo.h
        elif format_str == "gadget4":
            header_group.attrs[u'BoxSize'] = boxsize_in_Mpc_per_h

            # Write Gadget-4 snapshot attributes
            parameters_group = f.create_group('Parameters')
            parameters_group.attrs[u'Omega0'] = cosmo.Omega_m
            parameters_group.attrs[u'OmegaLambda'] = cosmo.Omega_de
            parameters_group.attrs[u'HubbleParam'] = cosmo.h
        elif format_str == "gadget123":
            header_group.attrs[u'BoxSize'] = boxsize_in_Mpc_per_h

            # Write traditional Gadget-1/2/3 snapshot attributes
            header_group.attrs[u'Omega0'] = cosmo.Omega_m
            header_group.attrs[u'OmegaLambda'] = cosmo.Omega_de
            header_group.attrs[u'HubbleParam'] = cosmo.h
        else:
            raise NotImplementedError

        # Write blocks
        prefix = 'PartType%d/' % 1  # only DM particles
        blocks = ("POS ", "VEL ", "ID  ", "MASS ")
        data = (x, p, onp.arange(npart, dtype=onp.uint32),
                onp.full(npart, particle_mass, dtype=onp.float32))
        for block_name, block_data in zip(blocks, data):
            if block_name == "POS ":
                suffix = "Coordinates"
                block_data = block_data.astype(onp.float32)
                if format_str == "swift":
                    block_data /= cosmo.h
            elif block_name == "MASS ":
                suffix = "Masses"
            elif block_name == "ID  ":
                suffix = "ParticleIDs"
            elif block_name == "VEL ":
                suffix = "Velocities"
                # p is comoving, in 100 km/s -> need to multiply by 100
                # also, our p = a^2 dx/dt, but Gadget expects a dx/dt / sqrt(a) -> divide by a ** 3/2
                block_data *= 100 / a ** 1.5
                if format_str == "swift":
                    block_data *= onp.sqrt(a)
            else:
                raise Exception('Block not implemented in write_blocks_to_hdf5!')

            dataset_name = prefix + suffix
            if compressed:
                f.create_dataset(dataset_name, data=block_data, compression="gzip", shuffle=True, fletcher32=True)
            else:
                f.create_dataset(dataset_name, data=block_data)

    print(f"Snapshot written to {filename}.")


def save_lightcone_as_hdf5(filename: str, cosmo: Cosmology, boxsize: float,
                            x: AnyArray, v: AnyArray, a_cross: AnyArray,
                            observer: AnyArray,
                            mask: AnyArray | None = None,
                            replica_idx: AnyArray | None = None,
                            shell_idx: AnyArray | None = None,
                            particle_idx: AnyArray | None = None,
                            n_part_per_replica: int | None = None,
                            compression: str | bool | None = "zstd",
                            storage_chunk_rows: int = 1 << 20):
    """Save a past-lightcone particle catalogue to HDF5 (Gadget-like layout).

    Differences from save_as_hdf5:
      - Per-particle a_cross written as ScaleFactor dataset.
      - Velocities converted to Gadget convention using each particle's own a_cross.
      - Optional mask applied host-side (streaming variants already pre-filter).
      - Header carries a LightconeMode flag and the Observer position.

    :param filename: output HDF5 path.
    :param cosmo: cosmology object.
    :param boxsize: simulation box size in Mpc/h.
    :param x: (M, 3) array of crossing positions in Mpc/h (un-wrapped, lightcone coords).
    :param v: (M, 3) array of dPsi/d(tildet) = a^2 dx/dt velocities at crossing.
    :param a_cross: (M,) per-particle scale factor at crossing.
    :param observer: (3,) observer position in box coords.
    :param mask: optional (M,) bool array. If None, all rows are written.
    :param replica_idx: (M,) optional replica labels (written if provided).
    :param shell_idx: (M,) optional shell labels (written if provided).
    :param particle_idx: (M,) optional Lagrangian particle index (written if provided).
    :param n_part_per_replica: optional, particles per replica (for mass scaling). If None,
        inferred from boxsize and the un-masked particle count using a single-box assumption.
    :param compression: codec name (``"zstd"`` default), or ``True`` (= ``"zstd"``),
        ``"gzip"``, ``"lz4"``, ``"blosc"``, ``False``/``None``/``"none"`` to disable.
        zstd/lz4/blosc require the ``hdf5plugin`` package.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py needs to be installed for saving to HDF5!")

    x = onp.asarray(x)
    v = onp.asarray(v)
    a_cross = onp.asarray(a_cross)
    observer = onp.asarray(observer, dtype=onp.float64)

    assert x.shape[-1] == 3, "Lightcone output is 3-D only."
    assert x.shape[0] == v.shape[0] == a_cross.shape[0], \
        "x, v, a_cross must agree on the leading axis."

    if mask is not None:
        mask = onp.asarray(mask).astype(bool)
        assert mask.shape[0] == x.shape[0], "mask must have the same leading axis as x."
        x = x[mask].astype(onp.float32)
        v = v[mask].astype(onp.float32)
        a_cross = a_cross[mask].astype(onp.float32)
        if replica_idx is not None:
            replica_idx = onp.asarray(replica_idx)[mask]
        if shell_idx is not None:
            shell_idx = onp.asarray(shell_idx)[mask]
        if particle_idx is not None:
            particle_idx = onp.asarray(particle_idx)[mask].astype(onp.int32)
    else:
        x = x.astype(onp.float32)
        v = v.astype(onp.float32)
        a_cross = a_cross.astype(onp.float32)
        if replica_idx is not None:
            replica_idx = onp.asarray(replica_idx)
        if shell_idx is not None:
            shell_idx = onp.asarray(shell_idx)
        if particle_idx is not None:
            particle_idx = onp.asarray(particle_idx).astype(onp.int32)

    npart = x.shape[0]

    # Velocity conversion: v_gadget = v * 100 / a^1.5 per-particle (see save_as_hdf5 comment).
    # Supports both the full 3-D vector ((M, 3)) and the line-of-sight scalar ((M,))
    # produced by the v_mode='radial' lightcone path.
    a_safe = onp.where(a_cross > 0, a_cross, 1.0)  # avoid 0**1.5 in masked-out rows (already filtered, but defensive)
    v_is_radial = (v.ndim == 1)
    if v_is_radial:
        v_gadget = v * (100.0 / a_safe ** 1.5)
    else:
        v_gadget = v * (100.0 / a_safe[:, None] ** 1.5)

    # Mass per particle. If n_part_per_replica is unset we infer the
    # Lagrangian-space cardinality (= particles per box, before replication
    # and before each particle's multiple-shell crossings).
    if n_part_per_replica is None:
        if particle_idx is not None and particle_idx.size > 0:
            # particle_idx is the per-row Lagrangian particle index in [0, N^3).
            n_part_per_replica = int(particle_idx.max()) + 1
        elif replica_idx is not None and replica_idx.size > 0:
            # Fallback heuristic: npart is total crossings (= particles per
            # replica * shells crossed per particle * R). Without particle_idx
            # we cannot recover the true Lagrangian count, so divide by R as
            # a coarse lower bound and hope each particle crosses ~1 shell.
            n_replicas = int(replica_idx.max()) + 1
            n_part_per_replica = max(npart // max(n_replicas, 1), 1)
        else:
            n_part_per_replica = npart
    G = 43.007105731706317
    Hubble = 100.0
    particle_mass = cosmo.Omega_m * 3 * Hubble * Hubble / (8 * onp.pi * G) \
                    * boxsize ** 3 / max(n_part_per_replica, 1)

    with h5py.File(filename, "w") as f:
        header = f.create_group("Header")
        header.attrs["LightconeMode"] = 1
        header.attrs["Observer"] = observer
        header.attrs["BoxSize"] = boxsize
        # Gadget convention: 32-bit per-type counts with HighWord for the
        # upper 32 bits. Required for N >= 1024 lightcone catalogues.
        low = npart & 0xFFFFFFFF
        high = npart >> 32
        header.attrs["NumPart_ThisFile"] = [0, low, 0, 0, 0, 0]
        header.attrs["NumPart_Total"] = [0, low, 0, 0, 0, 0]
        header.attrs["NumPart_Total_HighWord"] = [0, high, 0, 0, 0, 0]
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 1.0  # not meaningful for a lightcone
        header.attrs["Omega0"] = cosmo.Omega_m
        header.attrs["OmegaLambda"] = cosmo.Omega_de
        header.attrs["HubbleParam"] = cosmo.h
        header.attrs["MassTable"] = [0.0, particle_mass, 0.0, 0.0, 0.0, 0.0]
        header.attrs["NumPart_PerReplica"] = n_part_per_replica

        prefix = "PartType1/"
        ds_kwargs = compression_kwargs(compression)

        def _write(name, data):
            # Explicit chunks (1M rows by default) — Blosc multi-threading
            # only kicks in at chunk sizes >= ~10 MB. h5py's auto-chunker
            # picks 280 KB chunks which leave the codec stuck single-threaded.
            if ds_kwargs and storage_chunk_rows and data.shape[0] >= storage_chunk_rows:
                n = min(storage_chunk_rows, data.shape[0])
                chunks = (n,) + tuple(data.shape[1:])
                f.create_dataset(prefix + name, data=data, chunks=chunks, **ds_kwargs)
            else:
                f.create_dataset(prefix + name, data=data, **ds_kwargs)

        _write("Coordinates", x)
        if v_is_radial:
            _write("RadialVelocity", v_gadget)
        else:
            _write("Velocities", v_gadget)
        _write("ScaleFactor", a_cross)
        # uint64 because lightcone catalogues exceed 2^32 rows at N >= 1024.
        _write("ParticleIDs", onp.arange(npart, dtype=onp.uint64))
        _write("Masses", onp.full(npart, particle_mass, dtype=onp.float32))
        if replica_idx is not None:
            _write("ReplicaIndex", replica_idx)
        if shell_idx is not None:
            _write("ShellIndex", shell_idx)
        if particle_idx is not None:
            _write("LagrangianParticleIndex", particle_idx)

    print(f"Lightcone written to {filename} ({npart} particles).")
