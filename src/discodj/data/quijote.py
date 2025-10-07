import os
import numpy as onp
from tqdm import tqdm

__all__ = ["load_quijote_ini", "load_snaps"]


# Load Quijote initial condition file
def load_quijote_ini(filename: str, sub_spatial: int = 1, glass_size: int = 64) -> dict:
    """
    This function loads the initial conditions of a Quijote simulation.
    It subsamples the data to a manageable size if desired.

    :param filename: filename
    :param sub_spatial: spatial subsampling (1: extract full data, 2: only every 2nd particle per dim., etc.)
    :param glass_size: size of the glass file in each dimension
    :return: dictionary with the following keys: "pos", "vel", "a", plus setting keys
    """
    try:
        import readgadget
    except ImportError:
        raise ImportError("Pylians needs to be installed for loading Gadget files!")
    header = readgadget.header(filename)
    box_size = header.boxsize / 1e3  # Mpc/h
    n_all = header.nall  # Total number of particles
    print(f"Number of particles for each type: {n_all}")
    masses = header.massarr * 1e10  # Masses of the particles in Msun/h
    Omega_m = header.omega_m  # value of Omega_m
    Omega_de = header.omega_l  # value of Omega_de
    h = header.hubble  # value of h
    redshift = header.redshift  # redshift of the snapshot

    settings = {"boxsize": box_size, "n_all": n_all, "Omega_m": Omega_m, "Omega_de": Omega_de, "h": h, "z_i": redshift}

    ptype = [1]  # dark matter is particle type 1
    ids_i = onp.argsort(readgadget.read_block(filename, "ID  ", ptype) - 1)  # IDs starting from 0
    pos_i = readgadget.read_block(filename, "POS ", ptype)[ids_i] / 1e3  # positions in Mpc/h
    vel_i = readgadget.read_block(filename, "VEL ", ptype)[ids_i]  # peculiar velocities in km/s

    # Prepare subsampling
    assert 0 < sub_spatial < 512

    remaining_size = 512 // glass_size

    # Reordering data for simple reshaping
    pos_i = pos_i.reshape(remaining_size, remaining_size, remaining_size, glass_size, glass_size, glass_size, 3).transpose(
        0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)
    vel_i = vel_i.reshape(remaining_size, remaining_size, remaining_size, glass_size, glass_size, glass_size, 3).transpose(
        0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)

    a_i = 1. / (1 + redshift)

    pos_i = pos_i.reshape([512, 512, 512, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape([-1, 3])
    vel_i = (vel_i / 100 * a_i).reshape([512, 512, 512, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape(
        [-1, 3])  # Note: 100 comes from Hubble!

    return {"pos": onp.asarray(pos_i), "vel": onp.asarray(vel_i), "a": a_i, **settings}


def load_snaps(filename_or_folder: str, snap_inds: list | None = None, sub: int = 1, sub_spatial: int = 1,
               load_raw: bool = False, save_raw: bool = False) -> dict:
    """
    This function loads the snapshots of a CAMELS DM-only simulation. It subsamples the data to a manageable size if
    desired.

    :param filename_or_folder: filename or folder with the data (if folder: use the format "folder/snap_%03d.hdf5")
    :param sub: extract only every sub-th snapshot
    :param snap_inds: indices of the snapshots to be extracted
    :param sub_spatial: spatial subsampling factor (1: extract full data, 2: only every 2nd particle per dimension, etc.)
    :param load_raw (bool): this is being ignored at the moment and only here for consistency with the other data loaders
    :param save_raw (bool): this is being ignored at the moment and only here for consistency with the other data loaders
    :return: dictionary with the following keys: "pos", "vel", "a"
    """
    try:
        import readgadget
    except ImportError:
        raise ImportError("Pylians needs to be installed for loading Gadget files!")
    scales = []
    poss = []
    vels = []

    # Prepare subsampling
    assert 0 < sub_spatial < 512

    for i in tqdm(snap_inds):
        snapshot = os.path.join(filename_or_folder, 'snap_%03d') % i

        header = readgadget.header(snapshot)
        box_size = header.boxsize / 1e3  # Mpc/h
        redshift = header.redshift  # redshift of the snapshot

        ptype = [1]  # dark matter is particle type 1
        ids = onp.argsort(readgadget.read_block(snapshot, "ID  ", ptype) - 1)  # IDs starting from 0
        pos = readgadget.read_block(snapshot, "POS ", ptype)[ids] / 1e3  # positions in Mpc/h
        vel = readgadget.read_block(snapshot, "VEL ", ptype)[ids]  # peculiar velocities in km/s

        # Reordering data for simple reshaping
        pos = pos.reshape(8, 8, 8, 64, 64, 64, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)
        vel = vel.reshape(8, 8, 8, 64, 64, 64, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)

        a = 1. / (1 + redshift)

        pos = pos.reshape([512, 512, 512, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape([-1, 3])
        vel = (vel / 100 * a).reshape([512, 512, 512, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape([-1, 3])

        scales.append(a)
        poss.append(pos)
        vels.append(vel)

    # Reduce amount of time steps
    scales = scales[::sub]
    poss = poss[::sub]
    vels = vels[::sub]

    return {"a": scales, "pos": poss, "vel": vels}
