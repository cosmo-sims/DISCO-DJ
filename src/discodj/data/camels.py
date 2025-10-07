import os
import numpy as onp
from tqdm import tqdm

__all__ = ["load_camels_dm_ini", "load_snaps_dm"]


# Load Camels initial condition file
def load_camels_dm_ini(filename: str, sub_spatial: int = 1, glass_size: int = 64) -> dict:
    """
    This function loads the initial conditions of a CAMELS DM-only simulation.
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
    assert 0 < sub_spatial < 256

    remaining_size = 256 // glass_size

    # Reordering data for simple reshaping
    pos_i = pos_i.reshape(remaining_size, remaining_size, remaining_size, glass_size, glass_size, glass_size,
                          3).transpose(
        0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)
    vel_i = vel_i.reshape(remaining_size, remaining_size, remaining_size, glass_size, glass_size, glass_size,
                          3).transpose(
        0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)

    a_i = 1. / (1 + redshift)

    pos_i = pos_i.reshape([256, 256, 256, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape([-1, 3])
    vel_i = (vel_i / 100 * a_i).reshape([256, 256, 256, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape(
        [-1, 3])  # Note: 100 comes from Hubble!

    return {"pos": onp.asarray(pos_i), "vel": onp.asarray(vel_i), "a": a_i, **settings}


def load_snaps_dm(filename_or_folder: str, sub: int = 1, sub_spatial: int = 1, load_raw: bool = False,
                  save_raw: bool = False, snap_inds: list | None = None) -> dict:
    """This function loads the snapshots of a CAMELS DM-only simulation. It subsamples the data to a manageable size if
    desired.

    :param filename_or_folder: filename or folder with the data (if folder: use the format "folder/snap_%03d.hdf5")
    :param sub: extract only every sub-th snapshot
    :param sub_spatial: spatial subsampling factor (1: extract full data, 2: only every 2nd particle per dimension, etc.)
    :param load_raw: if True: load the raw data from the snapshots, if False: load the stored numpy files
    :param save_raw: if True: save the raw data from the snapshots as numpy files
    :param snap_inds: indices of the snapshots to load
    :return: dictionary with the following keys: "pos", "vel", "a"
    """
    try:
        import readgadget
    except ImportError:
        raise ImportError("Pylians needs to be installed for loading Gadget files!")

    scales = []
    poss = []
    vels = []

    # Loading all the intermediate snapshots
    n_snaps_all = 34  # originally: 34, there are 34 snapshots in total

    # Prepare subsampling
    assert 0 < sub_spatial < 256
    n_per_dim = 256 // sub_spatial

    if load_raw:
        ind_vec = snap_inds or range(n_snaps_all)

        for i in tqdm(ind_vec):
            snapshot = os.path.join(filename_or_folder, 'snap_%03d.hdf5') % i

            header = readgadget.header(snapshot)
            box_size = header.boxsize / 1e3  # Mpc/h
            redshift = header.redshift  # redshift of the snapshot

            ptype = [1]  # dark matter is particle type 1
            ids = onp.argsort(readgadget.read_block(snapshot, "ID  ", ptype) - 1)  # IDs starting from 0
            pos = readgadget.read_block(snapshot, "POS ", ptype)[ids] / 1e3  # positions in Mpc/h
            vel = readgadget.read_block(snapshot, "VEL ", ptype)[ids]  # peculiar velocities in km/s

            # Reordering data for simple reshaping
            pos = pos.reshape(4, 4, 4, 64, 64, 64, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)
            vel = vel.reshape(4, 4, 4, 64, 64, 64, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)

            a = 1. / (1 + redshift)

            pos = pos.reshape([256, 256, 256, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape([-1, 3])
            vel = (vel / 100 * a).reshape([256, 256, 256, 3])[::sub_spatial, ::sub_spatial, ::sub_spatial, :].reshape(
                [-1, 3])

            scales.append(a)
            poss.append(pos)
            vels.append(vel)

        if save_raw:
            onp.savez(os.path.join(filename_or_folder, f"snaps_{sub}_{sub_spatial}_{snap_inds}.npz"),
                     poss=poss, vels=vels, scales=scales)

    else:
        if snap_inds is not None:
            print(f"Warning: when loading data from .npz file, 'snap_inds'={snap_inds} will be ignored!")

        try:
            data_all = onp.load(filename_or_folder, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {filename_or_folder} does not exist! Run this function with load_raw=True first!")

        scales = [d for d in data_all["scales"]]
        poss = [d for d in data_all["poss"]]
        vels = [d for d in data_all["vels"]]

        assert poss[0].shape[0] == n_per_dim ** 3, (f"Data in {filename_or_folder} has resolution {poss[0].shape[0]}, "
                                                    f"but {n_per_dim ** 3} is requested! Run again with save_raw=False!")

    # Reduce amount of time steps
    scales = scales[::sub]
    poss = poss[::sub]
    vels = vels[::sub]

    return {"a": scales, "pos": poss, "vel": vels}
