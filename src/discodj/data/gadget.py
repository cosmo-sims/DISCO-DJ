import numpy as onp

__all__ = ["load_gadget"]


# Load Gadget file
def load_gadget(filename: str, boxsize_in_kpc_per_h: bool = False) -> dict:
    """
    Load Gadget snapshot file.

    :param filename: filename of the Gadget snapshot file
    :param boxsize_in_kpc_per_h: if True: boxsize in kpc/h, if False: boxsize in Mpc/h
    :return: dictionary with the following keys: "pos", "vel", "a", plus setting keys
    """
    try:
        import readgadget
    except ImportError:
        raise ImportError("Pylians needs to be installed for loading Gadget files!")
    header = readgadget.header(filename)
    if boxsize_in_kpc_per_h:
        box_size = header.boxsize / 1e3  # Mpc/h
    else:
        box_size = header.boxsize
    n_all = header.nall  # Total number of particles
    print(f"Number of particles for each type: {n_all}")
    masses = header.massarr * 1e10  # Masses of the particles in Msun/h
    Omega_m = header.omega_m  # value of Omega_m
    Omega_de = header.omega_l  # value of Omega_de
    h = header.hubble  # value of h
    redshift = header.redshift  # redshift of the snapshot
    Hubble = 100.0 * onp.sqrt(Omega_m * (1.0 + redshift) ** 3 + Omega_de)  # Value of H(z) in km/s/(Mpc/h)

    settings = {"boxsize": box_size, "n_all": n_all, "Omega_m": Omega_m, "Omega_de": Omega_de, "h": h, "z_i": redshift}

    ptype = [1]  # dark matter is particle type 1
    ids = onp.argsort(readgadget.read_block(filename, "ID  ", ptype) - 1)  # IDs starting from 0
    pos = readgadget.read_block(filename, "POS ", ptype)[ids]
    if boxsize_in_kpc_per_h:
          pos = pos / 1e3  # convert to positions in Mpc/h
    vel = readgadget.read_block(filename, "VEL ", ptype)[ids]  # peculiar velocities in km/s
    a = 1. / (1 + redshift)
    # NOTE: readgadget already multiplies Gadget velocity (sqrt(a) dx/dt) by sqrt(a) -> a dx/dt peculiar velocity
    # now, make it canonical momentum (* a) and convert from km/s -> 100 km/s for DiscoDJ (Mpc/h * H0 = 100 km/s)
    vel = vel / 100 * a
    return {"pos": pos, "vel": vel, "a": a, **settings}
