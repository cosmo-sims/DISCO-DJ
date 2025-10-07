import numpy as onp
from ..core.types import AnyArray
from ..cosmology.cosmology import Cosmology

__all__ = ["save_as_hdf5"]


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
