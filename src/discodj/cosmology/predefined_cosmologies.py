# This file contains predefined cosmologies for convenience.
__all__ = ["get_cosmology_dict_from_name"]


def get_cosmology_dict_from_name(name: str) -> dict:
    """Get a cosmology dictionary from a given name.

    :param name: Name of the cosmology.
    :return: Cosmology dictionary.
    """
    # Planck 2015 paper XII Table 4 final column (best fit)
    if name == "Planck15":
        dict_out = dict(Omega_c=0.2589,
                        Omega_b=0.04860,
                        Omega_k=0.0,
                        h=0.6774,
                        n_s=0.9667,
                        sigma8=0.8159,
                        w0=-1.0,
                        wa=0.0,
                        )

    # Default in Monofonic: Planck2018EE+BAO+SN
    elif name == "Planck18EEBAOSN":
        dict_out = dict(Omega_c=0.259622,
                        Omega_b=0.0488911,
                        Omega_k=0.0,
                        h=0.67742,
                        n_s=0.96822,
                        sigma8=0.8105,
                        w0=-1.0,
                        wa=0.0,
                        )

    # Camels CV values
    elif name == "CamelsCV":
        dict_out = dict(Omega_c=0.3 - 0.049,
                        Omega_b=0.049,
                        Omega_k=0.0,
                        h=0.6711,
                        n_s=0.9624,
                        sigma8=0.8,
                        w0=-1.0,
                        wa=0.0,
                        )

    # Quijote fiducial values
    elif name == "Quijote":
        dict_out = dict(Omega_c=0.3175 - 0.049,
                        Omega_b=0.049,
                        Omega_k=0.0,
                        h=0.6711,
                        n_s=0.9624,
                        sigma8=0.834,
                        w0=-1.0,
                        wa=0.0,
                        )
    else:
        raise ValueError(f"Unknown cosmology: {name}.")

    return dict_out
