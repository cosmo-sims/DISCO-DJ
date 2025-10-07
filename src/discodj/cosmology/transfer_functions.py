# This file contains different transfer functions that can be used in the DiscoDJ cosmology object.
# Note that not all transfer functions take into account all cosmological parameters.
import jax
import jax.numpy as jnp
from jax import Array
from .cosmology import Cosmology

__all__ = ["transfer_dict"]


def disco_eb(cosmo: Cosmology, k: Array, kmin: float = 1e-5, kmax: float = None, nmodes: int = 256,
             Tcmb: float = 2.72548) -> Array:
    """Compute the DiscoEB transfer function.

    :param cosmo: DiscoDJ cosmology object
    :param k: wavenumber in h / Mpc for evaluation
    :param kmin: minimum wavenumber in h / Mpc. Default is 1e-5
    :param kmax: maximum wavenumber in h / Mpc. Default is max(kmax)
    :param nmodes: number of modes for evaluation. Default is 256
    :param Tcmb: CMB temperature in K. Default is 2.72548 (mean value in https://arxiv.org/pdf/0911.1955.pdf, Table 2)
    :return: transfer function
    """
    from discoeb.background import evolve_background
    from discoeb.perturbations import evolve_perturbations, evolve_perturbations_batched, get_power
    # TODO: Think about default values for the settings below / make them parameters
    if not jax.config.read("jax_enable_x64"):
        raise ValueError("DiscoEB currently requires 'jax_enable_x64'.\n"
                         "To use double precision, make sure to include "
                         "config.update('jax_enable_x64', True) before using any jax features!")

    dtype = k.dtype

    ## Compute background evolution
    param = {}
    param['Omegam'] = cosmo.Omega_m
    param['Omegab'] = cosmo.Omega_b
    param['w_DE_0'] = jnp.clip(cosmo.w0, min=-0.99, max=None)  # w0 must be slightly greater than -1
    param['w_DE_a'] = cosmo.wa
    param['cs2_DE'] = 0.99  # sound speed of DE
    param['Omegak'] = cosmo.Omega_k  # NOTE: Omegak is ignored at the moment!
    A_s_ref = 2.1e-9  # we are using sigma_8 here, so we take a reference value for A_s (will be rescaled outside)
    param['A_s'] = A_s_ref
    param['n_s'] = cosmo.n_s
    param['H0'] = 100 * cosmo.h
    param['Tcmb'] = Tcmb
    param['YHe'] = 0.248
    Neff = 3.046  # -1 if massive neutrino present
    N_nu_mass = 1.0
    Tnu = (4.0 / 11.0) ** (1.0 / 3.0)
    N_nu_rel = Neff - N_nu_mass * (Tnu / ((4.0 / 11.0) ** (1.0 / 3.0))) ** 4
    param['Neff'] = N_nu_rel
    param['Nmnu'] = N_nu_mass
    param['mnu'] = 0.06  # eV
    param['k_p'] = 0.05  # in Mpc^-1
    param_bg = evolve_background(param=param, thermo_module='RECFAST')
    evolve_fn = evolve_perturbations  # evolve_perturbations_batched
    y, kmodes = evolve_fn(param=param_bg, kmin=kmin, kmax=kmax or k[-1], num_k=nmodes,
                          aexp_out=jnp.array([1.0]), lmaxg=11, lmaxgp=11, lmaxr=11, lmaxnu=11,
                          nqmax=3, max_steps=2048, rtol=1e-4, atol=1e-4)  #, batch_size=min(16, nmodes)
    Pk = get_power(k=kmodes, y=y[:, 0, :], idx=4, param=param_bg)
    Pk_in_Mpc_h = Pk * cosmo.h ** 3
    kmodes_in_Mpc_h = kmodes / cosmo.h
    T = jnp.sqrt(Pk_in_Mpc_h / (kmodes_in_Mpc_h ** cosmo.n_s))
    T /= T[0]
    return jnp.interp(k, kmodes_in_Mpc_h, T).astype(dtype)


# Define the transfer function using the Eisenstein-Hu fitting formula
# (translated to Jax from https://background.uchicago.edu/~whu/transfer/tf_fit.c, see TFset_parameters())
def eisenstein_hu(cosmo: Cosmology, k: float | Array, Tcmb: float = 2.72548) -> Array:
    """Compute the Eisenstein-Hu transfer function (https://arxiv.org/pdf/astro-ph/9709112).

    :param cosmo: DiscoDJ cosmology object
    :param k: wavenumber in h / Mpc
    :param Tcmb: CMB temperature in K. Default is 2.72548 (mean value in https://arxiv.org/pdf/0911.1955.pdf, Table 2)
    :return: transfer function
    """
    # Cosmological parameters
    T_ref = 2.7  # see https://background.uchicago.edu/~whu/transfer/tf_fit.c
    theta_cmb = Tcmb / T_ref  # CMB temperature scaled by reference temperature
    theta_cmb_sq = jnp.square(theta_cmb)
    theta_cmb_4 = jnp.square(theta_cmb_sq)
    omega0hh = cosmo.Omega_m * jnp.square(cosmo.h)  # Omega0 * h^2
    f_baryon = cosmo.Omega_b / cosmo.Omega_m  # Baryon fraction
    omegabhh = omega0hh * f_baryon  # Omega_b * h^2

    # Matter-radiation equality
    z_equality = 2.50e4 * omega0hh / theta_cmb_4  # z at matter-radiation equality (Eq. 2)
    k_equality = 0.0746 * omega0hh / theta_cmb_sq / cosmo.h  # k of equality in h / Mpc^-1 (Eq. 3)

    # Baryon drag (Eq. 4)
    z_drag_b1 = 0.313 * omega0hh ** -0.419 * (1.0 + 0.607 * omega0hh ** 0.674)
    z_drag_b2 = 0.238 * omega0hh ** 0.223
    z_drag = 1291.0 * omega0hh ** 0.251 / (1.0 + 0.659 * omega0hh ** 0.828) * (1.0 + z_drag_b1 * omegabhh ** z_drag_b2)

    R_drag = 31.5 * omegabhh / theta_cmb_4 * (1000 / (1.0 + z_drag))  # (Eq. 5)
    R_equality = 31.5 * omegabhh / theta_cmb_4 * (1000 / z_equality)  # (Eq. 5)

    # Sound horizon (Eq. 6)
    sound_horizon = 2.0 / 3.0 / k_equality * jnp.sqrt(6.0 / R_equality) * jnp.log(
        (jnp.sqrt(1 + R_drag) + jnp.sqrt(R_drag + R_equality)) / (1.0 + jnp.sqrt(R_equality)))

    # Silk damping scale in h / Mpc (Eq. 7)
    k_silk = 1.6 * omegabhh ** 0.52 * omega0hh ** 0.73 * (1.0 + (10.4 * omega0hh) ** -0.95) / cosmo.h

    # CDM transfer function (Eq. 9)
    a_1 = (46.9 * omega0hh) ** 0.670 * (1.0 + (32.1 * omega0hh) ** -0.532)  # Eq. 11
    a_2 = (12.0 * omega0hh) ** 0.424 * (1.0 + (45.0 * omega0hh) ** -0.582)  # Eq. 11
    alpha_cdm = a_1 ** -f_baryon * a_2 ** (-f_baryon ** 3)  # Eq. 11
    b_1 = 0.944 / (1 + (458 * omega0hh) ** -0.708)  # Eq. 12
    b_2 = (0.395 * omega0hh) ** -0.0266  # Eq. 12
    beta_cdm = 1 / (1 + b_1 * ((1 - f_baryon) ** b_2 - 1))  # Eq. 12

    q = k / (13.41 * k_equality)  # Eq. 10, note that h cancels out
    # T_cdm_limit = a_cdm * (jnp.log(1.8 * beta_cdm * q)) / (14.2 * q ** 2)  # Eq. 9

    G_of_y = lambda y: y * (-6.0 * jnp.sqrt(1.0 + y)
                            + (2.0 + 3.0 * y) * jnp.log((jnp.sqrt(1.0 + y) + 1.0) / (jnp.sqrt(1.0 + y) - 1.0)))
    alpha_b = 2.07 * k_equality * sound_horizon * (1 + R_drag) ** -0.75 * G_of_y((1.0 + z_equality) / (1.0 + z_drag))

    # Fitting formula for the CDM transfer function
    f = 1.0 / (1.0 + (k * sound_horizon / 5.4) ** 4)  # Eq. 18
    C_fun = lambda alpha_c: 14.2 / alpha_c + 386.0 / (1.0 + 69.9 * q ** 1.08)  # Eq. 20
    logterm = lambda beta_: jnp.log(jnp.exp(1.0) + 1.8 * beta_ * q)
    T0_tilde_fun = lambda alpha_c, beta_c: (logterm(beta_c) / (logterm(beta_c) + C_fun(alpha_c) * q ** 2))  # Eq. 19
    T0_tilde = T0_tilde_fun(alpha_cdm, beta_cdm)  # Eq. 19
    T_c = f * T0_tilde_fun(1.0, beta_cdm) + (1.0 - f) * T0_tilde  # Eq. 17

    # Node shifting for sinc function
    beta_node = 8.41 * omega0hh ** 0.435  # Eq. 23
    s_tilde = sound_horizon / (1.0 + (beta_node / (k * sound_horizon)) ** 3) ** (1 / 3)  # Eq. 22

    # Now: take care of baryons
    beta_b = 0.5 + f_baryon + (3.0 - 2.0 * f_baryon) * jnp.sqrt((17.2 * omega0hh) ** 2 + 1.0)  # Eq. 24
    T_b_term_1 = (T0_tilde_fun(1.0, 1.0) / (1.0 + (k * sound_horizon / 5.2) ** 2))
    T_b_term_2 = alpha_b / (1.0 + (beta_b / (k * sound_horizon)) ** 3) * jnp.exp(-(k / k_silk) ** 1.4)
    T_b = jnp.sinc(k * s_tilde / jnp.pi) * (T_b_term_1 + T_b_term_2)  # Eq. 21

    # Put everything together
    T_tot_cdm = f_baryon * T_b + (1.0 - f_baryon) * T_c  # Eq. 8

    return T_tot_cdm


def bbks(cosmo: Cosmology, k: float | Array, Neff: float = 3.046) -> Array:
    """BBKS transfer function (https://articles.adsabs.harvard.edu/pdf/1986ApJ...304...15B).

    :param cosmo: DiscoDJ cosmology object
    :param k: wavenumber in h / Mpc
    :param Neff: effective number of neutrino species. Default is 3.046
    :return: transfer function
    """
    rhor_over_rhogamma = 1.0 + Neff * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
    theta = rhor_over_rhogamma / 1.68
    omegamhh = cosmo.Omega_m * cosmo.h ** 2
    omegabhh = cosmo.Omega_b * cosmo.h ** 2
    q = k * theta ** 0.5 / (omegamhh - omegabhh)
    T_cdm = (jnp.log(1.0 + 2.34 * q)
             / (2.34 * q) * (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25)
    return T_cdm


def genetic_algorithm(cosmo: Cosmology, k: float | Array):
    """Genetic algorithm for transfer function (https://arxiv.org/pdf/2211.06393).

    :param cosmo: DiscoDJ cosmology object
    :param k: wavenumber in h / Mpc
    """
    omegamhh = cosmo.Omega_m * cosmo.h ** 2
    omegabhh = cosmo.Omega_b * cosmo.h ** 2
    x = k / (omegamhh - omegabhh)
    return (1 + 59.0998 * x ** 1.49177 + 4658.01 * x ** 4.02755 + 3170.79 * x ** 6.06 + 150.089 * x ** 7.28478) ** -0.25


transfer_dict = {"DiscoEB": disco_eb,
                 "Eisenstein-Hu": eisenstein_hu,
                 "BBKS": bbks,
                 "GeneticAlgorithm": genetic_algorithm}
