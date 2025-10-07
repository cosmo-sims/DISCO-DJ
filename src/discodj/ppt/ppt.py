import jax.numpy as jnp
import jax
import numpy as np
from jax import Array
from einops import rearrange
from ..core.grids import get_fourier_grid
from ..cosmology.cosmology import Cosmology
from ..core.utils import set_0_to_val, pad, crop, conv2_fourier

__all__ = ["evaluate_ppt"]


def evaluate_ppt(cosmo: Cosmology, boxsize: float, phi_ini: Array, a_end: float | Array, h_bar: float | None,
                 do_rsd: bool, lyman_alpha_dict: dict | None, propagator: str) -> dict:
    """Evaluate the propagator perturbation theory (PPT) fields at the given scale factor.

    :param cosmo: cosmology object
    :param boxsize: the size of the box in Mpc/h
    :param phi_ini: the initial potential field
    :param a_end: the scale factor at which to evaluate the PPT fields
    :param h_bar: the value of h_bar to use in the PPT computation. If None, we use the value of h_bar proposed in
        Eq. (24) in arXiv:2009.09124 with a safety margin of 10%
    :param do_rsd: whether to include redshift-space distortions in the PPT computation (along the last dimension)
    :param lyman_alpha_dict: a dictionary containing the parameters for the Lyman-alpha forest. If None, we do not
        include the Lyman-alpha forest in the PPT computation
    :param propagator: the propagator to use in the PPT computation. Can be either "free" or "kdk" (kick-drift-kick)
    :return: a dictionary containing the PPT fields
    """
    # Get full k matrix for complex FFT
    dtype = phi_ini.dtype
    dtype_num = np.finfo(dtype).bits
    dtype_c = getattr(jnp, f"complex{2 * dtype_num}")
    res = phi_ini.shape[0]
    assert all([res == phi_ini.shape[i] for i in range(1, len(phi_ini.shape))]), \
        "Only cubic boxes are supported for now"
    dim = phi_ini.ndim
    k_dict_full = get_fourier_grid(phi_ini.shape, boxsize=boxsize, sparse_k_vecs=True, full=True, relative=False,
                                   dtype_num=dtype_num, with_jax=False)
    k_full = k_dict_full["|k|"]
    k_full_constant_mode_one = set_0_to_val(dim=dim, t=k_full, v=1.0)
    k_vecs_full = k_dict_full["k_vecs"]

    # Determine growth factor at a_end
    Dplus_end = cosmo.Dplus(a_end)

    # Determine h_bar according to Eq. (24) in arXiv:2008.09124
    if h_bar is None:
        phi_ini_delta_max = jnp.max(jnp.asarray([
            jnp.max(jnp.abs(jnp.roll(phi_ini, shift=1, axis=ax) - phi_ini))
            for ax in range(dim)]))
        h_bar = 1.1 / jnp.pi * phi_ini_delta_max  # 1.1: safety factor

    # Fourier transform of initial wave function (Eq. (4) in arXiv:2005.12928)
    fpsi_0 = jnp.fft.fftn(jnp.exp(-1j / h_bar * phi_ini))

    # Compute wave function at final time with Eq. (5)
    if propagator == "free":
        fprop = jnp.exp((-1j / 2) * (k_full ** 2) * h_bar * Dplus_end)
        fpsi = fprop * fpsi_0
        psi = jnp.fft.ifftn(fpsi)

        # Compute pi = rho v
        # grad_psi = jnp.stack([jnp.fft.ifftn(1j * k * fpsi) for k in self.k_dict_full["kks"]], -1)
        # pi = 1j * h_bar / 2.0 * (psi[..., None] * 1j * grad_psi.conj() - psi[..., None].conj() * 1j * grad_psi)

    elif propagator == "kdk":
        # Get 2LPT potential
        D2phi_f = rearrange(jnp.asarray([[-k1 * k2 * jnp.fft.fftn(phi_ini)
                                          for k2 in k_vecs_full] for k1 in k_vecs_full]),
                            "d1 d2 ... -> ... d2 d1").astype(dtype)
        a = {"orig_res": res, "ext_res": (3 * res) // 2, "dtype_c": dtype_c, "with_jax": True, "full": True}
        if dim > 1:
            D2phi_f_padded = rearrange(
                jax.vmap(lambda ff: pad(dim=dim, ff=ff, **a), in_axes=-1, out_axes=-1)(
                    rearrange(D2phi_f, "... d1 d2 -> ... (d1 d2)")),
                "... (d1 d2) -> ... d1 d2", d1=dim, d2=dim)
            a = {**a, "do_crop": True}
        if dim == 1:
            pot_2lpt_f = D2phi_f[:, 0, 0]
        elif dim == 2:
            pot_2lpt_f = conv2_fourier(2, D2phi_f_padded[:, :, 0, 0], D2phi_f_padded[:, :, 1, 1], **a) \
                         - conv2_fourier(2, D2phi_f_padded[:, :, 0, 1], D2phi_f_padded[:, :, 1, 0], **a)
        elif dim == 3:
            pot_2lpt_f = conv2_fourier(3, D2phi_f_padded[:, :, :, 0, 0], D2phi_f_padded[:, :, :, 1, 1], **a) \
                         - conv2_fourier(3, D2phi_f_padded[:, :, :, 0, 1], D2phi_f_padded[:, :, :, 1, 0], **a) \
                         + conv2_fourier(3, D2phi_f_padded[:, :, :, 0, 0], D2phi_f_padded[:, :, :, 2, 2], **a) \
                         - conv2_fourier(3, D2phi_f_padded[:, :, :, 0, 2], D2phi_f_padded[:, :, :, 2, 0], **a) \
                         + conv2_fourier(3, D2phi_f_padded[:, :, :, 1, 1], D2phi_f_padded[:, :, :, 2, 2], **a) \
                         - conv2_fourier(3, D2phi_f_padded[:, :, :, 1, 2], D2phi_f_padded[:, :, :, 2, 1], **a)
            # D2phi = jax.vmap(lambda ff: jnp.fft.ifftn(ff), in_axes=-1, out_axes=-1)(
            #     D2phi_f.reshape(res, res, res, -1)).reshape(res, res, res, dim, dim)
        else:
            raise NotImplementedError

        pot_2lpt = -6.0 / 7.0 * jnp.fft.ifftn(pot_2lpt_f / k_full_constant_mode_one ** 2)

        # Kick 1
        psi_kick = jnp.exp(-1j / h_bar * (phi_ini + Dplus_end / 2.0 * pot_2lpt))

        # Drift
        fprop = jnp.exp((-1j / 2) * (k_full ** 2) * h_bar * Dplus_end)
        fpsi_drift = fprop * jnp.fft.fftn(psi_kick)
        psi_drift = jnp.fft.ifftn(fpsi_drift)

        # Kick again
        psi = jnp.exp(-1j / h_bar * Dplus_end / 2.0 * pot_2lpt) * psi_drift
        fpsi = jnp.fft.fftn(psi)

    else:
        raise NotImplementedError

    # Compute density at final time
    rho = jnp.abs(psi) ** 2

    # Compute momentum density
    psi_padded = jnp.fft.ifftn(pad(dim=dim, orig_res=res, ext_res=3 * (res // 2), ff=fpsi, dtype_c=dtype_c, full=True,
                                   with_jax=True))
    grad_psi_padded = jnp.stack([jnp.fft.ifftn(
        pad(dim=dim, orig_res=res, ext_res=3 * (res // 2), ff=1j * k * fpsi, dtype_c=dtype_c, full=True, with_jax=True))
        for k in k_dict_full["k_vecs"]], -1)  # TODO: dealias
    j_padded = ((1j * h_bar / 2)
                * (psi_padded[..., None] * grad_psi_padded.conj() - psi_padded.conj()[..., None] * grad_psi_padded)).real
    j = jax.vmap(lambda j_p: jnp.fft.ifftn(
        crop(dim=dim, orig_res=res, ext_res=3 * (res // 2), ff=jnp.fft.fftn(j_p), dtype_c=dtype_c, full=True,
             with_jax=True)), in_axes=-1, out_axes=-1)(j_padded).real

    # Same for redshift space distortions (Eqs. (8) - (9)): RSDs always along last dimension!
    if do_rsd:
        k_last = k_dict_full["k_vecs"][-1]
        f_end = cosmo.growth_rate(a_end)
        fprop_rsd = jnp.exp(-1j / 2 * k_last ** 2 * h_bar * f_end * Dplus_end)
        psi_rsd = jnp.fft.ifftn(fprop_rsd * fprop * fpsi_0)
        rho_rsd = jnp.abs(psi_rsd) ** 2

    out_dict = {"delta": rho - 1.0, "j": j}

    if do_rsd:
        out_dict["delta_rsd"] = rho_rsd - 1.0

    # Now: Lyman-alpha flux with fluctuating Gunn-Peterson?
    if lyman_alpha_dict is not None:
        A = lyman_alpha_dict["A"]
        beta = lyman_alpha_dict["beta"]

        chi = jnp.sqrt((A * rho ** beta) / rho) * psi  # Eq. (12)
        tau = jnp.abs(chi) ** 2
        out_dict["tau"] = tau

        if do_rsd:
            chi_rsd = jnp.fft.ifftn(fprop_rsd * jnp.fft.fftn(chi))
            tau_rsd = jnp.abs(chi_rsd) ** 2
            out_dict["tau_rsd"] = tau_rsd

    out_dict["h_bar"] = h_bar

    return out_dict
