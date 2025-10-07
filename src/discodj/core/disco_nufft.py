import jax
import jax.numpy as jnp
import numpy as onp
from functools import partial
from ..core.types import AnyArray

__all__ = ["disconufft3d1", "disconufft3d2", "disconufft3d1r", "disconufft3d2r"]


def get_legendre_q(p: int, with_jax: bool = True) -> AnyArray:
    """Gauss-Legendre quadrature points for p = 8, 10, 12, 14, 16, 18, 20, 32, 64
    integral from -1 to 1 of f(x) dx = sum(w_i * f(q_i))

    :param p: number of quadrature points
    :param with_jax: use Jax?
    :return: quadrature points
    """

    np = jnp if with_jax else onp
    assert p in [8, 10, 12, 14, 16, 18, 20, 32, 64], "p must be in [8, 10, 12, 14, 16, 18, 20, 32, 64]"

    if p == 8:
        return np.array([
            0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975362])
    elif p == 10:
        return np.array([
            0.1488743389816312, 0.4333953941292473, 0.6794095682990244,
            0.8650633666889845, 0.9739065285171717])
    elif p == 12:
        return np.array([
            0.1252334085114689, 0.3678314989981802, 0.5873179542866174,
            0.7699026741943046, 0.9041172563704748, 0.9815606342467192])
    elif p == 14:
        return np.array([
            0.1080549487073437, 0.3191123689278897, 0.5152486363581540, 0.6872929048116854,
            0.8272013150697649, 0.9284348836635734, 0.9862838086968123])
    elif p == 16:
        return np.array([
            0.09501250983763744, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
            0.7554044083550031, 0.8656312023878319, 0.9445750230732326, 0.9894009349916499])
    elif p == 18:
        return np.array([
            0.08477501304173531, 0.2518862256915055, 0.4117511614628427,
            0.5597708310739477, 0.6916870430603532, 0.8037049589725231,
            0.8926024664975558, 0.9558239495713978, 0.991565168420931])
    elif p == 20:
        return np.array([
            0.07652652113349734, 0.2277858511416451, 0.3737060887154195, 0.510867001950827,
            0.6360536807265149, 0.7463319064601507, 0.8391169718222188, 0.9122344282513258,
            0.9639719272779137, 0.9931285991850949])
    elif p == 32:
        return np.array([
            0.04830766568773832, 0.1444719615827965, 0.2392873622521371, 0.3318686022821276,
            0.4213512761306353, 0.5068999089322294, 0.5877157572407623, 0.6630442669302152,
            0.7321821187402897, 0.7944837959679424, 0.84936761373257, 0.8963211557660521,
            0.9349060759377397, 0.9647622555875064, 0.9856115115452683, 0.9972638618494816])
    elif p == 64:
        return np.array([
            0.02435029266342443, 0.07299312178779903, 0.1214628192961205, 0.1696444204239928,
            0.2174236437400071, 0.2646871622087674, 0.3113228719902109, 0.3572201583376681,
            0.4022701579639916, 0.446366017253464, 0.4894031457070529, 0.5312794640198945,
            0.5718956462026339, 0.6111553551723931, 0.6489654712546571, 0.685236313054233,
            0.7198818501716107, 0.7528199072605317, 0.7839723589433413, 0.8132653151227974,
            0.8406292962525802, 0.8659993981540927, 0.8893154459951139, 0.9105221370785026,
            0.9295691721319393, 0.9464113748584027, 0.9610087996520535, 0.9733268277899109,
            0.9833362538846259, 0.9910133714767443, 0.9963401167719552, 0.9993050417357721])


def get_legendre_w(p: int, with_jax: bool = True) -> AnyArray:
    """Gauss-Legendre quadrature weights for p = 8, 10, 12, 14, 16, 18, 20, 32, 64
    integral from -1 to 1 of f(x) dx = sum(w_i * f(q_i))

    :param p: number of quadrature points
    :param with_jax: use Jax?
    :return: quadrature weights
    """
    np = jnp if with_jax else onp
    assert p in [8, 10, 12, 14, 16, 18, 20, 32, 64], "p must be in [8, 10, 12, 14, 16, 18, 20, 32, 64]"

    if p == 8:
        return np.array([
            0.3626837833783623, 0.3137066458778873, 0.2223810344533743, 0.1012285362903761])
    if p == 10:
        return np.array([
            0.2955242247147530, 0.2692667193099963, 0.2190863625159819,
            0.1494513491505805, 0.06667134430868821])
    if p == 12:
        return np.array([
            0.2491470458134024, 0.2334925365383549, 0.2031674267230659,
            0.1600783285433462, 0.1069393259953185, 0.04717533638651189])
    if p == 14:
        return np.array([
            0.2152638534631582, 0.2051984637212959, 0.1855383974779377, 0.1572031671581933,
            0.1215185706879029, 0.08015808715976, 0.03511946033175185])
    elif p == 16:
        return np.array([
            0.1894506104550682, 0.1826034150449233, 0.1691565193950024, 0.1495959888165768,
            0.1246289712555341, 0.09515851168249308, 0.06225352393864799, 0.02715245941175415])
    elif p == 18:
        return np.array([
            0.1691423829631435, 0.1642764837458326, 0.1546846751262653, 0.1406429146706507,
            0.1225552067114785, 0.1009420441062871, 0.07642573025488909, 0.04971454889496983,
            0.02161601352648341])
    elif p == 20:
        return np.array([
            0.1527533871307259, 0.1491729864726037, 0.1420961093183822, 0.1316886384491767,
            0.1181945319615183, 0.1019301198172403, 0.08327674157670469, 0.06267204833410898,
            0.04060142980038687, 0.01761400713915211])
    elif p == 32:
        return np.array([
            0.09654008851472781, 0.09563872007927493, 0.09384439908080459, 0.09117387869576378,
            0.08765209300440373, 0.08331192422694676, 0.07819389578707026, 0.07234579410884845,
            0.06582222277636188, 0.05868409347853557, 0.05099805926237616, 0.04283589802222668,
            0.03427386291302142, 0.02539206530926202, 0.01627439473090559, 0.007018610009470164])
    elif p == 64:
        return np.array([
            0.04869095700913986, 0.04857546744150359, 0.0483447622348031, 0.04799938859645844,
            0.04754016571483039, 0.04696818281621007, 0.04628479658131444, 0.0454916279274182,
            0.04459055816375661, 0.04358372452932343, 0.04247351512365358, 0.04126256324262351,
            0.03995374113272034, 0.03855015317861565, 0.03705512854024006, 0.03547221325688241,
            0.03380516183714156, 0.03205792835485149, 0.03023465707240244, 0.02833967261425945,
            0.0263774697150546, 0.02435270256871081, 0.02227017380838316, 0.02013482315353012,
            0.01795171577569729, 0.01572603047602468, 0.01346304789671859, 0.01116813946013111,
            0.008846759826363923, 0.006504457968978387, 0.004147033260562449, 0.001783280721696303])


def phi(z: AnyArray | float, beta: float, with_jax: bool = True) -> AnyArray:
    """Normalized kernel function 'exponential of semicircle' for NUFFT, see https://arxiv.org/abs/1808.06736 (B18).
    `phi(z) = exp(beta * (sqrt(1-z^2) - 1))` for `|z| <= 1`, and 0 otherwise.

    :param z: input
    :param beta: kernel parameter
    :param with_jax: use Jax?
    :return: kernel values
    """
    np = jnp if with_jax else onp
    return np.where(np.abs(z) <= 1.0, np.exp(beta * (np.sqrt(1 - z ** 2) - 1)), 0.0)  # B18 eq. (1.8)


def grad_phi(z: AnyArray | float, beta: float, with_jax: bool = True) -> AnyArray:
    """Gradient of the normalized kernel function 'exponential of semicircle' for NUFFT.
    `grad_phi(z) = -beta * z * exp(beta * (sqrt(1-z^2) - 1)) / sqrt(1-z^2)` for `|z| <= 1`, and 0 otherwise.

    :param z: input
    :param beta: kernel parameter
    :param with_jax: use Jax?
    :return: gradient of the kernel values
    """
    np = jnp if with_jax else onp
    eps = np.finfo(z.dtype).eps
    return np.where(np.abs(z) <= 1.0 - eps,
                    -beta * z * phi(z, beta, with_jax) / np.sqrt(1 - z ** 2),
                    0.0)


def psi(*, x: AnyArray | float, alpha: float, beta: float, with_jax: bool = True) -> AnyArray:
    """Normalized kernel function 'exponential of semicircle' for NUFFT with non-unit support alpha
    `psi(x) = phi(x/alpha)` for `|x| <= alpha`, and 0 otherwise.
    :param x: input
    :param alpha: kernel support
    :param beta: kernel parameter
    :param with_jax: use Jax?
    :return kernel values
    """
    np = jnp if with_jax else onp
    x = np.where(x < -np.pi / 2, x + np.pi, np.where(x > np.pi / 2, x - np.pi, x))
    return phi(x / alpha, beta)  # B18 eq. (3.4)


def grad_psi(*, x: AnyArray | float, alpha: float, beta: float, with_jax: bool = True) -> AnyArray:
    """Gradient of the normalized kernel function 'exponential of semicircle' for NUFFT with non-unit support alpha
    `grad_psi(x) = grad_phi(x/alpha) / alpha` for `|x| <= alpha`, and 0 otherwise.
    :param x: input
    :param alpha: kernel support
    :param beta: kernel parameter
    :param with_jax: use Jax?
    :return gradient of the kernel values
    """
    np = jnp if with_jax else onp
    x = np.where(x < -np.pi / 2, x + np.pi, np.where(x > np.pi / 2, x - np.pi, x))
    return grad_phi(x / alpha, beta) / alpha


def psihat(*, k: AnyArray | float, w: int, alpha: float, beta: float, sigma: float | None = None, p: int | None = None,
           with_jax: bool = True) -> AnyArray:
    """Compute the Fourier transform of psi(x) = phi(x/alpha) using Gauss-Legendre quadrature of order p.

    :param k: Fourier modes
    :param w: kernel width
    :param alpha: kernel support
    :param beta: kernel parameter
    :param sigma: oversampling ratio
    :param p: number of quadrature points
    :param with_jax: use Jax?
    :return: Fourier transform of psi(x)
    """
    np = jnp if with_jax else onp

    # if p is None:
    # p = int(1.5*w+2)
    # p += p%2 # make p even

    # NOTE: sigma and p are not used for now, and p is hard-coded here!
    p = 64

    q = get_legendre_q(p)
    f = get_legendre_w(p) * phi(q, beta)
    if with_jax:
        ph = jax.vmap(lambda kk: jnp.sum(f * jnp.cos(alpha * kk * q)))(k)
    else:
        ph = np.array([np.sum(f * np.cos(alpha * kk * q)) for kk in k])
    return w * ph


def spread_to_grid_3d(*, x: AnyArray, f: AnyArray, w: int, ng: int, alpha: float, beta: float,
                      chunk_size: int | None = None, with_jax: bool = True) -> AnyArray:
    """Interpolation of f(x) to the non-uniform grid x = (0, 2*pi) using the kernel psi(x).

    :param x: non-uniform grid
    :param f: function values at the non-uniform grid
    :param w: kernel width
    :param ng: number of grid points
    :param alpha: kernel support
    :param beta: kernel parameter
    :param chunk_size: chunk size of particles that are scattered at once
    :param with_jax: use Jax?
    :return: interpolated function values at the grid points
    """
    np = jnp if with_jax else onp
    h = 2 * np.pi / ng  # B18 eq. (3.3)
    dtype_num = np.finfo(f.dtype).bits
    dtype_int = getattr(jnp, f"int{dtype_num}")
    dtype_float = getattr(jnp, f"float{dtype_num}")
    y = jnp.zeros((ng * ng * ng), dtype=f.dtype)

    if with_jax:
        def scatter_chunk(carry, chunk):
            xx, yy, zz, ff = chunk[..., 0], chunk[..., 1], chunk[..., 2], chunk[..., 3]
            ix = (xx / h).astype(jnp.int32)
            iy = (yy / h).astype(jnp.int32)
            iz = (zz / h).astype(jnp.int32)
            idx_range = jnp.arange(-w // 2, w // 2 + 1, dtype=dtype_int)
            i_indices = (ix[:, None] + idx_range[None] + ng) % ng
            j_indices = (iy[:, None] + idx_range[None] + ng) % ng
            k_indices = (iz[:, None] + idx_range[None] + ng) % ng
            indices = (((i_indices[:, :, None, None] * ng) + j_indices[:, None, :, None]) * ng
                       + k_indices[:, None, None, :]).flatten().astype(dtype_int)
            values_x = psi(x=(ix[:, None] + idx_range[None]) * h - xx[:, None], alpha=alpha, beta=beta)
            values_y = psi(x=(iy[:, None] + idx_range[None]) * h - yy[:, None], alpha=alpha, beta=beta)
            values_z = psi(x=(iz[:, None] + idx_range[None]) * h - zz[:, None], alpha=alpha, beta=beta)

            dim = 1
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=tuple(range(dim)),
                scatter_dims_to_operand_dims=tuple(range(dim)))
            indices = indices.reshape([-1, dim])
            ff_ext = ff[:, None, None, None]
            values = (ff_ext * (values_x[:, :, None, None] * values_y[:, None, :, None] * values_z[:, None, None, :]
                                )).flatten().astype(dtype_float)
            carry = jax.lax.scatter_add(carry, indices, values, dnums, mode="drop")
            return carry, None

        data = jnp.column_stack((x[..., 0], x[..., 1], x[..., 2], f)).reshape(-1, 4)
        if chunk_size is not None:
            assert chunk_size > 0 and chunk_size <= x.shape[0], \
                "chunk_size must be positive and less than the number of particles"
            assert data.shape[0] % chunk_size == 0, "Data length must be divisible by chunk_size"
        chunk_size = chunk_size or data.shape[0]
        num_chunks = data.shape[0] // chunk_size if chunk_size is not None else 1
        chunks = jnp.transpose(data.reshape(chunk_size, num_chunks, 4), (1, 0, 2))
        y, _ = jax.lax.scan(scatter_chunk, y, chunks)

    else:
        for xx, ff in zip(x, f):
            ix = int(xx / h)
            for i in range(-w // 2, w // 2 + 1):
                y[(ix + i + ng) % ng] += ff * psi(x=(ix + i) * h - xx, alpha=alpha, beta=beta)  # B18 eq. (3.6)

    return y.reshape((ng, ng, ng))


def interp_from_grid_3d(*, x: AnyArray, f: AnyArray, w: int, ng: int, alpha: float, beta: float,
                        chunk_size: int | None = None, with_jax: bool = True, grad: bool = False) -> AnyArray:
    """Interpolation of f(x) to the non-uniform grid x = (0, 2*pi) using the kernel psi(x).

    :param x: non-uniform grid
    :param f: function values at the non-uniform grid
    :param w: kernel width
    :param ng: number of grid points
    :param alpha: kernel support
    :param beta: kernel parameter
    :param chunk_size: chunk size of particles that are gathered at once
    :param with_jax: use Jax?
    :param grad: compute gradient?
    :return: interpolated function values at the non-uniform grid
    """
    np = jnp if with_jax else onp
    h = 2 * np.pi / ng  # B18 eq. (3.3)
    f = f.flatten()
    dtype_num = np.finfo(f.dtype).bits
    dtype_int = getattr(jnp, f"int{dtype_num}")
    dtype_float = getattr(jnp, f"float{dtype_num}")
    if grad:
        assert with_jax, "Kernel gradient is only implemented for the Jax version"

    if with_jax:
        def gather_chunk(carry, chunk):
            xx, yy, zz = chunk[..., 0], chunk[..., 1], chunk[..., 2]
            ix = (xx / h).astype(jnp.int32)
            iy = (yy / h).astype(jnp.int32)
            iz = (zz / h).astype(jnp.int32)
            idx_range = jnp.arange(-w // 2, w // 2 + 1)
            i_indices = (ix[:, None] + idx_range[None] + ng) % ng
            j_indices = (iy[:, None] + idx_range[None] + ng) % ng
            k_indices = (iz[:, None] + idx_range[None] + ng) % ng
            indices = (((i_indices[:, :, None, None] * ng) + j_indices[:, None, :, None]) * ng
                       + k_indices[:, None, None, :]).flatten().astype(dtype_int)
            kernel_x = psi(x=(ix[:, None] + idx_range[None]) * h - xx[:, None], alpha=alpha, beta=beta)
            kernel_y = psi(x=(iy[:, None] + idx_range[None]) * h - yy[:, None], alpha=alpha, beta=beta)
            kernel_z = psi(x=(iz[:, None] + idx_range[None]) * h - zz[:, None], alpha=alpha, beta=beta)
            if grad:
                kernel_x_grad = -grad_psi(x=(ix[:, None] + idx_range[None]) * h - xx[:, None], alpha=alpha, beta=beta)
                kernel_y_grad = -grad_psi(x=(iy[:, None] + idx_range[None]) * h - yy[:, None], alpha=alpha, beta=beta)
                kernel_z_grad = -grad_psi(x=(iz[:, None] + idx_range[None]) * h - zz[:, None], alpha=alpha, beta=beta)
                kernel = jnp.asarray(
                    [kernel_x_grad[:, :, None, None] * kernel_y[:, None, :, None] * kernel_z[:, None, None, :],
                     kernel_x[:, :, None, None] * kernel_y_grad[:, None, :, None] * kernel_z[:, None, None, :],
                     kernel_x[:, :, None, None] * kernel_y[:, None, :, None] * kernel_z_grad[:, None, None, :]]
                ).reshape(3, -1).astype(dtype_float)
                y_at_x = (carry[indices][None] * kernel).reshape(3, -1, (w + 1) ** 3).sum(-1)
            else:
                kernel = (kernel_x[:, :, None, None] * kernel_y[:, None, :, None]
                          * kernel_z[:, None, None, :]).flatten().astype(dtype_float)
                y_at_x = (carry[indices] * kernel).reshape(-1, (w + 1) ** 3).sum(-1)
            return carry, y_at_x

        data = jnp.column_stack((x[..., 0], x[..., 1], x[..., 2])).reshape(-1, 3)
        if chunk_size is not None:
            assert chunk_size > 0 and chunk_size <= x.shape[0], \
                "chunk_size must be positive and less than the number of particles"
            assert x.shape[0] % chunk_size == 0, "Data length must be divisible by chunk_size"
        chunk_size = chunk_size or x.shape[0]
        num_chunks = x.shape[0] // chunk_size if chunk_size is not None else 1
        chunks = jnp.transpose(data.reshape(chunk_size, num_chunks, 3), (1, 0, 2))
        if grad:
            y_at_x_raw = jnp.transpose(jax.lax.scan(gather_chunk, init=f, xs=chunks)[1], (1, 0, 2))
            y_at_x = (
                jax.vmap(lambda v: v.flatten(order="F"))(y_at_x_raw)).T  # order F: transpose above, T -> (ng ** 3, 3)
        else:
            y_at_x = jax.lax.scan(gather_chunk, init=f, xs=chunks)[1].flatten(order="F")  # order F: transpose above

    else:
        y_at_x = np.zeros_like(x)
        for j, xx in enumerate(x):
            ix = int(xx / h)
            for i in range(-w // 2, w // 2 + 1):
                y_at_x[j] += f[(ix + i + ng) % ng] * psi(x=(ix + i) * h - xx, alpha=alpha, beta=beta)  # B18 eq. (3.16)

    return y_at_x


@partial(jax.jit, static_argnames=('N', 'eps', 'sigma', 'chunk_size', 'with_jax'))
def disconufft3d1(x: AnyArray, f: AnyArray, N: int, eps: float = 1e-6, sigma: float = 1.25,
                  chunk_size: int | None = None, with_jax: bool = True) -> AnyArray:
    """1D type 1 complex real non-uniform FFT using the kernel 'exponential of semicircle'.

    :param x: non-uniform grid
    :param f: function values at x
    :param N: number of uniform grid points in Fourier domain
    :param eps: desired accuracy
    :param sigma: oversampling ratio
    :param chunk_size: chunk size of particles that are scattered at once
    :param with_jax: use Jax?
    :return: Fourier coefficients at the uniform grid points
    """
    assert with_jax, "Only Jax is supported for now"
    np = jnp if with_jax else onp
    gamma = 0.976
    w = int(onp.log(1 / eps) / (onp.pi * gamma * onp.sqrt(1 - 1 / sigma - (gamma ** 2 - 1) / (4 * sigma ** 2))) + 1)
    w += w % 2  # make w even

    ng = onp.maximum(int(sigma * N), 2 * w + 2)  # number of uniform grid points in Fourier domain after upsampling
    h = 2 * np.pi / ng  # B18 eq. (3.3)
    alpha = w * h / 2  # B18 eq. (3.3)
    beta = w * gamma * np.pi * (1 - 1 / (2 * sigma))  # B18 eq. (4.5)

    # step 1: interpolation
    y = spread_to_grid_3d(x=x, f=f, alpha=alpha, beta=beta, ng=ng, w=w, chunk_size=chunk_size, with_jax=with_jax)

    # step 2: FFT
    fy = np.fft.fftshift(np.fft.fftn(y))

    # step 3: deconvolution
    k = np.arange(-ng // 2, ng // 2)
    ph = psihat(k=k, alpha=alpha, beta=beta, sigma=sigma, w=w)
    fy = fy / (ph[:, None, None] * ph[None, :, None] * ph[None, None, :])

    # step 4: truncation
    return jax.lax.slice(fy, [ng // 2 - N // 2, ng // 2 - N // 2, ng // 2 - N // 2],
                         [ng // 2 + N // 2, ng // 2 + N // 2, ng // 2 + N // 2])


@partial(jax.jit, static_argnames=('eps', 'sigma', 'chunk_size', 'with_jax', 'grad'))
def disconufft3d2(x: AnyArray, ft: AnyArray, eps: float = 1e-6, sigma: float = 1.25,
                  chunk_size: bool | None = None, with_jax: bool = True, grad: bool = False) -> AnyArray:
    """1D type 2 complex non-uniform FFT using the kernel 'exponential of semicircle'.

    :param x: non-uniform grid
    :param ft: Fourier coefficients at the uniform grid points
    :param eps: desired accuracy
    :param sigma: oversampling ratio
    :param chunk_size: chunk size of particles that are gathered at once
    :param with_jax: use Jax?
    :param grad: compute gradient of the kernel instead of the kernel itself?
    :return: function values at the non-uniform grid
    """
    assert with_jax, "Only Jax is supported for now"
    np = jnp if with_jax else onp
    N = ft.shape[0]
    gamma = 0.976
    w = int(onp.log(1 / eps) / (onp.pi * gamma * onp.sqrt(1 - 1 / sigma - (gamma ** 2 - 1) / (4 * sigma ** 2))) + 1)
    # w = int(np.log10(1/eps)+1)+1  # B18 eq. (3.2)
    w += w % 2  # make w even

    ng = onp.maximum(int(sigma * N), 2 * w + 2)  # number of uniform grid points in Fourier domain after upsampling
    h = 2 * np.pi / ng  # B18 eq. (3.3)
    alpha = w * h / 2  # B18 eq. (3.3)
    beta = gamma * np.pi * w * (1 - 1 / (2 * sigma))  # B18 eq. (4.5)

    # step 1: correction
    k = np.arange(-N // 2, N // 2)
    ph = psihat(k=k, alpha=alpha, beta=beta, sigma=sigma, w=w, with_jax=with_jax)
    ft = ft / (ph[:, None, None] * ph[None, :, None] * ph[None, None, :])

    # zero pad from N to ng:
    ft = np.pad(ft, ng // 2 - N // 2)

    # step 2: FFT
    y = np.fft.ifftn(np.fft.ifftshift(ft))

    # step 3: interpolation
    f = interp_from_grid_3d(x=x, f=y, alpha=alpha, beta=beta, ng=ng, w=w, chunk_size=chunk_size, with_jax=with_jax,
                            grad=grad)

    return f * ng ** 3


@partial(jax.jit, static_argnames=('N', 'eps', 'sigma', 'chunk_size', 'with_jax'))
def disconufft3d1r(x: AnyArray, f: AnyArray, N: int, eps: float = 1e-6, sigma: float = 1.25,
                   chunk_size: int | None = None, with_jax: bool = True) -> AnyArray:
    """1D type 1 real non-uniform FFT using the kernel 'exponential of semicircle'.

    :param x: non-uniform grid
    :param f: function values at x
    :param N: number of uniform grid points in Fourier domain
    :param eps: desired accuracy
    :param sigma: oversampling ratio
    :param chunk_size: chunk size of particles that are scattered at once
    :param with_jax: use Jax?
    :return: Fourier coefficients at the uniform grid points
    """
    assert with_jax, "Only Jax is supported for now"
    np = jnp if with_jax else onp
    gamma = 0.976
    w = int(onp.log(1 / eps) / (onp.pi * gamma * onp.sqrt(1 - 1 / sigma - (gamma ** 2 - 1) / (4 * sigma ** 2))) + 1)
    w += w % 2  # make w even

    ng = onp.maximum(int(sigma * N), 2 * w + 2)  # number of uniform grid points in Fourier domain after upsampling
    h = 2 * np.pi / ng  # B18 eq. (3.3)
    alpha = w * h / 2  # B18 eq. (3.3)
    beta = w * gamma * np.pi * (1 - 1 / (2 * sigma))  # B18 eq. (4.5)

    # step 1: interpolation
    y = spread_to_grid_3d(x=x, f=f, alpha=alpha, beta=beta, ng=ng, w=w, chunk_size=chunk_size, with_jax=with_jax)

    # step 2: FFT
    fy = np.fft.fftshift(np.fft.rfftn(y), axes=(0, 1))

    # step 3: truncation
    fy = jax.lax.slice(fy, [ng // 2 - N // 2, ng // 2 - N // 2, 0],
                       [ng // 2 + N // 2 + 1, ng // 2 + N // 2 + 1, N // 2 + 1])

    # step 4: deconvolution
    ph = psihat(k=np.arange(-N // 2, N // 2 + 1), alpha=alpha, beta=beta, sigma=sigma, w=w, with_jax=with_jax)
    ph_half = psihat(k=np.arange(0, N // 2 + 1), alpha=alpha, beta=beta, sigma=sigma, w=w, with_jax=with_jax)
    fy = fy / (ph[:, None, None] * ph[None, :, None] * ph_half[None, None, :])

    return fy


@partial(jax.jit, static_argnames=('eps', 'sigma', 'chunk_size', 'with_jax', 'grad'))
def disconufft3d2r(x: AnyArray, ft: AnyArray, eps: float = 1e-6, sigma: float = 1.25, chunk_size: int | None = None,
                   with_jax: bool = True, grad: bool = False) -> AnyArray:
    """1D type 2 real non-uniform FFT using the kernel 'exponential of semicircle'.

    :param x: non-uniform grid
    :param ft: Fourier coefficients at the uniform grid points
    :param eps: desired accuracy
    :param sigma: oversampling ratio
    :param chunk_size: chunk size of particles that are gathered at once
    :param with_jax: use Jax?
    :param grad: compute gradient of the kernel instead of the kernel itself?
    :return: function values at the non-uniform grid
    """
    assert with_jax, "Only Jax is supported for now"
    np = jnp if with_jax else onp
    N = ft.shape[0] - 1
    gamma = 0.976
    w = int(onp.log(1 / eps) / (onp.pi * gamma * onp.sqrt(1 - 1 / sigma - (gamma ** 2 - 1) / (4 * sigma ** 2))) + 1)
    # w = int(np.log10(1/eps)+1)+1  # B18 eq. (3.2)
    w += w % 2  # make w even

    ng = onp.maximum(int(sigma * N), 2 * w + 2)  # number of uniform grid points in Fourier domain after upsampling
    h = 2 * np.pi / ng  # B18 eq. (3.3)
    alpha = w * h / 2  # B18 eq. (3.3)
    beta = gamma * np.pi * w * (1 - 1 / (2 * sigma))  # B18 eq. (4.5)

    # step 1: deconvolution
    ph = psihat(k=np.arange(-N // 2, N // 2 + 1), alpha=alpha, beta=beta, sigma=sigma, w=w, with_jax=with_jax)
    ph_half = psihat(k=np.arange(0, N // 2 + 1), alpha=alpha, beta=beta, sigma=sigma, w=w, with_jax=with_jax)
    ft = ft / (ph[:, None, None] * ph[None, :, None] * ph_half[None, None, :])

    # step 2: zero pad from N to ng:
    ft = np.pad(ft, (
        (ng // 2 - N // 2, ng // 2 - N // 2 - 1), (ng // 2 - N // 2, ng // 2 - N // 2 - 1), (0, ng // 2 - N // 2)))

    # step 3: iFFT
    y = np.fft.irfftn(np.fft.ifftshift(ft, axes=(0, 1)))

    # step 4: interpolation
    f = interp_from_grid_3d(x=x, f=y, alpha=alpha, beta=beta, ng=ng, w=w, chunk_size=chunk_size, with_jax=with_jax,
                            grad=grad)

    return f * ng ** 3
