import warnings
import jax
import jax.numpy as jnp
from jax import Array
import numpy as onp
from einops import rearrange
from functools import partial
from collections.abc import Callable
import diffrax
from diffrax import ODETerm, diffeqsolve, SaveAt, NoProgressMeter
# from diffrax import TqdmProgressMeter
from .cosmology.cosmology import Cosmology
from .cosmology.cosmo_utils import get_sigma8_squared_from_Pk
from .cosmology.predefined_cosmologies import get_cosmology_dict_from_name
from .cosmology.transfer_functions import transfer_dict
from .core.scatter_and_gather import (scatter, deconvolve_mak, spawn_interpolated_particles, interpolate_field,
                                      compute_field_quantity_from_particles)
from .core.grids import get_fourier_grid
from .core.utils import set_0_to_val, set_indices_to_val, preprocess_fixed_inds
from .core.kernels import inv_laplace_kernel, gradient_kernel
from .core.types import AnyArray
from .core.summary_statistics import power_spectrum, cross_power_spectrum, bispectrum
from .core.io import save_as_hdf5
from .nbody.acc import calc_acc_exact_1d, calc_acc_nufft, calc_acc_PM, calc_acc_Tree_PM
from .nbody.steppers.leapfrog_adjoint import BacksolveAdjointLeapfrog
from .nbody.steppers.dkd_leapfrog import DriftKickDriftAdjoint
from .nbody.steppers.dkd_pi_integrator import DKDPiIntegrator
from .nbody.steppers.dkd_symplectic_integrator import DKDSymplecticIntegrator
from .nbody.nbody_scan_functions import nbody_scan
from .ics.grf import get_ngenic_wnoise, generate_grf, get_k_ordering
from .ics.plane_wave import generate_ics_plane_wave
from .lpt.nlpt_1d import NLPT1
from .lpt.nlpt_2d import NLPT2
from .lpt.nlpt_3d_jax import NLPT3
from .ppt.ppt import evaluate_ppt


__all__ = ["DiscoDJ"]


@jax.tree_util.register_pytree_node_class
class DiscoDJ:
    def __init__(
            self,
            dim: int,
            res: int,
            name: str | None = None,
            device: str | None = None,
            precision: str = "single",
            boxsize: float = 1.0,
            cosmo: dict | str = "Planck18EEBAOSN",
            requires_grad_wrt_cosmo: bool = False,
            requires_jacfwd: bool = False,
    ):
        """A Jax-based cosmological simulation framework.

        :param dim: Dimensionality of the simulation (1, 2, or 3).
        :param res: Number of particles per dimension.
        :param name: Optional name for the DiscoDJ instance.
        :param device: Target device (e.g., 'cpu', 'gpu').
        :param precision: Floating-point precision, either 'single' or 'double'.
            NOTE: 'double' requires setting jax_enable_x64 to True in jax config before importing Jax.
        :param boxsize: Comoving size of the simulation box per dimension in Mpc/h.
        :param cosmo: Cosmological parameters. Either a dictionary or a string with a predefined cosmology.
        :param requires_grad_wrt_cosmo: If gradients w.r.t. the cosmological parameters stored in the DiscoDJ object
            are required (default: False).
        :param requires_jacfwd: If the DiscoDJ instance will be used with jax.jacfwd (default: False).
        """
        # Perform checks
        assert dim in (1, 2, 3), "Only 1D, 2D, and 3D simulations are supported."
        assert precision in ("single", "double"), "Precision must be either 'single' or 'double'."
        assert boxsize > 0, "Boxsize must be positive."
        assert res > 0, "Resolution must be positive."
        assert res % 4 == 0, "Resolution must be a multiple of 4 for now."

        if device is not None:
            try:
                jax.config.update("jax_platform_name", device)
                jax.numpy.zeros(1)  # test if operations work
            except RuntimeError as e:
                raise ValueError(f"Unknown device: {device}. "
                                 f"Please install the GPU version of JAX or use device='cpu'.") from e

        self._dim = dim
        self._res = res
        self._name = name or "DiscoDJ"
        self._device = device
        self._precision = precision
        self._boxsize = 1.0 * boxsize
        self._cosmo_settings = cosmo or {}
        self._res = res
        self._requires_grad_wrt_cosmo = requires_grad_wrt_cosmo
        self._requires_jacfwd = requires_jacfwd

        # Storage (auxiliary data)
        self._pk_table = {}
        self._ics = {}
        self._lpt = None

        # Precision settings
        if precision == "double":
            self.dtype_num = 64
            self.dtype_c_num = 128
            self.dtype = jnp.float64
            self.dtype_c = jnp.complex128
            if not jax.config.read("jax_enable_x64"):
                raise ValueError("'jax_enable_x64' is disabled, but double precision was requested.\n"
                                 "To use double precision, make sure to include "
                                 "config.update('jax_enable_x64', True) before using any jax features!")
        elif precision == "single":
            self.dtype_num = 32
            self.dtype_c_num = 64
            self.dtype = jnp.float32
            self.dtype_c = jnp.complex64
        else:
            raise NotImplementedError

        # Potential children: cosmology & white noise field
        # Initialize the cosmology
        cosmo_args = {"dtype_num": self.dtype_num, "requires_jacfwd": requires_jacfwd}

        def to_jax_array(input):
            if isinstance(input, jax.core.Tracer):
                return input
            return jnp.asarray(input, dtype=self.dtype)
        make_jnp_array = lambda c: jax.tree_util.tree_map(to_jax_array, c)
        if isinstance(cosmo, dict):
            cosmo_dict = make_jnp_array(cosmo)
            self._cosmo = Cosmology(**cosmo_dict, **cosmo_args)
        elif isinstance(cosmo, str):
            cosmo_dict = make_jnp_array(get_cosmology_dict_from_name(cosmo))
            self._cosmo = Cosmology(**cosmo_dict, **cosmo_args)
        elif isinstance(cosmo, Cosmology):
            self._cosmo = cosmo
        else:
            raise ValueError("Cosmology must be either a dictionary, a string with a predefined cosmology, "
                             "or a Cosmology object.")

        # Initialize the white noise field
        self._white_noise = None
        self._requires_grad_wrt_white_noise = False

        # Also: set a dummy attribute that prevents Jax from crashing when no gradients at all are required
        if (not self._requires_grad_wrt_cosmo) and self._white_noise is None:
            self.__dmy = jnp.zeros((1,) * self.dim, dtype=self.dtype)[0]

    def __str__(self) -> str:
        """String representation of the instance."""
        dim_str = {1: "", 2: "²", 3: "³"}[self.dim]
        return (f"DiscoDJ\n"
                f"——————————\n"
                f"Name: {self.name}\n"
                f"Dimensions: {self.dim}D\n"
                f"Resolution: {self.res}{dim_str} particles\n"
                f"Boxsize: {self.boxsize} Mpc/h\n"
                f"Precision: {self.precision}\n"
                f"Device: {self.device or 'default'}\n"
                f"Cosmology:\n{self.cosmo}"
                )

    def __repr__(self) -> str:
        """Detailed representation of the instance."""
        return (f"DiscoDJ("
                f"dim={self.dim}, "
                f"res={self.res}, "
                f"name='{self.name}', "
                f"device='{self.device}', "
                f"precision='{self.precision}', "
                f"boxsize={self.boxsize}, "
                f"cosmo={self.cosmo.__repr__()}, "  # use cosmo here, not cosmo_settings, in case self is a derivative                
                f"requires_grad_wrt_cosmo={self._requires_grad_wrt_cosmo}, "                
                f"requires_jacfwd={self._requires_jacfwd})")

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten the class for JAX transformations."""
        requires_any_gradient = self._requires_grad_wrt_cosmo or self._requires_grad_wrt_white_noise
        children = []
        if not requires_any_gradient:
            children.append(self.__dmy)
        add_aux = {}
        if self._requires_grad_wrt_cosmo:
            children.append(self.cosmo)
        else:
            add_aux["cosmo"] = self.cosmo
        if self._requires_grad_wrt_white_noise:
            children.append(self.white_noise)
        else:
            add_aux["white_noise"] = self.white_noise
        children = tuple(children)

        aux_data = {
            "dim": self.dim,
            "res": self.res,
            "name": self.name,
            "device": self.device,
            "precision": self.precision,
            "boxsize": self.boxsize,
            "dtype_num": self.dtype_num,
            "dtype_c_num": self.dtype_c_num,
            "dtype": self.dtype,
            "dtype_c": self.dtype_c,
            "cosmo_settings": self.cosmo_settings,
            "requires_grad_wrt_cosmo": self._requires_grad_wrt_cosmo,
            "requires_grad_wrt_white_noise": self._requires_grad_wrt_white_noise,
            "requires_jacfwd": self._requires_jacfwd,
            "pk_table": self._pk_table,
            "ics": self._ics,
            "lpt": self._lpt,
            **add_aux
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "DiscoDJ":
        """Reconstruct the class from flattened data."""
        if aux_data["requires_grad_wrt_cosmo"] and aux_data["requires_grad_wrt_white_noise"]:
            cosmo = children[0]
            white_noise = children[1]
        elif aux_data["requires_grad_wrt_cosmo"]:
            cosmo = children[0]
            white_noise = aux_data["white_noise"]
        elif aux_data["requires_grad_wrt_white_noise"]:
            cosmo = aux_data["cosmo"]
            white_noise = children[0]
        else:
            cosmo = aux_data["cosmo"]
            white_noise = aux_data["white_noise"]

        obj = cls(
            dim=aux_data["dim"],
            res=aux_data["res"],
            name=aux_data["name"],
            device=aux_data["device"],
            precision=aux_data["precision"],
            boxsize=aux_data["boxsize"],
            cosmo=aux_data["cosmo_settings"],
            requires_grad_wrt_cosmo=aux_data["requires_grad_wrt_cosmo"],
            requires_jacfwd=aux_data["requires_jacfwd"],
        )

        # Set the cosmology and white noise field
        obj._cosmo = cosmo
        obj._white_noise = white_noise
        obj._requires_grad_wrt_white_noise = aux_data["requires_grad_wrt_white_noise"]

        # Set the remaining attributes
        obj._pk_table = aux_data["pk_table"]
        obj._ics = aux_data["ics"]
        obj._lpt = aux_data["lpt"]

        return obj

    def update(self, **kwargs) -> "DiscoDJ":
        """Return a new instance of the class with updated attributes.

        :param kwargs: Key-value pairs of attributes to update.
        :return: A new instance of the class with the specified attributes updated.
        """
        if not any(kwargs.values()):
            return self

        # Flatten the current instance into children and auxiliary data
        children, aux_data = self.tree_flatten()

        # Update the relevant attribute(s) in the auxiliary data
        for key, value in kwargs.items():
            if key in aux_data:
                aux_data[key] = value
            elif key == "cosmo":
                children = list(children)
                children[0] = value  # if cosmo is a child
                children = tuple(children)
            elif key == "white_noise":
                children = list(children)
                if self._requires_grad_wrt_cosmo and self._requires_grad_wrt_white_noise:
                    children[1] = value
                elif self._requires_grad_wrt_white_noise:
                    children[0] = value
                children = tuple(children)
            else:
                raise ValueError(f"Attribute '{key}' is not part of the auxiliary data.")

        # Reconstruct the new instance using tree_unflatten
        return self.__class__.tree_unflatten(aux_data, children)

    # # # # # # # # # # # #
    # Cosmology
    # # # # # # # # # # # #
    def with_timetables(self, timetable_settings: dict | None = None) -> "DiscoDJ":
        """Compute the timetables for the cosmology.

        :param timetable_settings: settings for the timetables (see the Cosmology class)
        :return: new DiscoDJ object with the timetables computed
        """
        return self.update(cosmo=self.cosmo.compute_timetables(timetable_settings=timetable_settings))

    # # # # # # # # # # # #
    # Linear power spectrum
    # # # # # # # # # # # #
    def compute_primordial_ps(self, k: Array, normalization: float | Array = 1.0) -> Array:
        """Compute the primordial power spectrum
        :param k: wavenumbers
        :param normalization: P_prim = normalization * k^{n_s}
        :return: primordial power spectrum
        """
        normalization = self.cast_to_dtype(normalization)
        return normalization * k ** self.cosmo.n_s

    def with_linear_ps(self, n_modes: int = 1000, fix_sigma8: bool = False, transfer_function: str = "Eisenstein-Hu",
                       filename: str | None = None, k0_cutoff: float | Array | None = None, cutoff_type: str = "none",
                       transfer_function_kwargs: dict | None = None) -> "DiscoDJ":
        """Compute the linear power spectrum, store it in a table, and return a new DiscoDJ object with the table.

        :param n_modes: number of modes for evaluation. Default is 1000
        :param fix_sigma8: whether to normalize the power spectrum to sigma8 (only used when P(k) is tabulated)
        :param transfer_function: transfer function to use.
            Options: "Eisenstein-Hu" (default), "BBKS", "GeneticAlgorithm", "DiscoEB", "from_file", "none"
        :param filename: If transfer_function is "from_file", the function expects a filename of the tabulated
            power spectrum (NOT a transfer function!). This file should have two columns: k [h/Mpc], P(k) [Mpc^3/h^3]
            and should be normalized to redshift zero.
        :param k0_cutoff: cutoff scale for the power spectrum in Mpc/h
        :param cutoff_type: type of cutoff to apply. Options: "none" (default), "sharp", "gaussian"
        :param transfer_function_kwargs: additional keyword arguments for the transfer function
        :return: new DiscoDJ object with the linear power spectrum
        """
        if self.cosmo.timetables == {}:
            raise RuntimeError("Timetables must be computed first with 'with_timetables'.")

        if transfer_function == "from_file":
            assert filename is not None, "Filename must be provided for 'from_file' transfer function."
            d = onp.loadtxt(filename)
            k = d[:, 0]
            Pk = d[:, 1]
        else:
            # k range needs to be large enough for sigma8 normalization
            kmin = jnp.minimum(0.5 * self.k_fundamental, 1e-5)
            kmax_default = 1e1 if transfer_function == "DiscoEB" else 1e3  # going higher would make DiscoEB very slow
            kmax = jnp.maximum(2.0 * self.k_nyquist * jnp.sqrt(self.dim), kmax_default)
            k = jnp.geomspace(kmin, kmax, n_modes, dtype=self.dtype)
            if transfer_function == "none":
                Pk = self.compute_primordial_ps(k, normalization=1.0)
            else:
                transfer_fn = transfer_dict[transfer_function]
                T = transfer_fn(self.cosmo, k, **(transfer_function_kwargs or {}))
                Pk = self.compute_primordial_ps(k, normalization=1.0) * T ** 2

        # Normalize to sigma_8
        sigma8_sq_from_Pk = get_sigma8_squared_from_Pk(k, Pk)
        sigma8_sq_ratio = self.cosmo.sigma8 ** 2 / sigma8_sq_from_Pk

        if fix_sigma8 or transfer_function != "from_file":
            if transfer_function == "from_file":
                jax.debug.print("sigma8 according to file: {:.4f}, sigma8 according to cosmology: {:.4f}. "
                                "The value of sigma8 will be normalized to the cosmology.",
                                jnp.sqrt(sigma8_sq_from_Pk), jnp.sqrt(self.cosmo.sigma8 ** 2),
                                )
            Pk *= sigma8_sq_ratio

        # Apply cutoff
        if k0_cutoff is not None:
            if cutoff_type == "gaussian":
                cutoff = lambda k_: jnp.exp(-0.5 * (k_ / k0_cutoff) ** 2)
            elif cutoff_type == "sharp":
                cutoff = lambda k_: jnp.where(k_ < k0_cutoff, 1.0, 0.0)
            elif cutoff_type == "none":
                cutoff = lambda k_: 1.0
            else:
                raise ValueError(f"Unknown cutoff type '{cutoff_type}'!")
        else:
            if cutoff_type != "none":
                jax.debug.print("Warning: cutoff type is not 'none', but no cutoff scale is given! Using no cutoff...")
            cutoff = lambda k_: 1.0

        Pk *= cutoff(k)
        pk_table = {"k": k, "Pk": Pk}
        return self.update(pk_table=pk_table)

    def evaluate_linear_ps(self, a: float | Array, k: float | Array) -> Array:
        """Evaluate the linear power spectrum at a given scale factor and wavenumber.

        :param a: scale factor
        :param k: wavenumber in h / Mpc
        :return: linear power spectrum
        """
        if isinstance(a, jax.Array):
            assert a.ndim == 0, "Only scalar scale factors are supported."
        if not all(key in self._pk_table for key in ("k", "Pk")):
            raise RuntimeError("Linear power spectrum must be computed first with 'compute_linear_ps'.")
        return jnp.interp(k, self._pk_table["k"], self._pk_table["Pk"]) * self.cosmo.Dplus(a) ** 2

    def store_linear_ps_as_txt(self, filename: str):
        """Store the linear power spectrum in a text file with two columns: k, P(k).

        :param filename: name of the file
        """
        assert all(key in self._pk_table for key in ("k", "Pk")), "Linear power spectrum must be computed first!"
        assert not self.currently_jitting(), "Cannot store the linear power spectrum while jitting!"
        with open(filename, "w") as f:
            for i in range(len(self._pk_table["k"])):
                f.write(f"{self._pk_table['k'][i]} {self._pk_table['Pk'][i]}\n")

    # # # # # # # # # # # #
    # ICs
    # # # # # # # # # # # #
    def get_ngenic_noise(self, seed: int) -> Array:
        """Get a white noise field with the N-Genic random number generator.

        :param seed: random seed
        :return: white noise field
        """
        assert self.dim == 3, "N-GenIC only supports 3D simulations!"
        return get_ngenic_wnoise(seed=seed, res=self.res, dtype_num=self.dtype_num)

    def with_ics(self, white_noise_space: str = "real", white_noise_field: AnyArray | None = None,
                 requires_grad_wrt_white_noise: bool = False, wavelet_type: str = "db4", wavelet_baseres: int = 16,
                 seed: int = 0, k_order_fourier: str | None = "stable", enforce_mode_stability: bool = False,
                 sphere_mode: bool = False, with_nufft: bool = False, q: Array | None = None,
                 fix_std: float | None = None, convert_to_numpy: bool = False, try_to_jit: bool = False) -> "DiscoDJ":
        """Generates the Gaussian random field initial conditions for the simulation and returns a new DiscoDJ object
        with ICs in self.ics.

        :param white_noise_space:
            'real' (default): real-space white noise field (shape: (res,) * dim)
            'fourier': Fourier-space white noise field (shape: (res ** (dim - 1) * (res // 2 + 1), 2), 2: re. and im.)
            TODO: 'fourier_abs': Fourier-space white noise field with given absolute values and random phases
            TODO: 'wavelet': enable the option to provide a wavelet-space white noise field
            TODO: API for paired simulations
        :param white_noise_field: White noise field whose shape matches the "white_noise_space" (see above).
            If None, a random white noise field will be generated.
        :param requires_grad_wrt_white_noise: if gradients w.r.t. the white noise field are required (True), the white
            noise field will be stored as an attribute of the DiscoDJ object. When differentiating a function that
            receives the DiscoDJ object as an argument, the returned DiscoDJ object will contain the derivative w.r.t.
            the white noise field in the 'white_noise' attribute.
        :param wavelet_type: wavelet type to use if sampling_space == "wavelet", defaults to "db4" (Daubechies 4)
        :param wavelet_baseres: base resolution for the wavelet decomposition, defaults to 16
        :param seed: random seed for the initial conditions (only used if no white noise field has been provided)
        :param k_order_fourier: "stable" (for mode stability) or "natural" (for natural ordering, i.e. flat meshgrid)
        :param enforce_mode_stability: if True: Numpy RNG is used, which has a sequential equivalence guarantee
        :param sphere_mode: if True, all modes with k > k_Nyquist are set to zero
        :param with_nufft: whether to use NUFFT (requires Jax & sampling_space == "fourier")
        :param q: Lagrangian grid to use (if None: use the default grid) NOTE: requires with_nufft!
        :param fix_std: fix the standard deviation of the initial density contrast to this value.
            NOTE: sigma_8 is meaningless after this operation!
        :param convert_to_numpy: whether to convert the initial conditions to numpy after the computation
        :param try_to_jit: whether to try to jit the computation at the level of this function
            (this is not necessary if this method is called inside a jitted function). Defaults to False.
        :return: new DiscoDJ object with the initial conditions
        """
        if self._pk_table == {}:
            raise RuntimeError("The linear power spectrum needs to be computed before generating ICs with "
                               "'with_linear_ps()'! Exiting...")

        if white_noise_field is not None:
            correct_shapes = {"real": (self.res,) * self.dim,
                              "fourier": (self.res ** (self.dim - 1) * (self.res // 2 + 1), 2)}
            assert white_noise_space in correct_shapes.keys(), \
                f"Passing a white noise field in '{white_noise_space}' space is not supported!"
            assert white_noise_field.shape == correct_shapes[white_noise_space], \
                f"White noise field has incorrect shape. Expected: {correct_shapes[white_noise_space]}, " \
                f"got: {white_noise_field.shape}"

        if requires_grad_wrt_white_noise:
            assert white_noise_field is not None, "White noise field must be provided if gradients are required!"

        ics = {}

        if enforce_mode_stability:
            try_to_jit = False
            assert white_noise_space in ["fourier", "wavelet"], \
                "Mode stability is only possible in Fourier or wavelet space!"
            if self.white_noise is None:
                warnings.warn("Warning! JAX has no sequential equivalence guarantee and mode stability is "
                              "therefore not possible! Using numpy random number generation!")
            ics["key"] = seed
        else:
            # wavelet sampling is not jittable because it uses pywavelets
            try_to_jit = try_to_jit and (white_noise_space != "wavelet")
            ics["key"] = jax.random.PRNGKey(seed)

        if with_nufft:
            assert white_noise_space == "fourier", "NUFFT can only be used with sampling_space == 'fourier'!"
            if q is None:
                q = self.q
        else:
            q = None

        ics["fphi"] = generate_grf(sampling_space=white_noise_space, res=self._res, boxsize=self._boxsize,
                                   dim=self.dim, pk_table=self._pk_table, rand_key=ics["key"],
                                   wavelet_type=wavelet_type, wavelet_baseres=wavelet_baseres,
                                   k_order_fourier=k_order_fourier, enforce_mode_stability=enforce_mode_stability,
                                   base_field=white_noise_field, sphere_mode=sphere_mode, with_nufft=with_nufft,
                                   q=q, with_jax=True, try_to_jit=try_to_jit, dtype_num=self.dtype_num,
                                   dtype_c_num=self.dtype_c_num)

        if convert_to_numpy:
            assert not self.currently_jitting(), "Cannot convert to numpy while jitting!"
            ics["fphi"] = onp.array(ics["fphi"])  # asarray would not create a copy by default -> fphi non-writeable

        if fix_std is not None:
            delta = self.get_delta_from_phi(jnp.fft.irfftn(ics["fphi"]))
            current_std = delta.std()
            ics["fphi"] *= fix_std / current_std

        if requires_grad_wrt_white_noise:
            # first, update the requires_grad_wrt_white_noise attribute
            obj = self.update(requires_grad_wrt_white_noise=True)
            # now, set the white noise field and the ICs
            return obj.update(ics=ics, white_noise=ics["fphi"])
        else:
            return self.update(ics=ics)

    def with_external_ics(self, pos: AnyArray | None = None, vel: AnyArray | None = None,
                          delta: AnyArray | None = None, phi: AnyArray | None = None) -> "DiscoDJ":
        """Set the initial conditions from external data.
        If positions and velocites are provided, they can directly be used for starting a simulation.
        If only the linear density contrast delta is provided, the potential phi is computed from it.
        In this case, or if phi itself is provided, LPT can be used to compute initial conditions.

        :param pos: positions of the particles
        :param vel: velocities of the particles
        :param delta: density contrast field
        :param phi: potential field
        :return: new DiscoDJ object with the initial conditions
        """
        ics = {}
        flat_shape = (self.res ** self.dim, self.dim)
        if pos is not None:
            assert vel is not None, "Velocities must be provided if positions are provided!"
            assert pos.shape == flat_shape, f"Positions have incorrect shape: {pos.shape}, should be: {flat_shape}"
            ics["pos"] = pos.astype(self.dtype)
        if vel is not None:
            assert pos is not None, "Positions must be provided if velocities are provided!"
            assert vel.shape == flat_shape, f"Velocities have incorrect shape: {vel.shape}, should be: {flat_shape}"
            ics["vel"] = vel.astype(self.dtype)
        if delta is not None:
            assert phi is None, "Cannot provide both delta and phi!"
            assert (pos is None) and (vel is None), "Either provide delta or positions & velocities!"
            assert delta.shape == (self.res,) * self.dim, \
                f"Density contrast has incorrect shape: {delta.shape}, should be: {(self.res,) * self.dim}"
            ics["fphi"] = jnp.fft.rfftn(self.get_phi_from_delta(delta.astype(self.dtype)))
        if phi is not None:
            assert delta is None, "Cannot provide both delta and phi!"
            assert (pos is None) and (vel is None), "Either provide phi or positions & velocities!"
            assert phi.shape == (self.res,) * self.dim, \
                f"Potential has incorrect shape: {phi.shape}, should be: {(self.res,) * self.dim}"
            ics["fphi"] = jnp.fft.rfftn(phi.astype(self.dtype))
        return self.update(ics=ics)

    def get_k_in_fourier_order(self, k_order_fourier: str | None) -> onp.ndarray:
        """Get the k-vectors in the order that was used to generate the initial conditions in Fourier space.
        That is, when providing white noise fields in Fourier space of shape (res,) * (dim - 1) + (res // 2 + 1),
        the (i, d)th element of the output of this method defines the wave number that corresponds to the i-th element
        of the white noise field in the d-th dimension.

        :param k_order_fourier:
            "stable" for stable ordering of wave vectors (large -> small scales, stable when increasing resolution)
            "natural" for natural ordering (increasing in each dimension, with +/- modes subsequently)
            None for no ordering (default, simply reshape the random numbers to (res,) * dim + (res // 2 + 1,))
        :return: k-vectors in Fourier space in the given order (shape: ((res,) * (dim - 1) + (res // 2 + 1), dim))
        """
        if k_order_fourier is None:
            k_vecs_full = get_fourier_grid((self.res,) * self.dim, self.boxsize, sparse_k_vecs=False,
                                           dtype_num=self.dtype_num, with_jax=False)["k_vecs"]
            return onp.asarray([k_vec.flatten() for k_vec in k_vecs_full]).T
        else:
            indices = get_k_ordering(k_order_fourier, self.res, self.dim, with_jax=False)
            return onp.asarray([self.k_vecs[d].flatten()[indices[d]] for d in range(self.dim)]).T

    def with_ics_plane_wave(self, a_cross: float = 1.0, A_dec: float = 0.0, n_mode: int = 1,
                            from_potential: bool = False, a_ini: float | None = None,
                            convert_to_numpy: bool = False) -> "DiscoDJ":
        """Generate plane wave initial conditions and return a new DiscoDJ object with the ICs in self.ics.

        :param a_cross: shell-crossing scale factor, defaults to 1.0
        :param A_dec: decaying mode amplitude, defaults to 0.0
        :param n_mode: mode to use for the plane wave, defaults to 1
        :param from_potential: whether to generate the plane wave from the potential, defaults to False.
            Specifically, if this is True, this function only sets the potential, and the caller must
            then compute the initial displacement field from the potential with (1)LPT.
        :param a_ini: initial scale factor (must be provided if and only if from_potential is False because positions
            and velocities will directly be stored in this case)
        :param convert_to_numpy: whether to convert the initial conditions to numpy after the computation
        :return: new DiscoDJ object with the initial conditions
        """
        if not from_potential:
            assert a_ini is not None, "Initial scale factor 'a_ini' must be provided if 'from_potential' is False!"
        else:
            assert a_ini is None, "Initial scale factor 'a_ini' must not be provided if 'from_potential' is True!"
        wave = generate_ics_plane_wave(a=a_ini or 1.0, q=self.q, cosmo=self.cosmo, boxsize=self.boxsize,
                                       a_cross=a_cross, A_dec=A_dec, n_mode=n_mode, from_potential=from_potential)

        if convert_to_numpy:
            assert not self.currently_jitting(), "Cannot convert to numpy while jitting!"
            wave = onp.array(wave)

        return self.update(ics=wave)

    def evaluate_plane_wave(self, a: float | Array, a_cross: float = 1.0, A_dec: float = 0.0, n_mode: int = 1) -> dict:
        """Evaluate the plane wave at a given scale factor.

        :param a: scale factor at which to evaluate the plane wave
        :param a_cross: shell-crossing scale factor, defaults to 1.0
        :param A_dec: decaying mode amplitude, defaults to 0.0
        :param n_mode: mode to use for the plane wave, defaults to 1
        :return: plane wave field (X and P)
        """
        return generate_ics_plane_wave(a=a, q=self.q, cosmo=self.cosmo, boxsize=self.boxsize, a_cross=a_cross,
                                       A_dec=A_dec, n_mode=n_mode, from_potential=False)

    def get_delta_linear(self, a: float | Array) -> Array:
        """Get the linear density contrast at a given scale factor.

        :param a: scale factor
        :return: linear density contrast
        """
        return self.cosmo.Dplus(a) * self.delta_ini

    # # # # # # # # # # # #
    # Loading ICs and Snaps
    # # # # # # # # # # # #
    def load_ics(self, filename: str, data_type: str) -> dict:
        """This method loads initial conditions from a file.

        :param filename: path to the file containing the initial conditions (e.g. a Gadget file)
        :param data_type: type of initial conditions, e.g. "camels_dm" or "quijote"
        :return: dictionary with the initial conditions
        """
        if self._res is None:
            raise RuntimeError("Resolution has not been set yet!")

        if data_type == "camels_dm":
            from .data.camels import load_camels_dm_ini
            assert self.dim == 3
            assert self.res <= 256, "Camels resolution is 256!"
            sub_spatial = 256 // self.res  # 256 is the Camels resolution
            out_dict = load_camels_dm_ini(filename, sub_spatial=sub_spatial)

        elif data_type == "quijote":
            from .data.quijote import load_quijote_ini
            assert self.dim == 3
            assert self.res <= 512, "Quijote resolution is 512!"
            sub_spatial = 512 // self.res  # 512 is the Camels resolution
            out_dict = load_quijote_ini(filename, sub_spatial=sub_spatial)

        elif data_type == "gadget_generic":
            from .data.gadget import load_gadget
            assert self.dim == 3
            out_dict = load_gadget(filename)

        else:
            raise NotImplementedError

        # Particles might have been wrapped around already (e.g. for Camels) -> undo wrapping to get correct psi
        out_dict["pos"] = (onp.reshape(self.q, (-1, self.dim))
                           + self.periodic_correct_relative_vector(out_dict["pos"]
                                                                   - onp.reshape(self.q, (-1, self.dim))))
        return out_dict

    # # # # # # # # # # # #
    # Snaps:
    # # # # # # # # # # # #
    def load_snaps(self, folder_or_filename: str, data_type: str, sub: int = 1, load_raw: bool = True,
                   save_raw: bool = False, snap_inds: list | tuple | None = None) -> dict:
        """This method loads snapshots from a folder or file.

        :param folder_or_filename: path to the folder or file containing the snapshots
            (if folder: snaps should be named "snap_000.hdf5", "snap_001.hdf5", ...)
            Filename should only be provided if load_raw is False.
        :param sub: sub-sampling factor, i.e. only every sub-th snapshot is loaded
        :param data_type: type of snapshots, e.g. "camels_dm" or "quijote"
        :param load_raw: whether to load the raw snapshots instead of a stored Numpy file
        :param save_raw: whether to save the loaded snapshots as a Numpy file
        :param snap_inds: indices of the snapshots to load (if None, all snapshots are loaded)
        :return: dictionary with the snapshot properties ("pos", "vel", ...)
        """
        if data_type == "camels_dm":
            from .data.camels import load_snaps_dm
            assert self.dim == 3
            assert self.res <= 256, "Camels resolution is 256!"
            sub_spatial = 256 // self.res  # 256 is the Camels resolution
            return load_snaps_dm(folder_or_filename, sub_spatial=sub_spatial, sub=sub,
                                 load_raw=load_raw, save_raw=save_raw, snap_inds=snap_inds)

        elif data_type == "quijote":
            from .data.quijote import load_snaps
            assert self.dim == 3
            assert self.res <= 512, "Quijote resolution is 512!"
            sub_spatial = 512 // self.res  # 512 is the Quijote resolution
            assert load_raw and not save_raw, "Only raw files are supported for Quijote!"
            return load_snaps(folder_or_filename, sub_spatial=sub_spatial, sub=sub,
                              load_raw=load_raw, save_raw=save_raw, snap_inds=snap_inds)

        elif data_type == "gadget_generic":
            from .data.gadget import load_gadget
            assert self.dim == 3
            return load_gadget(folder_or_filename)

        else:
            raise NotImplementedError

    # # # # # # # # # # # #
    # LPT
    # # # # # # # # # # # #
    def with_lpt(self, n_order: int, grad_kernel_order: int = 0, convert_to_numpy: bool = False,
                 no_mode_left_behind: bool = False, no_transverse: bool = False, no_factors: bool = False,
                 exact_growth: bool = False, try_to_jit: bool = False) -> "DiscoDJ":
        """Compute the Lagrangian perturbation theory displacement field up to the specified order and return a new
        DiscoDJ object with the LPT displacment fields in self.lpt.

        :param n_order: the highest order of the LPT field to compute
        :param grad_kernel_order: the order of the kernel used to compute the gradient of the displacement field
            (0: exact Fourier kernel ik, 2/4/6: finite difference kernel of order 2/4/6)
        :param convert_to_numpy: whether to convert the LPT field to numpy arrays after the computation
        :param no_mode_left_behind: whether to trace all higher modes occurring at higher orders through the computation.
            This is computationally expensive and only implemented in 2D.
        :param no_transverse: whether to neglect the transverse component of the LPT displacement (in 2D/3D).
        :param no_factors: whether to neglect the factors in front of each term and set them to 1. This is only useful
            when manually rescaling the individual LPT terms.
        :param exact_growth: whether to compute the exact growth factors, rather than D^n
        :param try_to_jit: whether to try to jit the computation at the level of this function
            (this is not necessary if this method is called inside a jitted function). Defaults to False.
        """
        if "fphi" not in self._ics.keys():
            if "pos" in self._ics.keys():
                jax.debug.print("'fphi' not found in initial conditions dictionary! Computing phi from initial positions...")
                fphi_ini = jnp.fft.rfftn(self.get_phi_from_delta(self.get_delta_from_pos(self._ics["pos"])))
            else:
                raise KeyError("'fphi' not found in initial conditions dictionary! Please call 'generate_ics()' before "
                               "computing LPT! Exiting...")
        else:
            fphi_ini = self._ics["fphi"]

        if self.dim == 1:
            if n_order > 1:
                warnings.warn("n_order > 1 has no effect for dim = 1, so n_order = 1 will be used...")
                n_order = 1
            nlpt_class = NLPT1
        elif self.dim == 2:
            nlpt_class = NLPT2
        elif self.dim == 3:
            nlpt_class = NLPT3
        else:
            raise NotImplementedError

        # in case LPT is computed twice (e.g. with exact & approximate growth), store all results
        old_lpt_psi = self._lpt.psi if self._lpt is not None else {}
        lpt = nlpt_class(dim=self.dim, res=self.res, cosmo=self.cosmo, boxsize=self.boxsize, n_order=n_order,
                         grad_kernel_order=grad_kernel_order, with_jax=True, try_to_jit=try_to_jit,
                         dtype_num=self.dtype_num, dtype_c_num=self.dtype_c_num, batched=False,
                         exact_growth=exact_growth, no_mode_left_behind=no_mode_left_behind,
                         no_transverse=no_transverse, no_factors=no_factors)

        # Compute the LPT displacement field
        psi_lpt = {**old_lpt_psi, **lpt.compute(fphi_ini=fphi_ini)}

        if convert_to_numpy:
            assert not self.currently_jitting(), "Cannot convert to numpy while jitting!"
            psi_lpt = {k: onp.array(v) for k, v in psi_lpt.items()}

        out = self.update(lpt=lpt)
        out._lpt.psi = psi_lpt
        return out

    def _evaluate_lpt_property_at_a(self, a: Array | float, n_order: int | None = None, include_psi_0: bool = False,
                                    time_derivative: bool = False, D_derivative: bool = False,
                                    eulerian_acc: bool = False, exact_growth: bool = False) -> Array:
        r"""Evaluate the LPT displacement / velocity / position field at the given scale factor and LPT order.

        :param a: the scale factor at which to evaluate the LPT displacement field
        :param n_order: the order of the LPT displacement field to evaluate
        :param include_psi_0: whether to include the zeroth order displacement field (i.e. the initial positions)
            if True -> positions, if False -> displacements
        :param time_derivative: whether to evaluate the time derivative of the LPT displacement field
            (w.r.t. superconformal time)
        :param D_derivative: whether to evaluate the derivative of the LPT displacement field w.r.t. the linear growth
        :param eulerian_acc: whether to evaluate the Eulerian acceleration field instead of the displacement field,
            given by (2/3 a^2 \partial_a^2 + a \partial_a) psi (only for EdS).
            NOTE: it is NOT checked whether the cosmology is EdS, so this is up to the user!
        :param exact_growth: whether to use the exact growth functions, rather than D^n
        :return: the LPT displacement / position field at the given scale factor and LPT order
        """
        if self._lpt is None:
            raise RuntimeError("LPT has not been computed yet, call 'compute_lpt()' first!")

        if n_order is None:
            n_order = self._lpt.n_order  # Take highest computed order as default
        assert n_order > 0, "n_order must be at least 1!"

        if time_derivative:
            assert not include_psi_0, "Time derivative of psi_0 is zero, so include_psi_0 must be False!"

        assert not (time_derivative and D_derivative),\
            "Either 'time_derivative' or 'D_derivative' can be True, but not both!"

        if eulerian_acc:
            assert not time_derivative, "Either 'time_derivative' or 'eulerian_acc' can be True, but not both!"

        return self._lpt.evaluate(a=a, n_order=n_order, q=self.q if include_psi_0 else None, include_psi_0=include_psi_0,
                                  time_derivative=time_derivative, D_derivative=D_derivative,
                                  eulerian_acc=eulerian_acc, exact_growth=exact_growth)

    def evaluate_lpt_psi_at_a(self, a: Array | float, n_order: int | None = None, exact_growth: bool = False) -> Array:
        """Evaluate the LPT displacement field psi(q, t) at the given scale factor and LPT order.

        :param a: the scale factor at which to evaluate the LPT displacement field
        :param n_order: the order of the LPT displacement field to evaluate
        :param exact_growth: whether to use the exact growth functions or the D^n approximation
        :return: the LPT displacement field at the given scale factor and LPT order
        """
        return self._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=False, exact_growth=exact_growth)

    def evaluate_lpt_pos_at_a(self, a: Array | float, n_order: int | None = None, exact_growth: bool = False) -> Array:
        """Evaluate the LPT position field at the given scale factor and LPT order.

        :param a: the scale factor at which to evaluate the LPT position field
        :param n_order: the order of the LPT position field to evaluate
        :param exact_growth: whether to use the exact growth functions
        :return: the LPT position field at the given scale factor and LPT order
        """
        return self.periodic_wrap(self._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=True,
                                                                    exact_growth=exact_growth))

    # Time derivative w.r.t. superconformal time \tilde{t}
    def evaluate_lpt_psi_dot_at_a(self, a: Array | float, n_order: int | None = None, exact_growth: bool = False) \
            -> Array:
        """Evaluate the time derivative of the LPT displacement field w.r.t. superconformal time(!)
        at the given scale factor and LPT order. Note that dx/dsuperconft = a^2 dx/dt is canonical momentum (for m = 1).

        :param a: the scale factor at which to evaluate the time derivative of the LPT displacement field
        :param n_order: the order of the time derivative of the LPT displacement field to evaluate
        :param exact_growth: whether to use the exact growth functions
        :return: the time derivative of the LPT displacement field w.r.t. superconformal time at the given scale factor
        """
        return self._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=False,
                                                time_derivative=True, exact_growth=exact_growth)

    # Eulerian acceleration via Vlasov-Poisson equation
    def evaluate_lpt_eulerian_acc_at_a(self, a: Array | float, n_order: int | None = None) -> Array:
        """Evaluate the Eulerian acceleration field at the given scale factor and LPT order.

        :param a: the scale factor at which to evaluate the Eulerian acceleration field
        :param n_order: the order of the Eulerian acceleration field to evaluate
        :return: the Eulerian acceleration field at the given scale factor and LPT order
        """
        return self._evaluate_lpt_property_at_a(a=a, n_order=n_order, include_psi_0=False,
                                                eulerian_acc=True)
    
    # # # # # # # # # # # #
    # PPT: propagator perturbation theory
    # # # # # # # # # # # #
    def evaluate_ppt(self, a_end: float | Array = 1.0, h_bar: float | None = None, do_rsd: bool = False,
                     lyman_alpha_dict: dict | None = None, propagator: str = "free") -> dict:
        """Evaluate the propagator perturbation theory (PPT) fields at the given scale factor.

        :param a_end: the scale factor at which to evaluate the PPT fields
        :param h_bar: the value of h_bar to use in the PPT computation. If None, we use the value of h_bar proposed in
            Eq. (24) in arXiv:2009.09124
        :param do_rsd: whether to include redshift-space distortions in the PPT computation
        :param lyman_alpha_dict: a dictionary containing the parameters for the Lyman-alpha forest. If None, we do not
            include the Lyman-alpha forest in the PPT computation
        :param propagator: the propagator to use in the PPT computation. Can be either "free" or "kdk" (kick-drift-kick)
        :return: a dictionary containing the PPT fields
        """
        if "fphi" not in self._ics.keys():
            if "pos" in self._ics.keys():
                jax.debug.print("'fphi' not found in initial conditions dictionary!"
                                " Computing phi from initial positions...")
                fphi_ini = self.get_delta_from_pos(self._ics["pos"])
            else:
                raise KeyError("'fphi' not found in initial conditions dictionary! Please call 'generate_ics()' before"
                               "computing PPT! Exiting...")
        else:
            fphi_ini = self._ics["fphi"]
        phi_ini = jnp.fft.irfftn(fphi_ini)

        return evaluate_ppt(cosmo=self.cosmo, boxsize=self._boxsize, phi_ini=phi_ini, a_end=a_end, h_bar=h_bar,
                            do_rsd=do_rsd, lyman_alpha_dict=lyman_alpha_dict, propagator=propagator)

    # # # # # # # # # # # #
    # N-body:
    # # # # # # # # # # # #
    def run_nbody(self, a_ini: float | Array, a_end: float | Array, n_steps: int, time_var: str | AnyArray = "D",
                  stepper: str = "bullfrog", method: str = "pm", res_pm: int | None = None, antialias: int = 0,
                  grad_kernel_order: int = 4, laplace_kernel_order: int = 0, alpha: float = 1.5, theta: float = 0.5,
                  nlpt_cola: float = -2.5, worder: int = 2, deconvolve: bool = False, n_resample: int = 1,
                  resampling_method: str = "fourier", eps_nufft: float | None = None,
                  kernel_size_nufft: int | None = None, sigma_nufft: float = 1.25, ic_method: str = "lpt",
                  bullfrog_ic_settings: dict | None = None, nlpt_order_ics: int | None = None,
                  chunk_size: int | None = None, return_all_a: bool = False,
                  exact_growth: bool = False, convert_to_numpy: bool = False, use_diffrax: bool = False,
                  adjoint_method: bool | None = None, collect_all: bool = False, use_custom_derivatives: bool = True,
                  return_displacement: bool = False) \
            -> tuple[Array, Array, Array]:
        """Run a simulation and return the output.

        :param a_ini: initial scale factor of the simulation
        :param a_end: final scale factor of the simulation
        :param n_steps: number of steps to take
        :param time_var: time variable to use for the simulation ("a", "log_a", "superconft", "D").
            Alternatively, an array of scale factors can be provided (which should include a_ini and a_end).
            Defaults to "D".
        :param stepper: stepper to use for the simulation ("symplectic", "bullfrog", ...). Defaults to "bullfrog".
        :param method: force computation method to use for the simulation ("pm", "nufftpm", ...)
        :param res_pm: resolution of the PM grid to use for the simulation (if applicable)
        :param antialias: antialiasing factor to use for the simulation (if > 0, an interlaced grid is used)
        :param grad_kernel_order: order of the kernel for the gradient computation
            (0: ik, 2/4/6: finite diff., -1: kernel derivative for NUFFT)
        :param laplace_kernel_order: order of the kernel for the Laplacian computation (0: -k^2, 2/4/6: finite diff.)
        :param alpha: tree force splitting scale
        :param theta: tree force opening angle
        :param nlpt_cola: COLA hyperparameter, see https://arxiv.org/abs/1301.0322 TODO: implement!
        :param worder: order of the window function to use for the PM force computation (2: CIC, 3: TSC, 4: PCS)
        :param deconvolve: whether to deconvolve the window function from the PM force computation
        :param n_resample: number of times to resample the particles per dimension (sheet-based interpolation)
            (not implemented for all acceleration methods)
        :param resampling_method: method to use for resampling the particles ("fourier" or "linear")
        :param eps_nufft: tolerance for the NUFFT computation (if applicable, default is 1e-6 for single precision)
        :param kernel_size_nufft: size of the kernel for the NUFFT computation (if applicable), should be even for now.
            Note: specify either eps_nufft or kernel_size_nufft, not both!
        :param sigma_nufft: upsampling factor for NUFFT computation
        :param ic_method: method to use for the initial conditions ("lpt" (default), "bullfrog", "none").
            "lpt" uses LPT at order nlpt_order_ics (default)
            "bullfrog" performs one discreteness-suppressed time step with the BullFrog integrator a=0 -> a_ini,
            "none" assumes that a D-time integrator is used from a = 0 without an explicit initialization step.
        :param bullfrog_ic_settings: settings for the initial conditions if ic_method = "bullfrog"
            (None for default settings as in arXiv:2309.10865)
        :param nlpt_order_ics: order of the initial conditions for the nLPT computation (if None, use the maximum order
            for which LPT has been computed)
        :param chunk_size: split up the computation in chunks of particles to reduce memory usage
        :param return_all_a: whether to return the scale factors for each step
        :param exact_growth: whether to use the exact 3LPT displacement field
        :param convert_to_numpy: whether to convert the output to numpy arrays after the computation
        :param use_diffrax: whether to use Diffrax for the time integration (defaults to False)
        :param adjoint_method: whether to use the adjoint method for reverse-mode differentiation
            (default: True, but False if Jacobian forward mode is used)
        :param collect_all: whether to collect the outputs at every timestep (instead of only at the final time)
        :param use_custom_derivatives: whether to use custom derivatives for the acceleration computation if available
            (not implemented for all acceleration methods)
        :param return_displacement: whether to return the displacements instead of positions (defaults to False)
        :returns: positions (or displacements), velocities, all (if return_all_a=True) / final scale factor(s)
        """
        dtype = self.dtype

        # Diffrax automatically sets type to current Jax standard, so if x64 is enabled, the integration will
        # be done in x64 -> convert the ICs
        if self.dtype_num == 32 and jax.config.read("jax_enable_x64") and use_diffrax:
            dtype_loc = jnp.float64
        else:
            dtype_loc = self.dtype
        dtype_num_loc = onp.finfo(dtype_loc).bits

        # Use adjoint method if no forward mode is required and not collect_all
        if adjoint_method is None:
            adjoint_method = not self._requires_jacfwd and not collect_all

        if adjoint_method:
            assert not self._requires_jacfwd, "The adjoint method is not compatible with Jacobian forward mode!"

        # If pos and vel are stored in ICs, take them
        if "pos" in self._ics.keys() and "vel" in self._ics.keys():
            # jax.debug.print("Positions and velocities found --> using them as ICs!")
            X_ini = self.ensure_flat_shape(self._ics["pos"]).astype(dtype)
            Psi_ini = self.periodic_correct_relative_vector(X_ini - self.ensure_flat_shape(self.q))
            del X_ini
            P_ini = self.ensure_flat_shape(self._ics["vel"]).astype(dtype)
            Pi_ini = P_ini / self.cosmo.Fplus(a_ini)
            del P_ini

        # Else:
        else:
            # initialise with LPT
            # NOTE: "none" can be used when a_ini is set to 0 and a D-time integrator is used without an explicit
            # "initialization step" (with increased discreteness suppression). However, a_ini = 0 is not checked,
            # and under the hood, the 1LPT term (-grad phi_ini) is used, so this is identical to "lpt" in practice.
            if ic_method in ("lpt", "none"):
                if self._lpt is None:
                    raise RuntimeError("LPT terms need to be computed first with 'compute_lpt()")

                if nlpt_order_ics is None:
                    nlpt_order_ics = self._lpt.n_order
                else:
                    assert nlpt_order_ics <= self._lpt.n_order, f"nlpt_order_ics = {nlpt_order_ics} has been provided, " \
                                                                f"but nLPT has only been computed up to order " \
                                                                f"{self._lpt.n_order}!"
                Psi_ini = self.evaluate_lpt_psi_at_a(a_ini, n_order=nlpt_order_ics, exact_growth=exact_growth)
                Pi_ini = self._evaluate_lpt_property_at_a(a=a_ini, n_order=nlpt_order_ics, include_psi_0=False,
                                                          D_derivative=True, exact_growth=exact_growth)
                Psi_ini = self.ensure_flat_shape(Psi_ini)
                Pi_ini = self.ensure_flat_shape(Pi_ini)

            # initialise with BullFrog
            elif ic_method == "bullfrog":
                if self._lpt is None:
                    raise RuntimeError("1LPT term need to be computed first with 'compute_lpt(n_order=1, ...)'")
                # These are the default settings in arXiv:2309.10865
                bullfrog_ic_settings_final = {"a_ini": 0.0, "a_end": a_ini, "n_steps": 1, "time_var": "D",
                                              "stepper": "bullfrog", "method": method, "res_pm": res_pm, "antialias": 1,
                                              "grad_kernel_order": 0, "laplace_kernel_order": 0, "worder": 4,
                                              "deconvolve": True, "n_resample": 2, "resampling_method": "fourier",
                                              "ic_method": "lpt", "nlpt_order_ics": 1, "chunk_size": chunk_size,
                                              "exact_growth": exact_growth, "convert_to_numpy": convert_to_numpy,
                                              "use_diffrax": use_diffrax,
                                              "use_custom_derivatives": use_custom_derivatives,
                                              "return_displacement": True}
                if bullfrog_ic_settings is not None:
                    try:
                        bullfrog_ic_settings_final.update(bullfrog_ic_settings)
                    except:
                        raise ValueError("Invalid BullFrog IC settings provided!")

                Psi_ini, P_ini, _ = self.run_nbody(**bullfrog_ic_settings_final)
                Psi_ini = self.ensure_flat_shape(Psi_ini)
                Pi_ini = self.ensure_flat_shape(P_ini) / self.cosmo.Fplus(a_ini)

            else:
                raise NotImplementedError(f"Initial condition method '{ic_method}' is not implemented!")

        # TODO: take care of COLA

        # Get the stepper
        stepper_dict = {"bullfrog": DKDPiIntegrator,
                        "fastpm": DKDPiIntegrator,
                        "symplectic": DKDSymplecticIntegrator}

        # If steps are explicitly given in terms of a:
        if n_steps is None:
            assert isinstance(time_var, AnyArray), \
                ("If n_steps is not provided, time_var must be an array with scale factors"
                 "(including the initial and final scale factors)!")
            n_steps = len(time_var) - 1
        if isinstance(time_var, AnyArray):
            assert (n_steps is None) or (n_steps == len(time_var) - 1), \
                "n_steps must be equal to len(time_var) - 1 if an array is passed as time_var (or set n_steps = None)!"
            assert len(time_var) > 1, "time_var must contain at least two scale factors (a_ini and a_end)!"
        time_settings = dict(n_steps=n_steps, a_ini=a_ini, a_end=a_end, time_var=time_var)
        solver = stepper_dict[stepper](cosmo=self.cosmo, time_dict=time_settings, integrator_name=stepper,
                                       use_diffrax=use_diffrax, dtype_num=dtype_num_loc)

        # Get all time variables that will be used by the integrator
        args = solver.get_integrator_args()

        # Will return all_a if requested or otherwise a_save (which is a_end if not collect_all)
        a_out = solver.all_a_and_internal[0] if (return_all_a or collect_all) else a_end

        # Get the acceleration function
        if method == "exact":
            assert self.dim == 1, "Exact acceleration is only available in 1D"
            acc_fct = partial(calc_acc_exact_1d, boxsize=self._boxsize, n_part=self.res, with_jax=True)
        elif method == "nufftpm":
            if self.dim != 3:
                warnings.warn("Disco-NUFFT currently only supports 3D simulations! Using FINUFFT instead...")
                if self.dim == 1:
                    assert jax.default_backend() == "cpu", ("Disco-NUFFT currently only supports 3D simulations, "
                                                            "and FINUFFT is not available on GPUs in 1D!")
            use_finufft = self.dim != 3
            acc_fct = partial(calc_acc_nufft, dim=self._dim, boxsize=self._boxsize, n_part=self.res,
                              res_pm=res_pm, grad_order=grad_kernel_order, n_resample=n_resample,
                              resampling_method=resampling_method, dtype_num=dtype_num_loc, chunk_size=chunk_size,
                              with_jax=True, use_finufft=use_finufft, eps=eps_nufft, kernel_size=kernel_size_nufft,
                              sigma=sigma_nufft)
        elif method == "pm":
            assert res_pm is not None, "res_pm must be provided for PM or TreePM acceleration!"
            shape = (res_pm,) * self.dim
            k_dict_pm = get_fourier_grid(shape=shape, boxsize=self.boxsize, sparse_k_vecs=True, full=False,
                                         relative=False, dtype_num=self.dtype_num, with_jax=False)
            k_pm = set_0_to_val(self.dim, k_dict_pm["|k|"], 1.0)  # set constant mode to 1 for k
            k_vecs_pm = k_dict_pm["k_vecs"]

            args_for_pm_or_treepm = dict(dim=self._dim, res_pm=res_pm, n_part=self.res, k_vecs=k_vecs_pm,
                                         k=k_pm, boxsize=self._boxsize, antialias=antialias,
                                         grad_order=grad_kernel_order, lap_order=laplace_kernel_order,
                                         dtype_num=dtype_num_loc, n_resample=n_resample,
                                         resampling_method=resampling_method, worder=worder, deconvolve=deconvolve,
                                         chunk_size=chunk_size, with_jax=True, try_to_jit=False,
                                         requires_jacfwd=self._requires_jacfwd,
                                         use_custom_derivatives=use_custom_derivatives)
            this_acc = calc_acc_PM if method == "pm" else calc_acc_Tree_PM
            acc_fct = partial(this_acc, **args_for_pm_or_treepm)
        elif method == "treepm":
            raise NotImplementedError("TreePM acceleration is currently not supported, "
                                      "but might be added in the future!")
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented!")

        # Set up the ODE terms
        rhs_drift = lambda a, v, args: v
        rhs_kick = lambda a, x, args: acc_fct(x)
        terms = (ODETerm(jax.tree_util.Partial(rhs_drift)),
                 ODETerm(jax.tree_util.Partial(rhs_kick)))  # wrap in Partial so it can be passed to jitted function

        # Extract velocity conversion functions now, as solver.cosmo will be deleted later
        v_to_Pi, Pi_to_v = solver.velocity_to_Pi, solver.Pi_to_velocity

        # Convert the momentum variable to internal momentum used by the integrator
        Mom_ini = Pi_to_v(Pi_ini, a_ini)
        del Pi_ini

        # Delete other solver properties in order to avoid issues with Diffrax
        solver = solver.reset_properties()

        # With diffrax
        if use_diffrax:
            raise NotImplementedError("Diffrax support is currently disabled, as the structure of the Diffrax solvers"
                                      "has changed. Support might be added again at a later stage. For now, please "
                                      "set use_diffax = False.")

            # Set up the time settings for Diffrax
            t_save = jnp.arange(n_steps + 1) if collect_all else jnp.array([n_steps])
            saveat = SaveAt(ts=t_save)
            max_steps = n_steps

            # Set up the adjoint solver
            if self._requires_jacfwd:
                adjoint = diffrax.ForwardMode()
            else:
                if adjoint_method:
                    adjoint_solver = DriftKickDriftAdjoint()
                    adjoint = BacksolveAdjointLeapfrog(solver=adjoint_solver)
                else:
                    adjoint = diffrax.RecursiveCheckpointAdjoint()

            # Run the simulation
            # progress_meter = TqdmProgressMeter(refresh_steps=1)  # Diffrax progress bar currently seems to break VJP
            progress_meter = NoProgressMeter()

            if self.dtype_num == 32 and jax.config.read("jax_enable_x64"):
                Psi_ini, Mom_ini = Psi_ini.astype(jnp.float64), Mom_ini.astype(jnp.float64)
                args = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), args)

            sol = diffeqsolve(terms, solver, t0=0, t1=n_steps, dt0=1, y0=(Psi_ini, Mom_ini),
                              saveat=saveat, max_steps=max_steps, adjoint=adjoint, args=args,
                              progress_meter=progress_meter)
            Psi_out, Mom_out = sol.ys

            if self.dtype_num == 32 and jax.config.read("jax_enable_x64"):
                Psi_out, Mom_out = Psi_out.astype(jnp.float32), Mom_out.astype(jnp.float32)

        # Without diffrax
        else:
            adjoint_solver = DriftKickDriftAdjoint(use_diffrax=False)

            # Run the simulation
            Psi_out, Mom_out = nbody_scan(args, (Psi_ini, Mom_ini), terms, n_steps, adjoint_method,
                                          collect_all, solver, adjoint_solver)

        # If only final fields are requested, squeeze the output
        if not collect_all:
            Psi_out, Mom_out = jnp.squeeze(Psi_out, 0), jnp.squeeze(Mom_out, 0)
            Psi_or_X_out_mesh, Mom_out_mesh = self.ensure_mesh_shape(Psi_out), self.ensure_mesh_shape(Mom_out)
        else:
            Psi_out = jnp.concatenate([Psi_ini[None], Psi_out], axis=0)
            Mom_out = jnp.concatenate([Mom_ini[None], Mom_out], axis=0)
            Psi_or_X_out_mesh = jax.vmap(self.ensure_mesh_shape)(Psi_out)
            Mom_out_mesh = jax.vmap(self.ensure_mesh_shape)(Mom_out)

        # Conversion: internal momentum -> Pi -> canonical momentum
        if collect_all:
            P_out_mesh = v_to_Pi(Mom_out_mesh, a_out) * self.cosmo.Fplus(a_out)[:, None, None]
        else:
            P_out_mesh = v_to_Pi(Mom_out_mesh, a_end) * self.cosmo.Fplus(a_end)

        # Convert Psi -> X
        if not return_displacement:
            if collect_all:
                Psi_or_X_out_mesh = self.periodic_wrap(Psi_or_X_out_mesh + self.q[None])
            else:
                Psi_or_X_out_mesh = self.periodic_wrap(Psi_or_X_out_mesh + self.q)

        # Convert to numpy if requested
        if convert_to_numpy:
            assert not self.currently_jitting(), "Cannot convert to numpy while jitting!"
            Psi_or_X_out_mesh = onp.array(Psi_or_X_out_mesh)
            P_out_mesh = onp.array(P_out_mesh)

        return Psi_or_X_out_mesh, P_out_mesh, a_out

    # # # # # # # # # # # #
    # Converting between quantities
    # # # # # # # # # # # #
    def compute_field_quantity_from_particles(self, pos: Array, quantity: Array | None = None, method: str = "pm",
                                              res: int | None = None, worder: int = 2, deconvolve: bool = False,
                                              n_resample: int = 1,
                                              eps_nufft: float | None = None,
                                              kernel_size_nufft: int | None = None,
                                              antialias: int = 0, chunk_size: int | None = None,
                                              fixed_inds: list | tuple | None = None, try_to_jit: bool = False,
                                              normalize_by_density: bool = True, in_redshift_space: bool = False,
                                              vel: Array | None = None, a: float | Array | None = None,
                                              average_radial_velocity: bool = False, radial_dim: int = -1,
                                              apply_fun_before_mapping_to_redshift_space:
                                              Callable[[Array], Array] | None = None) \
            -> Array:
        """This method computes an arbitrary field quantity given at the positions of the particles.
        It also supports resampling the field with sheet-based interpolation, antialiasing and deconvolution of the
        mass assignment kernel. Further, it can be used to compute the density (contrast) by setting quantity=None.
        If a quantity is provided, the field will be computed as (F * rho) / rho, where F is the quantity and rho the
        density field (note that this requires a sufficient number of particles per cell such that rho is not 
        zero anywhere). The field can be computed on a slice or skewer by setting fixed_inds.        

        :param pos: particle positions, flat indexing
        :param quantity: field quantity to compute (if None, the density field is computed)
        :param method: method to use for the computation ("pm", "nufftpm")
        :param res: resolution of the density field (if None, the 1 cell per particle is used)
        :param worder: order of the MAK (2: CIC, 3: TSC, 4: PCS)
        :param deconvolve: whether to deconvolve the mass assignment kernel (only for PM, as NUFFT always deconvolves)
        :param n_resample: number of times to resample the gravity-source particles with sheet-based interpolation        
        :param eps_nufft: tolerance for the NUFFT computation (if applicable, default is 1e-6 for single precision)
        :param kernel_size_nufft: size of the kernel for the NUFFT computation (if applicable), should be even for now.
            Note: specify either eps_nufft or kernel_size_nufft, not both!
        :param antialias: antialiasing factor for PM (0: no antialiasing, 1: 1xAA with interlaced grid, 2: 2xAA, 3: 3xAA)
        :param chunk_size: size of the chunks of particles for which the computation is performed at once
               (if None, the whole array is used)
        :param fixed_inds: list of indices, one per dimension. If the index is None, all indices along the dimension are
            used. If the index is an integer, the density field is sliced at that index w.r.t. to the grid of size
            res ** dim. If the index is a 2-tuple, the first element is the starting index and the second element is
            the ending index of the slice.
            NOTE: the combination of fixed_inds and n_resample > 1 currently requires a non-JIT context!
        :param normalize_by_density: whether to normalize the field by the density field, i.e. compute (F * rho) / rho 
            instead of (F * rho), where F is the quantity and rho the density field (defaults to True). 
            NOTE: this only affects the case where quantity is not None!
        :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to False.
        :param in_redshift_space: whether to compute the field in redshift space
        :param vel: particle velocities, flat indexing (only used if in_redshift_space=True)
        :param a: scale factor (only used if in_redshift_space=True)
        :param average_radial_velocity: when computing the field in redshift space, there are two options for the
            velocity: either the velocity of the particle is used, or the average velocity of the particles in the cell
            is used. This parameter controls which option is used.
        :param radial_dim: dimension along which the redshift space distortion is applied
        :param apply_fun_before_mapping_to_redshift_space: function to apply to the field before mapping to redshift
            space (only used if in_redshift_space=True and average_radial_velocity=True)!
        :return: array with the field quantity
        """
        pos = self.ensure_flat_shape(pos)

        if quantity is not None:
            assert pos.shape[0] == quantity.shape[0], "pos and quantity must have the same number of particles!"

        if res is None:
            res = self._res

        return compute_field_quantity_from_particles(pos=pos, res=res, boxsize=self.boxsize, cosmo=self.cosmo,
                                                     requires_jacfwd=self._requires_jacfwd, q=self.q, quantity=quantity,
                                                     method=method, worder=worder, deconvolve=deconvolve,
                                                     n_resample=n_resample,
                                                     eps_nufft=eps_nufft, kernel_size_nufft=kernel_size_nufft,
                                                     antialias=antialias, chunk_size=chunk_size,
                                                     fixed_inds=fixed_inds, try_to_jit=try_to_jit,
                                                     normalize_by_density=normalize_by_density,
                                                     in_redshift_space=in_redshift_space, vel=vel, a=a,
                                                     average_radial_velocity=average_radial_velocity,
                                                     radial_dim=radial_dim, apply_fun_before_mapping_to_redshift_space
                                                     =apply_fun_before_mapping_to_redshift_space)

    def get_delta_from_pos(self, pos: Array, method: str = "pm", res: int | None = None, worder: int = 2,
                           deconvolve: bool = False, n_resample: int = 1, antialias: bool = False,
                           eps_nufft: float | None = None, kernel_size_nufft: int | None = None,
                           chunk_size: int | None = None, try_to_jit: bool = True) -> Array:
        """Get the density contrast from the particle positions.

        :param pos: positions
        :param method: method to use for the computation ("pm", "nufftpm")
        :param res: resolution of the density grid (if None: 1 cell per particle)
        :param worder: order of the mass assignment kernel (2: CIC, 3: TSC, 4: PCS)
        :param deconvolve: whether to deconvolve the mass assignment kernel (only for PM, as NUFFT always deconvolves)
        :param n_resample: number of resamplings per dimension with sheet-based interpolation
        :param antialias: whether to antialias the density field (only for PM)
        :param eps_nufft: tolerance for the NUFFT computation (if applicable, default is 1e-6 for single precision)
        :param kernel_size_nufft: size of the kernel for the NUFFT computation (if applicable), should be even for now.
        :param chunk_size: chunk size for the density computation (if None: no chunking)
        :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to True.
        :return: density contrast
        """
        return self.compute_field_quantity_from_particles(pos=pos, quantity=None, method=method, res=res, worder=worder,
                                                          deconvolve=deconvolve, n_resample=n_resample,
                                                          antialias=antialias, eps_nufft=eps_nufft,
                                                          kernel_size_nufft=kernel_size_nufft, chunk_size=chunk_size,
                                                          try_to_jit=try_to_jit)

    def get_delta_from_pos_on_slice_or_skewer(self, pos: Array, fixed_inds: list | None, res: int | None = None,
                                              worder: int = 2, deconvolve: bool = False, n_resample: int = 1,
                                              antialias: bool = False, chunk_size: int | None = None,
                                              try_to_jit: bool = False) -> Array:
        """Get the density contrast from the particle positions, on a slice or skewer.

        :param pos: positions
        :param fixed_inds: list of indices, one per dimension. If the index is None, all indices along the dimension are
            used. If the index is an integer, the density field is sliced at that index w.r.t. to the grid of size
            res ** dim. If the index is a 2-tuple, the first element is the starting index and the second element is
            the ending index of the slice.
        :param res: resolution of the density grid (if None: 1 cell per particle)
        :param worder: order of the mass assignment kernel (2: CIC, 3: TSC, 4: PCS)
        :param deconvolve: whether to deconvolve the mass assignment kernel
        :param n_resample: number of resamplings per dimension with sheet-based interpolation
        :param antialias: whether to antialias the density field
        :param chunk_size: chunk size for the density computation (if None: no chunking)
        :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to False.
        """
        fixed_inds, was_integer_before = preprocess_fixed_inds(self.dim, fixed_inds)

        # Compute the delta field (in all dimensions, but only depositing particles affecting the slice or skewer):
        delta = self.compute_field_quantity_from_particles(pos, quantity=None, method="pm", res=res, worder=worder,
                                                           deconvolve=deconvolve, fixed_inds=fixed_inds,
                                                           n_resample=n_resample, antialias=antialias,
                                                           chunk_size=chunk_size, try_to_jit=try_to_jit)

        # Extract the slice or skewer (delta has length `ind[1] - ind[0]` in the sliced dimensions):
        if fixed_inds is not None:
            delta = delta[tuple(slice(None) if ind is None else (0 if was_int else slice(0, ind[1] - ind[0]))
                                for (was_int, ind) in zip(was_integer_before, fixed_inds))]
        return delta


    def get_delta_from_psi(self, psi: Array, method: str = "pm", res: int | None = None, worder: int = 2,
                           deconvolve: bool = False, n_resample: int = 1, antialias: bool = False,
                           eps_nufft: float | None = None, kernel_size_nufft: int | None = None,
                           chunk_size: int | None = None, try_to_jit: bool = True) -> Array:
        """Get the density contrast from the particle displacements.

        :param psi: displacements
        :param method: method to use for the computation ("pm", "nufftpm")
        :param res: resolution of the density grid (if None: 1 cell per particle)
        :param worder: order of the mass assignment kernel (2: CIC, 3: TSC, 4: PCS)
        :param deconvolve: whether to deconvolve the mass assignment kernel (only for PM, as NUFFT always deconvolves)
        :param n_resample: number of resamplings with sheet-based interpolation
        :param antialias: whether to antialias the density field (only for PM)
        :param eps_nufft: tolerance for the NUFFT computation (if applicable, default is 1e-6 for single precision)
        :param kernel_size_nufft: size of the kernel for the NUFFT computation (if applicable), should be even for now.
            Note: specify either eps_nufft or kernel_size_nufft, not both!
        :param chunk_size: chunk size for the density computation (if None: no chunking)
        :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to True.
        :return: density contrast
        """
        pos = self.get_pos_from_psi(psi)
        return self.get_delta_from_pos(pos, method=method, res=res, worder=worder, deconvolve=deconvolve,
                                       n_resample=n_resample, antialias=antialias, eps_nufft=eps_nufft,
                                       kernel_size_nufft=kernel_size_nufft, chunk_size=chunk_size,
                                       try_to_jit=try_to_jit)

    def get_phi_from_delta(self, delta: Array) -> Array:
        """Compute the potential from the density contrast using the Poisson equation in Fourier space.

        :param delta: density contrast
        :return: potential
        """
        fdelta = jnp.fft.rfftn(delta)
        fphi = inv_laplace_kernel(self.k_vecs) * fdelta
        return jnp.fft.irfftn(fphi)

    def get_delta_from_phi(self, phi: Array) -> Array:
        """Compute the density contrast from the potential using the Poisson equation in Fourier space.

        :param phi: potential
        :return: density contrast
        """
        fphi = jnp.fft.rfftn(phi)
        fdelta = -self.k ** 2 * fphi
        return jnp.fft.irfftn(fdelta)

    def get_pos_from_psi(self, psi: Array) -> Array:
        """
        Get the position from the displacement field.

        :param psi: the displacement field.
        """
        pos = self.periodic_wrap(psi + self.q)
        return pos

    def get_lagrangian_grid(self, res: int | None = None, with_jax: bool = True) -> Array:
        """Get the Lagrangian grid at a given resolution

        :param res: resolution of the density grid (if None: self.res is used, corresponding to 1 cell per particle)
        :param with_jax: whether to output the grid as a Jax or NumPy array (default: True)
        :return: Lagrangian grid q
        """
        if res is None:
            res = self.res

        dx = self.boxsize / res
        meshgrid_vec = (onp.tile(onp.arange(res) * dx, [self.dim, 1])).astype(self.dtype)
        mesh = onp.meshgrid(*meshgrid_vec, indexing="ij")
        np = jnp if with_jax else onp
        return rearrange(np.asarray(mesh), "d ... -> ... d")

    def get_lagrangian_grid_in_redshift_space(self, x: Array, v: Array, a: Array | float, radial_dim: int = -1,
                                              res: int | None = None, worder: int = 2) -> Array:
        """Get the Lagrangian grid in redshift space.

        :param x: position field, flat indexing
        :param v: velocity field, flat indexing
        :param a: scale factor
        :param radial_dim: dimension along which the redshift space mapping is performed (default: last dimension)
        :param res: resolution of the density grid (if None: 1 cell per particle)
        :param worder: order of the mass assignment kernel (2: CIC, 3: TSC, 4: PCS)
        :return: field in redshift space
        """
        if radial_dim == -1:
            radial_dim = self.dim - 1

        # x = self._np.copy(x)
        # v = self._np.copy(v)
        assert x.ndim == v.ndim == 2, "pos_field and vel_field must have shape (n_part, dim)."
        assert x.shape == v.shape, "Positions and velocities must have the same shape."

        mesh_res = res if res is not None else self.res
        vz_grid = self.compute_field_quantity_from_particles(pos=x, quantity=v[..., radial_dim:radial_dim + 1],
                                                             res=mesh_res, worder=worder, deconvolve=False,
                                                             n_resample=1, antialias=False, chunk_size=None)

        # Compute the redshift space positions
        shift = vz_grid / (a ** 2 * self.cosmo.E(a))
        if res == self.res:
            q_shifted = jnp.copy(self.q)
        else:
            dx = self._boxsize / mesh_res
            meshgrid_vec = (onp.tile(onp.arange(mesh_res) * dx, [self.dim, 1])).astype(self.dtype)
            q_mesh = rearrange(jnp.asarray(onp.meshgrid(*meshgrid_vec, indexing="ij")),"d ... -> ... d")
            q_shifted = q_mesh

        return self.periodic_wrap(set_indices_to_val(q_shifted,
                                                     tuple([slice(None)] * self.dim + [radial_dim]),
                                                     q_shifted[..., radial_dim] + shift))

    def map_field_to_redshift_space(self, field: Array, q_in_redshift_space: Array, res: int | None = None,
                                    worder: int = 2) -> Array:
        """Map a field to redshift space.
        TODO: add resampling

        :param field: field to map to redshift space, in mesh shape
        :param q_in_redshift_space: Lagrangian positions in redshift space, indexed in Q-space (i.e. mesh-shape)
        :param res: resolution of the density grid (if None: same resolution as the field)
        :param worder: order of the mass assignment kernel (2: CIC, 3: TSC, 4: PCS)
        :return: field in redshift space
        """
        mesh_res = res if res is not None else q_in_redshift_space.shape[0]
        assert field.shape[0] == q_in_redshift_space.shape[0]
        assert [s == field.shape[0] for s in field.shape[1:]]

        return scatter(mesh=jnp.zeros([mesh_res] * self.dim, dtype=self.dtype),
                       positions=jnp.reshape(q_in_redshift_space, (-1, self.dim)),
                       res=mesh_res,
                       n_part_tot=self.res ** self.dim,
                       boxsize=self.boxsize,
                       dtype_num=self.dtype_num,
                       weight=jnp.reshape(field, (-1, 1)),
                       with_jax=True,
                       worder=worder)

    # # # # # # # # # # # #
    # Periodicity methods:
    # # # # # # # # # # # #
    def periodic_wrap(self, v: Array) -> Array:
        """Wrap a vector v periodically to be between 0 <= v_i < L.

        :param v: input vector
        :return: wrapped vector
        """
        return jnp.mod(v + self._boxsize, self._boxsize)

    def periodic_correct_relative_vector(self, v: Array) -> Array:
        """Wrap a vector v periodically to be between -L/2 <= v_i < L/2.

        :param v: input vector
        :return: wrapped relative vector
        """
        return jnp.mod(v + self._boxsize / 2, self._boxsize) - self._boxsize / 2

    def periodic_distance(self, x1: Array, x2: Array) -> Array:
        """Calculate the distance between two points in a periodic box.

        :param x1: first point
        :param x2: second point
        :return: distance
        """
        return jnp.linalg.norm(self.periodic_correct_relative_vector(x1 - x2), axis=-1)

    # # # # # # # # # # # #
    # Utilities for sheet interpolation
    # # # # # # # # # # # #
    def spawn_interpolated_particles(self, positions: Array, dshift: Array | list | tuple, linear: bool = False)\
            -> Array:
        """Generate a new set of particles by Fourier/linear interpolation of the dark matter sheet to position q+dshift.

        :param positions: particle positions
        :param dshift: 1D vector containing the shift vector in units of grid cells
        :param linear: if True, use piecewise linear interpolation instead of Fourier interpolation (default: False)
        :return: new interpolated particles
        """
        return spawn_interpolated_particles(positions=positions, q=self.q, dshift=dshift, boxsize=self.boxsize,
                                            dtype_num=self.dtype_num, linear=linear, with_jax=True)


    def get_shift_vecs_for_sheet_interpolation(self, n_resample: int) -> onp.ndarray:
        """Get a NumPy array of shift vectors for sheet interpolation for a given resampling factor.

        :param n_resample: resampling factor (e.g. 2 means each particle is interpolated to 2^dim new particles)
        :return: array of shift vectors
        """
        d_vec = onp.linspace(0, 1, n_resample, endpoint=False)
        grid = onp.meshgrid(*((d_vec,) * self.dim), indexing="ij")
        return onp.asarray([g.flatten() for g in grid]).T  # (0, 0, ...)

    def get_lagrangian_pos_for_shift_vecs(self, dshifts: list, with_jax: bool = True) -> AnyArray:
        """Returns the Lagrangian positions associated with the shift vectors

        :param dshifts: shift vectors (as provided by get_shift_vecs_for_sheet_interpolation)
        :param with_jax: whether to output the grid as a Jax or NumPy array (default: True)
        :return: Lagrangian grid positions in the same order as collect_interpolated_particle_positions and
            collect_interpolated_field_property
        """
        np = jnp if with_jax else onp
        if with_jax:
            def scan_fun(carry, dd):
                return carry, np.moveaxis(np.asarray([self.q[..., i] + d_per_dim * self.boxsize / self.res
                                                                  for i, d_per_dim in enumerate(dd)],
                                                                 dtype=self.dtype), 0, -1).reshape(-1, self.dim)

            return np.reshape(jax.lax.scan(scan_fun, 0.0, dshifts)[1], (-1, self.dim))

        else:
            interpolated_q = np.zeros((0, self.dim), dtype=self.dtype)
            for d in dshifts:
                qshifted = np.moveaxis(np.asarray([self.q[..., i] + d_per_dim * self.boxsize / self.res
                                                               for i, d_per_dim in enumerate(d)], dtype=self.dtype),
                                             0, -1).reshape(-1, self.dim)
                interpolated_q = np.concatenate((interpolated_q, qshifted), axis=0)
            return interpolated_q


    def collect_interpolated_particle_positions(self, positions: Array, dshifts: Array | list, linear: bool = False) \
            -> Array:
        """Collect all interpolated particles for a given set of shift vectors.

        :param positions: particle positions indexed in Q-space (i.e. in mesh shape)
        :param dshifts: lists of shift vectors
        :param linear: if True, use piecewise linear interpolation instead of Fourier interpolation (default: False)
        :return: interpolated particle positions (in flat shape, as there is no natural order over the dshifts!)
        """
        spawn_fun = lambda shift: self.spawn_interpolated_particles(positions, shift, linear=linear)
        def scan_fun(carry, dd):
            return carry, self.ensure_flat_shape(spawn_fun(dd))

        return jnp.reshape(jax.lax.scan(scan_fun, 0.0, dshifts)[1], (-1, self.dim))

    def collect_interpolated_field_property(self, field: Array, dshifts: Array | list, linear: bool = False) -> Array:
        """Collect interpolated field property over all particles for a given set of shift vectors.

        :param field: value of field at each grid point on the original mesh in Q-space (at resolution self.res)
        :param dshifts: lists of shift vectors
        :param linear: if True, use piecewise linear interpolation instead of Fourier interpolation (default: False)
        :return: interpolated field property (in flat shape, as there is no natural order over the dshifts!)
        """
        which = "linear" if linear else "fourier"

        if field.ndim == self.dim:
            flat_shape = (-1,)
        elif field.ndim == self.dim + 1:
            flat_shape = (-1, field.shape[-1])
        else:
            raise ValueError("Field must be either a scalar field or a vector field in mesh shape!")

        def scan_fun(carry, dd):
            return carry, jnp.reshape(interpolate_field(self.dim, field, dd, self.boxsize, self.dtype_num,
                                                        with_jax=True, which=which), flat_shape)

        return jnp.reshape(jax.lax.scan(scan_fun, 0.0, dshifts)[1], (-1, self.dim))

    # # # # # # # # # # # #
    # Analysis
    # # # # # # # # # # # #
    def evaluate_jacobian_from_psi(self, psi: Array, grad_kernel_order: int = 0, only_det: bool = False) \
            -> tuple[Array, Array, Array]:
        """Compute the displacement Jacobian from the displacement field.

        :param psi: displacement field between Lagrangian and Eulerian coordinates (in Lagrangian mesh shape)
        :param grad_kernel_order: order for gradient kernel
        :param only_det: if True, only return the determinant of the Jacobian, None for the other outputs
        :return:    (1) J: Jacobian determinant for each q coordinate
                    (2) evals: eigenvalues of the Jacobian for each q coordinate
                    (3) dpsi: Jacobian matrix for each q coordinate
        """
        psi_mesh = self.ensure_mesh_shape(psi)
        all_grad_kernels = [gradient_kernel(self.k_vecs, d, order=grad_kernel_order, with_jax=True)
                            for d in range(self.dim)]
        dpsi = jnp.asarray([jax.vmap(jnp.fft.rfftn, in_axes=-1, out_axes=-1)(psi_mesh)
                             * grad[..., None] for grad in all_grad_kernels])
        dpsi = rearrange(dpsi, "g ... d -> ... (d g)")
        dpsi = jax.vmap(jnp.fft.irfftn, in_axes=-1, out_axes=-1)(dpsi)
        dpsi = rearrange(dpsi, "... (d g) -> ... d g", d=self.dim, g=self.dim)

        # Add 1 to the diagonal
        diag_inds = tuple([slice(None)] * self.dim) + (jnp.arange(self.dim),) * 2
        dpsi = set_indices_to_val(dpsi, diag_inds, 1.0 + dpsi[diag_inds])

        # Also compute determinant and eigenvalues
        J = jnp.linalg.det(dpsi)

        if only_det:
            del dpsi
            dpsi = None
            evals = None
        else:
            evals = jnp.linalg.eigvalsh(dpsi)

        return J, evals, dpsi

    def evaluate_power_spectrum(self, field: Array, bins: int | tuple = 16, deconvolve: bool = False, worder: int = 2,
                                compute_std: bool = False, logarithmic: bool = True, try_to_jit: bool = True) \
            -> tuple[Array, Array, Array]:
        """Evaluate the power spectrum of a field.

        :param field: field of which to evaluate power spectrum
        :param bins: number of bins or alternatively a tuple of bin edges
        :param deconvolve: whether to deconvolve the field with the MAK before the computation
        :param worder: order of the MAK
        :param compute_std: compute standard deviation of power in each bin (default: False)? If not, return None.
        :param logarithmic: whether to use logarithmic bins
        :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to True.
        :return: k, Pk, Pk_err
        """
        cellsize = self.boxsize / field.shape[0]
        if deconvolve:
            jit_if_requested = partial(jax.jit, static_argnames=("with_jax", "double_exponent", "worder")) \
                if try_to_jit else lambda x: x
            field_comp = jit_if_requested(deconvolve_mak)(field, with_jax=True, double_exponent=False, worder=worder)
        else:
            field_comp = field

        jit_if_requested = partial(jax.jit,
                                   static_argnames=("with_jax", "dtype_num", "compute_std", "logarithmic", "bins")) \
            if try_to_jit else lambda x: x

        if isinstance(bins, (Array, list)):
            bins = tuple(bins)

        k, Pk, Pk_err = jit_if_requested(power_spectrum)(field_comp, cellsize=cellsize, bins=bins,
                                                         with_jax=True, dtype_num=self.dtype_num,
                                                         compute_std=compute_std, logarithmic=logarithmic)
        return k, Pk, Pk_err

    def evaluate_cross_power_spectrum(self, field1: Array, field2: Array, bins: int | tuple = 16,
                                      deconvolve: bool = False, worder: int = 2, logarithmic: bool = True,
                                      try_to_jit: bool = True) -> tuple[Array, Array, Array, Array]:
        """Evaluate the cross power spectrum between two fields.

        :param field1: first field
        :param field2: second field
        :param bins: number of bins or alternatively a tuple of bin edges
        :param deconvolve: whether to deconvolve the field with the MAK before the computation
        :param worder: order of the MAK
        :param logarithmic: whether to use logarithmic bins
        :param try_to_jit: whether to try to jit the computation at the level of this function. Defaults to True.
        :return: k, Pkx, Pk1, Pk2
        """

        assert field1.shape == field2.shape
        cellsize = self.boxsize / field1.shape[0]
        if deconvolve:
            jit_if_requested = partial(jax.jit, static_argnames=("with_jax", "double_exponent", "worder")) \
                if try_to_jit else lambda x: x
            field1_comp = jit_if_requested(deconvolve_mak)(field1, with_jax=True, double_exponent=False,
                                                           worder=worder)
            field2_comp = jit_if_requested(deconvolve_mak)(field2, with_jax=True, double_exponent=False,
                                                           worder=worder)
        else:
            field1_comp = field1
            field2_comp = field2
        jit_if_requested = partial(jax.jit, static_argnames=("with_jax", "dtype_num", "logarithmic", "bins")) \
            if try_to_jit else lambda x: x

        if isinstance(bins, (Array, list)):
            bins = tuple(bins)

        k, Pkx, Pk1, Pk2 = jit_if_requested(cross_power_spectrum)(field1_comp, field2_comp, cellsize, bins=bins,
                                                                  with_jax=True, dtype_num=self.dtype_num,
                                                                  logarithmic=logarithmic)
        return k, Pkx, Pk1, Pk2

    def evaluate_bispectrum(self, field: Array, bins: int | tuple = 16, deconvolve: bool = False,
                            worder: int = 2, fast: bool = True, only_B: bool = True, only_equilateral: bool = False)\
            -> dict:
        """Evaluate the full bispectrum of a field using Jax.

        :param field: field of which to compute the bispectrum
        :param bins: number of bins for the spectrum or alternatively a tuple of bin edges
        :param deconvolve: whether to deconvolve the field with the MAK
        :param worder: order of the MAK
        :param fast: whether to use the fast algorithm that precomputes all bins
            (due to memory restrictions in 3D this is limited to resolutions < 512^3)
        :param only_B: whether to return only the bispectrum measurements or also power spectrum and triangles.
            Should be true when trying to compute the gradient of the bispectrum!
        :param only_equilateral: whether to return only the equilateral bispectrum or not.
        :return: dictionary containing k, Bk, Pk
        """
        if deconvolve:
            field = deconvolve_mak(field, with_jax=True, double_exponent=False, worder=worder)

        B_data = bispectrum(field=field, boxsize=self.boxsize, bins=bins, fast=fast, only_B=only_B,
                            compute_norm=False, only_equilateral=only_equilateral)
        normalization = bispectrum(field=field, boxsize=self.boxsize, bins=bins, fast=fast, only_B=only_B,
                                   compute_norm=True, only_equilateral=only_equilateral)
        B_data["Bk"] /= normalization["Bk"]
        if not only_B:
            B_data["Pk"] /= normalization["Pk"]

        return B_data

    # # # # # # # # # # # #
    # Energy functions:
    # # # # # # # # # # # #
    def evaluate_acceleration(self, psi: Array, method: str, antialias: int, grad_kernel_order: int = 4,
                              laplace_kernel_order: int = 0, alpha: float = 1.5, theta: float = 0.5,
                              res_pm: int | None = None, worder: int = 2, deconvolve: bool = False,
                              short_or_long: str | None = None, n_resample: int = 1,
                              resampling_method: str = "fourier", use_finufft: bool = False,
                              eps_nufft: float | None = None, kernel_size_nufft: int | None = None,
                              sigma_nufft: float = 1.25,
                              return_on_grid: bool = False, chunk_size: int | None = None) -> Array:
        """
        Evaluate the Eulerian acceleration field from the input displacements psi.
        Note: the 3/2 Omega_m term is NOT included here!

        :param psi: displacements
        :param method: method to use for calculating the potential ("pm" or "treepm" (2D) or "exact" (1D))
        :param antialias: antialiasing factor
        :param grad_kernel_order: order of the gradient kernel
        :param laplace_kernel_order: order of the Laplace kernel
        :param alpha: TreePM parameter (tree force splitting scale)
        :param theta: TreePM parameter (opening angle)
        :param res_pm: resolution of the PM grid to use for the simulation (if applicable; defaults to self.res)
        :param worder: order of the MAK
        :param deconvolve: whether to deconvolve the field with the MAK
        :param short_or_long: if "short" / "long": only compute use short-range / long-range force
        :param n_resample: number of times to resample the particles (sheet-based interpolation)
        :param resampling_method: method to use for resampling ("fourier" or "linear")
        :param use_finufft: when using NUFFT: use finufft instead of custom NUFFT implementation
        :param eps_nufft: tolerance for the NUFFT computation (if applicable, default is 1e-6 for single precision)
        :param kernel_size_nufft: size of the kernel for the NUFFT computation (if applicable), should be even for now.
            Note: specify either eps_nufft or kernel_size_nufft, not both!
        :param sigma_nufft: upsampling factor for NUFFT computation
        :param return_on_grid: whether to return the acceleration on the grid (True) or on the particles (False)
        :param chunk_size: chunk size for the density computation in order to reduce memory (if None: no chunking)
        :return: acceleration field
        """
        psi = self.ensure_flat_shape(psi)
        n_part_tot, dim = psi.shape
        n_part = int(onp.round(n_part_tot ** (1 / dim)))

        if res_pm is None:
            res_pm = self.res

        if short_or_long is not None:
            assert short_or_long in ["short", "long"], "short_or_long must be either 'short' or 'long'!"
            assert method == "pm", "short_or_long can only be used with method='pm'!"

        if self.dim == 1 and method == "exact":
            acc_fct = partial(calc_acc_exact_1d, boxsize=self.boxsize, n_part=n_part)

        else:
            shape = (res_pm,) * dim
            k_dict = get_fourier_grid(shape=shape, boxsize=self._boxsize, sparse_k_vecs=True, full=False, relative=False,
                                      dtype_num=self.dtype_num, with_jax=False)  # k_dict can be in numpy
            k = set_0_to_val(dim, k_dict["|k|"], 1.0)  # set constant mode to 1 for k
            k_vecs = k_dict["k_vecs"]

            # PM only
            if method == "pm":
                acc_fct = partial(calc_acc_PM, dim=self.dim, n_part=n_part, k=k, k_vecs=k_vecs,
                                  grad_order=grad_kernel_order, lap_order=laplace_kernel_order, res_pm=res_pm,
                                  boxsize=self.boxsize, antialias=antialias, worder=worder, deconvolve=deconvolve,
                                  short_or_long=short_or_long, alpha=alpha, n_resample=n_resample,
                                  resampling_method=resampling_method, dtype_num=self.dtype_num,
                                  return_on_grid=return_on_grid, chunk_size=chunk_size, try_to_jit=True, with_jax=True)
            # TreePM:
            elif method == "treepm":
                raise NotImplementedError

            elif method == "nufftpm":
                acc_fct = partial(calc_acc_nufft, dim=dim, boxsize=self.boxsize, res_pm=res_pm, n_part=n_part,
                                  dtype_num=self.dtype_num, grad_order=grad_kernel_order,
                                  n_resample=n_resample, resampling_method=resampling_method,
                                  return_on_grid=return_on_grid, use_finufft=use_finufft,
                                  chunk_size=chunk_size, eps=eps_nufft, kernel_size=kernel_size_nufft,
                                  sigma=sigma_nufft, with_jax=True)
            else:
                raise NotImplementedError

        return acc_fct(psi)

    # # # # # # # # # # # #
    # Saving
    # # # # # # # # # # # #
    def save_as_hdf5(self, filename: str, x: Array, p: Array, a: Array | float, format_str: str = "gadget123",
                     compressed: bool = True):
        """Save the particle positions and velocities to an HDF5 file in the Gadget (or similar) format.
        NOTE: this file uses the standard Gadget unit system, i.e. masses in 10^10 M_sun, velocities in km/s
        (note that Gadget sqrt(a) convention!), coordinates in Mpc/h.

        :param filename: path to the file to be written
        :param x: positions of the particles
        :param p: velocities of the particles
        :param a: scale factor at which the particles are saved
        :param format_str: format of the file, either "gadget123", "gadget4" or "swift"
        :param compressed: whether to use compression for the HDF5 file
        """
        assert self.dim == 3, "Only 3D supported at the moment!"
        assert not isinstance(x, jax.core.Tracer), "This function must not be called in a JIT context!"
        x = onp.copy(self.ensure_flat_shape(x))  # positions in Mpc / h
        p = onp.copy(self.ensure_flat_shape(p))  # comoving velocities in Mpc / h * H0 = 100 km/s
        save_as_hdf5(filename=filename, cosmo=self.cosmo, boxsize=self._boxsize, x=x, p=p, a=a, format_str=format_str,
                     compressed=compressed)

    # # # # # # # # # # # #
    # Helper functions
    # # # # # # # # # # # #
    def cast_to_dtype(self, val: Array | float) -> Array:
        return jnp.asarray(val, dtype=self.dtype)

    def cast_to_dtype_c(self, val: Array | float) -> Array:
        return jnp.asarray(val, dtype=self.dtype_c)

    def ensure_flat_shape(self, v: AnyArray) -> AnyArray:
        """Reshape a vector of particles to a flat array of shape (res**dim, dim).

        :param v: vector of particles
        :return: particle data in flat shape
        """
        out = v.reshape(-1, self.dim)
        return out

    def ensure_mesh_shape(self, v: AnyArray) -> AnyArray:
        """Reshape a vector of particles to a mesh of shape [res] * dim + [dim], i.e. indexing in Q-space

        :param v: vector of particles
        :return: particle data in mesh shape
        """
        out = v.reshape(self.shape + [v.shape[-1]])
        return out

    @staticmethod
    def currently_jitting() -> bool:
        """
        Check if the function is currently being called in a jitted context.
        :return: True if the function is being called in a jitted context, False otherwise.
        """
        return isinstance(jnp.array(1) + 1, jax.core.Tracer)

    def copy(self) -> "DiscoDJ":
        """Return a copy of the current DiscoDJ object."""
        children, aux_data = self.tree_flatten()
        return self.tree_unflatten(aux_data, children)

    # # # # # # # # # # # #
    # Properties
    # # # # # # # # # # # #
    @property
    def dim(self):
        return self._dim

    @property
    def res(self):
        return self._res

    @property
    def name(self):
        return self._name

    @property
    def device(self):
        return self._device

    @property
    def precision(self):
        return self._precision

    @property
    def boxsize(self):
        return self._boxsize

    @property
    def cosmo_settings(self):
        return self._cosmo_settings

    @property
    def cosmo(self):
        return self._cosmo

    @property
    def white_noise(self):
        return self._white_noise

    @property
    def requires_grad_wrt_cosmo(self):
        return self._requires_grad_wrt_cosmo

    @property
    def requires_grad_wrt_white_noise(self):
        return self._requires_grad_wrt_white_noise

    @property
    def requires_jacfwd(self):
        return self._requires_jacfwd

    # # # # # # # # # # # #
    # Lagrangian grid settings
    # # # # # # # # # # # #
    @property
    def shape(self):
        return self.dim * [self._res]

    @property
    def cellsize(self):
        return self._boxsize / self._res  # note: this is per dimension!

    @property
    def k_fundamental(self):
        return 2 * jnp.pi / self.boxsize

    @property
    def k_nyquist(self):
        return self.res * jnp.pi / self.boxsize

    @property
    def k_dict(self):
        return get_fourier_grid(shape=self.shape, boxsize=self.boxsize, full=False, sparse_k_vecs=True,
                                relative=False, dtype_num=self.dtype_num, with_jax=False)

    @property
    def k(self):
        return self.k_dict["|k|"]

    @property
    def k_vecs(self):
        return self.k_dict["k_vecs"]

    @property
    def k_constant_mode_one(self):
        k_raw = self.k
        k_constant_mode_one = set_0_to_val(self.dim, k_raw, 1.0)
        return k_constant_mode_one

    @property
    def mesh(self):
        dx = self.cellsize
        meshgrid_vec = (onp.tile(onp.arange(self._res) * dx, [self.dim, 1])).astype(self.dtype)
        return onp.meshgrid(*meshgrid_vec, indexing="ij")

    @property
    def q(self):
        return rearrange(onp.asarray(self.mesh), "d ... -> ... d")

    # # # # # # # # # # # #
    # IC convenience properties
    # # # # # # # # # # # #
    @property
    def phi_ini(self):
        return jnp.fft.irfftn(self._ics["fphi"])

    @property
    def delta_ini(self):
        return jnp.fft.irfftn(-self.k ** 2 * self._ics["fphi"])
