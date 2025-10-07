import jax
import jax.numpy as jnp
from ..core.types import AnyArray
from ..core.utils import rk4_integrate

__all__ = ["Cosmology"]


@jax.tree_util.register_pytree_node_class
class Cosmology:
    __slots__ = [
        "_dtype_num", "_dtype",
        "_Omega_c", "_Omega_b", "_h", "_sigma8", "_n_s", "_Omega_k", "_w0", "_wa",
        "_requires_jacfwd", "_timetables", "is_derivative"
    ]
    def __init__(
            self,
            Omega_c: float | AnyArray,
            Omega_b: float | AnyArray,
            h: float | AnyArray,
            sigma8: float | AnyArray,
            n_s: float | AnyArray,
            Omega_k: float | AnyArray = 0.0,
            w0: float | AnyArray = -1.0,
            wa: float | AnyArray = 0.0,
            timetable_settings: dict | None = None,
            dtype_num: int = 32,            
            requires_jacfwd: bool = False,
            is_derivative: bool = False   # TODO: should be automatically detected!
    ):
        """The DiscoDJ cosmology class.

        :param Omega_c: Cold dark matter density parameter.
        :param Omega_b: Baryonic matter density parameter.
        :param h: Dimensionless Hubble parameter (H_0 / 100 km/s/Mpc).
        :param sigma8: RMS mass fluctuation in spheres of 8 Mpc/h.
        :param n_s: Scalar spectral index.
        :param Omega_k: Curvature density parameter. Default is 0.0.
        :param w0: Dark energy equation of state parameter at z=0. Default is -1.0.
        :param wa: Evolution of the dark energy equation of state. Default is 0.0.
        :param dtype_num: Number of bits for the data type. Default is 32.        
        :param requires_jacfwd: Whether the cosmology requires jax.jacfwd. Default is False.
        :param is_derivative: Whether this object stores the derivatives w.r.t. the cosmological parameters
            (should not be used directly by the user). Default is False.
        """
        self._dtype_num = dtype_num
        self._dtype = jnp.float32 if dtype_num == 32 else jnp.float64
        parameter_keys = ("Omega_c", "Omega_b", "h", "sigma8", "n_s", "Omega_k", "w0", "wa")
        parameters = (Omega_c, Omega_b, h, sigma8, n_s, Omega_k, w0, wa)
        for k, param in zip(parameter_keys, parameters):
            setattr(self, f"_{k}", param)

        self._requires_jacfwd = requires_jacfwd
        self._timetables = {}
        self.is_derivative = is_derivative

    @staticmethod
    def forbidden_for_derivative(func):
        """Decorator to prevent evaluation of a function when object is a derivative object."""
        def wrapper(self, *args, **kwargs):
            if self.is_derivative:
                raise RuntimeError(f"Method {func.__name__} is not available for derivative object!")
            return func(self, *args, **kwargs)

        return wrapper

    @forbidden_for_derivative
    def compute_timetables(self, timetable_settings: dict | None) -> "Cosmology":
        """Compute the timetables for growth factor and superconformal time.

        :param timetable_settings: dictionary with settings for the timetables.
        :return: A new Cosmology instance with computed timetables.
        """
        # Set timetable setting defaults
        a_min = 1e-10
        steps_default = 2500
        s = {"a_min": a_min, "spacing_power": 0.0, "steps": steps_default, "a_max": 1.0}

        if timetable_settings is not None:
            assert all(key in s for key in timetable_settings), "Invalid timetable settings."
            if "a_min" in timetable_settings:
                assert timetable_settings["a_min"] > 0.0, "Invalid minimum scale factor."
            if "steps" in timetable_settings:
                assert timetable_settings["steps"] > 0, "Number of steps must be positive."
            if "a_max" in timetable_settings:
                assert timetable_settings["a_max"] >= 1.0, "Maximum scale factor should be >= 1.0."
            if "spacing_power" in timetable_settings:
                assert 0.0 <= timetable_settings["spacing_power"] <= 1.0, "Invalid spacing power."
            s.update(timetable_settings)

        # Log spacing
        if s["spacing_power"] == 0:
            a_table = jnp.geomspace(s["a_min"], s["a_max"], s["steps"], dtype=self._dtype)
        else:
            exponent = s["spacing_power"]
            a_table = jnp.linspace(s["a_min"] ** exponent, s["a_max"] ** exponent, s["steps"],
                                   dtype=self._dtype) ** (1 / exponent)

        growth_dict = self.compute_unnormed_growth(a_table, a_min_integration=s["a_min"])
        superconft_table = self.compute_superconft(a_table)

        # Update timetables
        new_timetables = {"a": a_table, "superconft": superconft_table, **growth_dict}
        return self.update_timetables(**new_timetables)

    def __repr__(self):
        """Technical string representation of the cosmology."""
        return (
            f"Cosmology(Omega_c={self._Omega_c}, Omega_b={self._Omega_b}, h={self._h}, "
            f"sigma8={self._sigma8}, n_s={self._n_s}, Omega_k={self._Omega_k}, w0={self._w0}, wa={self._wa}, "
            f"dtype_num={self._dtype_num}, requires_jacfwd={self._requires_jacfwd}, "
            f"is_derivative={self.is_derivative})"
        )

    def __str__(self):
        """String representation of the cosmology."""
        format_str = "{:.4f}"
        try:
            cosmo_str = (
                f"  Omega_c: {format_str.format(self._Omega_c)} \n"
                f"  Omega_b: {format_str.format(self._Omega_b)} \n"
            )

            if not self.is_derivative:
                cosmo_str += (
                    f"  Omega_de: {format_str.format(self.Omega_de)} \n"
                )

            cosmo_str += (
                f"  h: {format_str.format(self._h)} \n"
                f"  n_s: {format_str.format(self._n_s)} \n"
                f"  sigma8: {format_str.format(self._sigma8)} \n"
                f"  Omega_k: {format_str.format(self._Omega_k)} \n"
                f"  w0: {format_str.format(self._w0)} \n"
                f"  wa: {format_str.format(self._wa)}"
            )

        except TypeError:
            cosmo_str = self.__repr__()  # Fallback to technical representation if format_str fails (for derivatives)
        if self.is_derivative:
            cosmo_str += "\n  (derivative object!)"
        return cosmo_str

    # # # # # # # # # # # #
    # Properties
    # # # # # # # # # # # #
    @property
    def Omega_c(self):
        """Cold dark matter density parameter."""
        return self._Omega_c

    @property
    def Omega_b(self):
        """Baryonic matter density parameter."""
        return self._Omega_b

    @property
    def Omega_m(self):
        """Total matter density parameter."""
        return self._Omega_c + self._Omega_b

    @property
    def h(self):
        """Dimensionless Hubble parameter (H_0 / 100 km/s/Mpc)."""
        return self._h

    @property
    def n_s(self):
        """Scalar spectral index."""
        return self._n_s

    @property
    def sigma8(self):
        """RMS mass fluctuation in spheres of 8 Mpc/h."""
        return self._sigma8

    @property
    def Omega_k(self):
        """Curvature density parameter."""
        return self._Omega_k

    @property
    def w0(self):
        """Dark energy equation of state parameter at z=0."""
        return self._w0

    @property
    def wa(self):
        """Evolution of the dark energy equation of state."""
        return self._wa

    @property
    @forbidden_for_derivative
    def Omega_de(self):
        """Compute the dark energy density parameter."""
        return 1.0 - self._Omega_c - self._Omega_b - self._Omega_k

    @property
    @forbidden_for_derivative
    def is_EdS(self):
        """Check if the cosmology is Einstein-de Sitter."""
        return jnp.abs(self.Omega_de) <= 1e-6 and self._Omega_k == 0.0

    @property
    def timetables(self):
        """Return the timetables."""
        return self._timetables

    # # # # # # # # # # # #
    # Jax PyTree methods
    # # # # # # # # # # # #
    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten the class for JAX transformations.

        :return: A tuple containing the child attributes and auxiliary data.
        """
        children = (
            self.Omega_c,
            self.Omega_b,
            self.h,
            self.sigma8,
            self.n_s,
            self.Omega_k,
            self.w0,
            self.wa,
        )
        aux_data = {
            "dtype_num": self._dtype_num,
            "dtype": self._dtype,
            "requires_jacfwd": self._requires_jacfwd,
            "timetables": self._timetables,
            "is_derivative": self.is_derivative
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple):
        """Reconstruct the class from flattened data.

        :param aux_data: Auxiliary data containing the fixed attributes.
        :param children: Flattened child attributes.
        :return: Reconstructed `Cosmology` instance.
        """
        num_params = 8  # Number of cosmological parameters
        # is_derivative = all(isinstance(c, jax.interpreters.ad.JVPTracer) for c in children) or aux_data["is_derivative"]
        cosmological_params = children[:num_params]
        instance = cls(
            Omega_c=cosmological_params[0],
            Omega_b=cosmological_params[1],
            h=cosmological_params[2],
            sigma8=cosmological_params[3],
            n_s=cosmological_params[4],
            Omega_k=cosmological_params[5],
            w0=cosmological_params[6],
            wa=cosmological_params[7],
            dtype_num=aux_data["dtype_num"],
            requires_jacfwd=aux_data["requires_jacfwd"],
            is_derivative=aux_data["is_derivative"]
        )
        instance._timetables = aux_data["timetables"]
        return instance

    def update_timetables(self, **new_tables) -> "Cosmology":
        """
        Return a new Cosmology instance with updated timetables.

        :param new_tables: Key-value pairs to add to or replace in the timetables.
        :return: A new Cosmology instance with updated timetables.
        """
        # Update the timetables dictionary
        updated_timetables = {**self._timetables, **new_tables}

        # Flatten the current object
        children, aux_data = self.tree_flatten()

        # Replace the old timetables with the new one in the auxiliary data
        aux_data["timetable_keys"] = list(updated_timetables.keys())

        # Rebuild the instance using `tree_unflatten`
        new_instance = self.__class__.tree_unflatten(aux_data, children)
        new_instance._timetables = updated_timetables
        return new_instance

    # # # # # # # # # # # #
    # Background cosmology
    # # # # # # # # # # # #
    @forbidden_for_derivative
    def w(self, a: float | AnyArray) -> AnyArray:
        """Dark energy equation of state parameter as a function of scale factor."""
        return jnp.asarray(self.w0 + self.wa * (1 - a))

    @forbidden_for_derivative
    def Omega_de_of_a(self, a: float | AnyArray) -> AnyArray:
        """Dark energy density term as a function of scale factor."""
        return self.Omega_de * (a ** (-3 * (1 + self.w0 + self.wa)) * jnp.exp(-3 * self.wa * (1 - a)))

    @forbidden_for_derivative
    def E(self, a: float | AnyArray) -> AnyArray:
        """Dimensionless Hubble parameter as a function of scale factor."""
        # see https://arxiv.org/pdf/astro-ph/0508156 (Eqs. 3 & 5) for w(a) = w0 + wa (1-a)
        return jnp.sqrt(self.Omega_m * a ** -3 + self.Omega_k * a ** -2 + self.Omega_de_of_a(a))

    @forbidden_for_derivative
    def Eda(self, a: AnyArray) -> AnyArray:
        """Compute the derivative of the dimensionless Hubble parameter E(a) with respect to a.
        """
        # Matter term
        matter_term = self.Omega_m * a ** -3
        dmatter_da = -3 * self.Omega_m * a ** -4

        # Curvature term
        curvature_term = self.Omega_k * a ** -2
        dcurvature_da = -2 * self.Omega_k * a ** -3

        # Dark energy term
        de_term = self.Omega_de_of_a(a)
        dde_da = self.Omega_de * (3 * a ** (-4 - 3 * self.w0 - 3 * self.wa)
                                  * jnp.exp(-3 * self.wa * (1 - a)) * (-1 - self.w0 + (a - 1) * self.wa))

        dE2_da = dmatter_da + dcurvature_da + dde_da
        E = jnp.sqrt(matter_term + curvature_term + de_term)
        return 0.5 * dE2_da / E

    @forbidden_for_derivative
    def H(self, a: float | AnyArray) -> AnyArray:
        """Hubble parameter as a function of scale factor."""
        return self.H0() * self.E(a)

    @forbidden_for_derivative
    def H0(self) -> AnyArray:
        """Hubble parameter at z=0."""
        return 100.0 * self.h

    # # # # # # # # # # # #
    # Time table functions
    # # # # # # # # # # # #
    @forbidden_for_derivative
    def compute_unnormed_growth(self, a: AnyArray, a_min_integration: float) -> dict:
        """Solve the ODE for the unnormalized growth factor as a function of scale factor."""
        # If double precision is enabled, solve the ODE with 64-bit precision and convert to 32-bit
        if jax.config.read("jax_enable_x64") and a.dtype == jnp.float32:
            a = a.astype(jnp.float64)
            eps = jnp.finfo(jnp.float64).eps
        else:
            eps = jnp.finfo(jnp.float32).eps

        ln_a = jnp.log(a)

        y1_0 = jnp.log(a_min_integration)  # D1 ~ a
        f1_0 = 1.0
        D2_0 = -3.0 / 7.0 * a_min_integration ** 2  # D2 ~ -3/7 a^2
        y2_0 = jnp.log(jnp.abs(D2_0))
        f2_0 = 2.0
        D3a_0 = 1.0 / 3.0 * a_min_integration ** 3  # D3a ~ +1/3 a^3
        y3a_0 = jnp.log(jnp.abs(D3a_0))
        f3a_0 = 3.0
        D3b_0 = -10.0 / 21.0 * a_min_integration ** 3  # D3b ~ -10/21 a^3
        y3b_0 = jnp.log(jnp.abs(D3b_0))
        f3b_0 = 3.0
        D3c_0 = 1.0 / 7.0 * a_min_integration ** 3  # D3c ~ +1/7 a^3
        y3c_0 = jnp.log(jnp.abs(D3c_0))
        y0_log = jnp.array([y1_0, f1_0, y2_0, f2_0, y3a_0, f3a_0, y3b_0, f3b_0, y3c_0])

        # Signs from EdS expressions at small a
        s2 = jnp.sign(-3.0 / 7.0)  # D2 ~ -3/7 a^2
        s3a = jnp.sign(1.0 / 3.0)  # D3a ~ +1/3 a^3
        s3b = jnp.sign(-10.0 / 21.0)  # D3b ~ -10/21 a^3
        s3c = jnp.sign(1.0 / 7.0)  # D3c ~ +1/7 a^3

        def growth_derivs_log_signed(ln_a, y):
            """
            State: [y1,f1, y2,f2, y3a,f3a, y3b,f3b, y3c]
            """
            a_loc = jnp.exp(ln_a)
            a_loc = jnp.maximum(a_loc, eps)
            Esqr = self.E(a_loc) ** 2
            Om_a = (self.Omega_m * a_loc ** -3) / Esqr
            Ox_a = self.Omega_de_of_a(a_loc) / Esqr
            w_a = self.w(a_loc)

            # Unpack
            (y1, f1, y2, f2, y3a, f3a, y3b, f3b, y3c) = y

            # Drag coefficient
            drag = (1.0 - 0.5 * (Om_a + (1.0 + 3.0 * w_a) * Ox_a))

            # D1
            dy1 = f1
            df1 = 1.5 * Om_a - f1 ** 2 - f1 * drag
            D1 = jnp.exp(y1)
            D1_sq = D1 ** 2
            D1_cu = D1 ** 3

            # D2
            D2 = s2 * jnp.exp(y2)
            dy2 = f2
            df2 = 1.5 * Om_a * (1.0 - D1_sq / D2) - f2 ** 2 - f2 * drag

            # D3a
            D3a = s3a * jnp.exp(y3a)
            dy3a = f3a
            df3a = 1.5 * Om_a * (1.0 + 2.0 * D1_cu / D3a) - f3a ** 2 - f3a * drag

            # D3b
            D3b = s3b * jnp.exp(y3b)
            dy3b = f3b
            df3b = 1.5 * Om_a * (1.0 + (2.0 * D1 * D2 - 2.0 * D1_cu) / D3b) - f3b ** 2 - f3b * drag

            # D3c (transverse)
            D3c = s3c * jnp.exp(y3c)
            dy3c = D1 * D2 / D3c * (f1 - f2)
            return jnp.array([dy1, df1, dy2, df2, dy3a, df3a, dy3b, df3b, dy3c])

        ys_log = rk4_integrate(jax.jit(growth_derivs_log_signed), y0_log, ln_a)
        (y1, f1, y2, f2, y3a, f3a, y3b, f3b, y3c) = ys_log.T
        D1 = jnp.exp(y1)
        D2 = s2 * jnp.exp(y2)
        D3a = s3a * jnp.exp(y3a)
        D3b = s3b * jnp.exp(y3b)
        D3c = s3c * jnp.exp(y3c)

        D1p = f1 * D1 / a
        D2p = f2 * D2 / a
        D3ap = f3a * D3a / a
        D3bp = f3b * D3b / a

        ys = jnp.stack([D1, D1p, D2, D2p, D3a, D3ap, D3b, D3bp, D3c], axis=-1)
        gradients_log = jax.vmap(growth_derivs_log_signed)(a, ys)
        gradients = gradients_log * ys / a[:, None]  # Convert to linear derivatives

        # Normalize and build a dictionary
        Dplus_unnormed_at_1 = jnp.interp(1.0, a, ys[:, 0])
        names = ("Dplus", "Dplusda", "D2plus", "D2plusda", "D3plusa", "D3plusada", "D3plusb", "D3plusbda", "D3plusc")
        norm_exponent = (1, 1, 2, 2, 3, 3, 3, 3, 3)
        growth_dict = {name: ys[:, i] / Dplus_unnormed_at_1 ** norm_exponent[i] for i, name in enumerate(names)}
        growth_dict["Dplus_unnormed_at_1"] = Dplus_unnormed_at_1

        # Append the second derivative of Dplus
        Dplusdada = gradients[:, 1]
        growth_dict["Dplusdada"] = Dplusdada / Dplus_unnormed_at_1

        # Also append the derivative of the transversal 3rd order term
        D3plusc_da_unnormed = D2 * D1p - D1 * D2p
        growth_dict["D3pluscda"] = D3plusc_da_unnormed / Dplus_unnormed_at_1 ** 3

        # Convert to 32-bit precision if necessary
        growth_dict = {key: val.astype(self._dtype) for key, val in growth_dict.items()}

        return growth_dict

    @forbidden_for_derivative
    def compute_superconft(self, a: float | AnyArray):
        def cumulative_trapezoidal(a, f):
            da = jnp.diff(a)
            midpoints = (f[:-1] + f[1:]) / 2
            return -jnp.cumsum(jnp.concatenate([jnp.zeros(1), (da * midpoints)[::-1]]))[::-1]

        # Compute the superconformal time
        superconft = cumulative_trapezoidal(a, 1.0 / (self.E(a) * a ** 3))

        # Normalize the superconformal time such that it is 0 at a = 1
        superconft -= superconft[-1]

        return superconft

    @forbidden_for_derivative
    def get_interpolated_property(self, x: float | AnyArray, key_from: str, key_to: str) -> AnyArray:
        """
        Function to interpolate properties stored in the timetable.

        :param x: value(s) at which to interpolate
        :param key_from: key in the timetable dictionary to look up
        :param key_to: key in the timetable dictionary to interpolate
        :return: interpolated value(s)
        """
        interp_fct = jnp.interp

        # If timetables are missing, compute them and replace self with a new instance
        if not self._timetables:
            new_self = self.compute_timetables()
            return new_self.get_interpolated_property(x, key_from, key_to)

        if key_from not in self._timetables.keys():
            raise RuntimeError(f"Key {key_from} not found in the timetable.")
        if key_to not in self._timetables.keys():
            raise RuntimeError(f"Key {key_to} not found in the timetable.")

        # Interpolate
        a_is_scalar = jnp.asarray(x).ndim == 0
        interp_out = interp_fct(jnp.atleast_1d(x),
                                jnp.asarray(self._timetables[key_from]),
                                jnp.asarray(self._timetables[key_to])).astype(self._dtype)
        return interp_out[0] if a_is_scalar else interp_out

    @forbidden_for_derivative
    def superconft_to_a(self, superconft: float | AnyArray) -> AnyArray:
        """Convert superconformal time to scale factor."""
        return self.get_interpolated_property(superconft, "superconft", "a")

    @forbidden_for_derivative
    def a_to_superconft(self, a: float | AnyArray) -> AnyArray:
        """Convert scale factor to superconformal time."""
        return self.get_interpolated_property(a, "a", "superconft")

    @forbidden_for_derivative
    def Dplus(self, a: float | AnyArray) -> AnyArray:
        """Growth factor as a function of scale factor."""
        return self.get_interpolated_property(a, "a", "Dplus")

    @forbidden_for_derivative
    def Dplusda(self, a: float | AnyArray) -> AnyArray:
        """Derivative of the growth factor with respect to scale factor."""
        return self.get_interpolated_property(a, "a", "Dplusda")

    @forbidden_for_derivative
    def Dplusdada(self, a: float | AnyArray) -> AnyArray:
        """Second derivative of the growth factor with respect to scale factor."""
        return self.get_interpolated_property(a, "a", "Dplusdada")

    @forbidden_for_derivative
    def Dplus_to_a(self, Dplus: float | AnyArray) -> AnyArray:
        """Convert growth factor to scale factor."""
        return self.get_interpolated_property(Dplus, "Dplus", "a")

    @forbidden_for_derivative
    def growth_rate(self, a: float | AnyArray) -> AnyArray:
        """Growth rate d ln(D) / d ln(a) as a function of scale factor."""
        return self.Dplusda(a) * a / self.Dplus(a)

    @forbidden_for_derivative
    def Fplus(self, a: float | AnyArray) -> AnyArray:
        """Momentum growth factor F(a) = a^3 E(a) Dplus'(a) as a function of scale factor."""
        return a ** 3 * self.E(a) * self.Dplusda(a)

    @forbidden_for_derivative
    def Fplusda(self, a: float | AnyArray) -> AnyArray:
        """Derivative of the momentum growth factor with respect to scale factor."""
        E_of_a = self.E(a)
        return a ** 2 * (a * E_of_a * self.Dplusdada(a) + self.Dplusda(a) * (3 * E_of_a + a * self.Eda(a)))
