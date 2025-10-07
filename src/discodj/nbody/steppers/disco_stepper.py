from diffrax import AbstractTerm
from typing import Tuple, ClassVar
from typing_extensions import TypeAlias
from collections.abc import Callable
from jax import Array
import jax.numpy as jnp
from diffrax._custom_types import BoolScalarLike, DenseInfo, RealScalarLike, Args, Y, VF
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from diffrax._solver import AbstractSolver
from ...cosmology.cosmology import Cosmology
from ...core.utils import AnyArray

_ErrorEstimate_Disco: TypeAlias = None
_SolverState_Disco: TypeAlias = None

__all__ = ["DiscoStepper"]


class DiscoStepper(AbstractSolver):
    """Time integrator for DiscoDJ. Specific integrators will inherit from this class.
    """
    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = LocalLinearInterpolation
    cosmo: Cosmology | None = None
    time_dict: dict[str, jnp.ndarray | float | int | str] | None = None
    use_diffrax: bool = True
    integrator_name: str | None = None
    dtype_num: int = 32

    def order(self, terms):
        return 2

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState_Disco:
        return None

    def reset_properties(self) -> "DiscoStepper":
        return self.__class__(cosmo=None, time_dict=None, use_diffrax=self.use_diffrax, integrator_name=None)

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        term_drift, term_kick = terms
        x0, p0 = y0
        f1 = term_drift.vf(t0, p0, args)
        f2 = term_kick.vf(t0, x0, args)
        return (f1, f2)

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState_Disco,
        made_jump: BoolScalarLike,
    ) -> Tuple[Y, _ErrorEstimate_Disco, DenseInfo | None, _SolverState_Disco, RESULTS]:
        raise NotImplementedError("This method must be implemented in the derived class.")

    def a_to_internal(self, a: float | Array) -> Array:
        all_mappings = {
            "a": lambda x: x,
            "log_a": jnp.log,
            "D": self.cosmo.Dplus,
            "superconft": self.cosmo.a_to_superconft
        }
        try:
            return all_mappings[self.time_dict["time_var"]](a)
        except TypeError:
            # Time steps explicitly given in terms of a?
            if isinstance(self.time_dict["time_var"], AnyArray):
                dtype = getattr(jnp, f"float{self.dtype_num}")
                return jnp.linspace(0, self.time_dict["n_steps"], len(a), dtype=dtype)
            else:
                raise ValueError(f"Time variable {self.time_dict['time_var']} not recognized.")
        except KeyError:
            raise ValueError(f"Time variable {self.time_dict['time_var']} not recognized.")

    def internal_to_a(self, internal: float | Array) -> Array:
        all_mappings = {
            "a": lambda x: x,
            "log_a": jnp.exp,
            "D": self.cosmo.Dplus_to_a,
            "superconft": self.cosmo.superconft_to_a,
        }
        try:
            return all_mappings[self.time_dict["time_var"]](internal)
        except TypeError:
            if isinstance(self.time_dict["time_var"], AnyArray):
                dtype = getattr(jnp, f"float{self.dtype_num}")
                # linearly interpolate between the given a's
                return jnp.interp(internal,
                                  jnp.linspace(0, self.time_dict["n_steps"], (self.time_dict["n_steps"]) + 1,
                                               dtype=dtype),
                                  self.time_dict["time_var"])
            else:
                raise ValueError(f"Time variable {self.time_dict['time_var']} not recognized.")
        except KeyError:
                raise ValueError(f"Time variable {self.time_dict['time_var']} not recognized.")

    def get_integrator_args(self) -> dict:
        raise NotImplementedError("This method must be implemented in the derived class.")

    def Pi_to_velocity(self, pi: Array, a: float | Array) -> Array:
        raise NotImplementedError("This method must be implemented in the derived class.")

    def velocity_to_Pi(self, vel: Array, a: float | Array) -> Array:
        raise NotImplementedError("This method must be implemented in the derived class.")

    @property
    def all_a_and_internal(self) -> tuple[Array, Array]:
        dtype = getattr(jnp, f"float{self.dtype_num}")
        internal_start_and_end = self.a_to_internal(jnp.asarray([self.time_dict["a_ini"], self.time_dict["a_end"]],
                                                                dtype=dtype))
        internal_start, internal_end = internal_start_and_end
        all_internal = jnp.linspace(internal_start, internal_end, self.time_dict["n_steps"] + 1, dtype=dtype)
        all_a = self.internal_to_a(all_internal)
        return all_a, all_internal
