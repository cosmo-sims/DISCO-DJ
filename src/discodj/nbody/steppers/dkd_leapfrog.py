import jax
import jax.numpy as jnp
from typing import Tuple
from typing_extensions import TypeAlias
from equinox.internal import ω
from diffrax._custom_types import BoolScalarLike, DenseInfo, RealScalarLike, Y, Args
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm
from .disco_stepper import DiscoStepper
from ...core.utils import set_indices_to_val

_ErrorEstimate_DKD: TypeAlias = None
_SolverState_DKD: TypeAlias = None

__all__ = ["DriftKickDrift", "DriftKickDriftAdjoint"]


class DriftKickDrift(DiscoStepper):
    """Drift-kick-drift leapfrog Verlet with uniform step size.
    """

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,  # start in integer time
        t1: RealScalarLike,  # end in integer time
        y0: Y,
        args: Args,
        solver_state: _SolverState_DKD,
        made_jump: BoolScalarLike,
    ) -> Tuple[Y, _ErrorEstimate_DKD, DenseInfo | None, _SolverState_DKD, RESULTS]:
        del solver_state, made_jump

        # Map t0 to index (t1 is not used, as we use integer times)
        idx = jnp.round(t0).astype(int)
        ddrift1 = args["ddrift1"][idx]
        ddrift2 = args["ddrift2"][idx]
        alpha = args["alpha"][idx]
        beta = args["beta"][idx]

        # Extract terms
        term_drift, term_kick = terms

        # Extract positions and momenta
        x0, p0 = y0
        dtype = x0.dtype

        # Drift 1 (0.0: dummy, as time is not used)
        x_mid = (x0 ** ω + term_drift.vf_prod(0.0, p0, args, ddrift1) ** ω).ω

        # Kick
        p1 = (alpha * p0 ** ω + beta * term_kick.vf_prod(0.0, x_mid, args, 1.0).astype(dtype) ** ω).ω

        # Drift 2
        x1 = (x_mid ** ω + term_drift.vf_prod(0.0, p1, args, ddrift2) ** ω).ω

        # Assemble solution
        y1 = (x1, p1)
        dense_info = dict(y0=y0, y1=y1) if self.use_diffrax else None
        return y1, None, dense_info, None, RESULTS.successful


class DriftKickDriftAdjoint(DiscoStepper):
    """Adjoint integrator for drift-kick-drift leapfrog Verlet.
    """
    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState_DKD,
        made_jump: BoolScalarLike,
    ) -> Tuple[Y, _ErrorEstimate_DKD, DenseInfo | None, _SolverState_DKD, RESULTS]:
        del solver_state, made_jump

        # Map t0 to index (t1 is not used, as we use integer times)
        idx = -jnp.round(t0).astype(int) - 1  # t0 is negative in the backward pass and runs from -n_steps to -1
        ddrift1 = args["ddrift1"][idx]
        ddrift2 = args["ddrift2"][idx]
        alpha = args["alpha"][idx]
        beta = args["beta"][idx]

        # Extract terms
        term_drift, term_kick = terms

        # Extract the primals and adjoints
        y, a_y, a_diff_args, a_diff_terms = y0

        x1, p1 = y[0], y[1]
        xi1, pi1 = a_y[0], a_y[1]

        # Invert drift 2
        x_mid = x1 - ddrift2 * p1
        pi_mid = pi1 + ddrift2 * xi1
        a_diff_args["ddrift2"] = set_indices_to_val(a_diff_args["ddrift2"], idx, jnp.sum(xi1 * p1))

        # Invert kick
        # Diffrax wraps the vector field several times
        vector_field = term_kick.term.term.term.vector_field if self.use_diffrax else term_kick.vector_field
        A, A_vjp = jax.vjp(lambda x_: vector_field(0.0, x_, {}), x_mid)  # 0.0: acc is not time-dependent

        dA_pi_mid = A_vjp(pi_mid)[0]
        p0 = (p1 - beta * A) / alpha
        xi0 = xi1 + beta * dA_pi_mid
        a_diff_args["alpha"] = set_indices_to_val(a_diff_args["alpha"], idx, jnp.sum(pi_mid * p0))
        a_diff_args["beta"] = set_indices_to_val(a_diff_args["beta"], idx, jnp.sum(pi_mid * A))
        pi0 = alpha * pi_mid

        # Invert drift 1
        x0 = x_mid - ddrift1 * p0
        pi0 = pi0 + ddrift1 * xi0
        a_diff_args["ddrift1"] = set_indices_to_val(a_diff_args["ddrift1"], idx, jnp.sum(xi0 * p0))

        y_out = ((x0, p0),
                 (xi0, pi0),
                 a_diff_args,
                 a_diff_terms)  # no derivatives w.r.t. terms

        dense_info = dict(y0=y0, y1=y_out) if self.use_diffrax else None
        return y_out, None, dense_info, None, RESULTS.successful
