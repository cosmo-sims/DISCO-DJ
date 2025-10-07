import jax
import jax.numpy as jnp
from diffrax import AbstractTerm, AbstractSolver, ODETerm
from functools import partial

__all__ = ["nbody_scan"]


def scan_body(fn, c, x_, collect=False):
    t0, t1 = x_
    c = fn(y0=c, t0=t0, t1=t1)[0]
    return c, c if collect else None


def nbody_scan(args: dict, init: tuple, terms: tuple, n_steps: int, adjoint_method: bool, collect_all: bool,
               stepper_fwd: AbstractSolver, stepper_bwd: AbstractTerm | None = None) -> tuple:
    """Wrapper for scanning over the timesteps when not using Diffrax.

    :param terms: ODE terms (drift, kick)
    :param args: arguments for the integrator
    :param init: initial conditions (positions, momenta)
    :param n_steps: number of steps
    :param adjoint_method: use adjoint method
    :param collect_all: collect all outputs?
    :param stepper_fwd: integrator for forward pass
    :param stepper_bwd: integrator for backward pass (only needed if adjoint_method=True)
    :return: tuple of positions and momenta, at all times (collect_all=True) or only at final time (collect_all=False)
    """
    if adjoint_method:
        assert not collect_all, "Adjoint method does not support collect_all=True"
        assert stepper_bwd is not None, "Adjoint method requires stepper_bwd"
        # Wrap lambda functions inside jax.tree_util.Partial to make them valid inputs for jitted function
        non_diff_args = (terms, n_steps, stepper_fwd, stepper_bwd)
        return scan_wrapper(non_diff_args, args, init)  # diff'able arguments last
    else:
        return scan_wrapper_no_custom_vjp(args, init, terms, n_steps, stepper_fwd, collect_all)


def scan_wrapper_no_custom_vjp(args, init, terms, n_steps, stepper_fwd, collect_all):
    step_fct = partial(stepper_fwd.step, terms=terms, args=args, solver_state=None, made_jump=None)
    carry = init
    dtype = init[0].dtype
    xs = (jnp.arange(n_steps, dtype=dtype), jnp.arange(1, n_steps + 1, dtype=dtype))
    scan_fn = partial(scan_body, step_fct, collect=collect_all)
    carry, ys = jax.lax.scan(scan_fn, carry, xs)
    if collect_all:
        return ys
    else:
        return jax.tree_util.tree_map(lambda x: x[None, ...], carry)  # add leading dimension for consistency


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def scan_wrapper(non_diff_args, args, init):
    terms, n_steps, stepper_fwd, stepper_bwd = non_diff_args
    step_fct = partial(stepper_fwd.step, terms=terms, args=args, solver_state=None, made_jump=None)
    carry = init
    dtype = init[0].dtype
    xs = (jnp.arange(n_steps, dtype=dtype), jnp.arange(1, n_steps + 1, dtype=dtype))
    scan_fn = partial(scan_body, step_fct)
    carry, ys = jax.lax.scan(scan_fn, carry, xs)
    return jax.tree_util.tree_map(lambda x: x[None, ...], carry)  # add leading dimension for consistency

def scan_wrapper_fwd(non_diff_args, args, init):
    result = scan_wrapper(non_diff_args, args, init)
    return result, (result,) + (args,)

def scan_wrapper_bwd(non_diff_args, primals, g):
    terms, n_steps, stepper_fwd, stepper_bwd = non_diff_args
    result, args = primals
    step_fct_adjoint = partial(stepper_bwd.step, terms=terms, args=args, solver_state=None, made_jump=None)
    init_for_bwd = (result[0][0, ...], result[1][0, ...])  # initialize adjoint eq. with the final state
    dtype = init_for_bwd[0].dtype
    zeros_like_args = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), args)
    zeros_like_terms = (ODETerm(None), ODETerm(None))
    carry = (init_for_bwd, (g[0][0, :, :], g[1][0, :, :]), zeros_like_args, zeros_like_terms)
    # iterate from final to initial time (step beginnings: -n_steps to -1, step ends: -n_steps + 1 to 0)
    xs = (jnp.arange(-n_steps, 0, dtype=dtype), jnp.arange(-n_steps + 1, 1, dtype=dtype))
    scan_fn = partial(scan_body, step_fct_adjoint)
    # carry: 1. (Psi, Mom) at initial time -> discard, 2. adjoints for (Psi, Mom) at initial time -> return for init
    # 3. adjoints for args -> return for args, 4. adjoints for terms -> discard
    carry, ys = jax.lax.scan(scan_fn, carry, xs)
    return carry[2], carry[1]

scan_wrapper.defvjp(scan_wrapper_fwd, scan_wrapper_bwd)
