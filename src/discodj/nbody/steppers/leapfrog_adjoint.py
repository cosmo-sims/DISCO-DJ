# This file is a customized version of the original Diffrax _adjoint.py file
# https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_adjoint.py.
from typing import Any
import equinox as eqx
from diffrax import AbstractAdjoint
from diffrax._saveat import save_y, SubSaveAt
from diffrax._adjoint import _only_transpose_ys, _nondiff_solver_controller_state, _loop_backsolve
import jax.tree_util as jtu

__all__ = ["BacksolveAdjointLeapfrog"]


def _is_none(x):
    return x is None


def _is_subsaveat(x: Any) -> bool:
    return isinstance(x, SubSaveAt)


class BacksolveAdjointLeapfrog(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous
    adjoint equations backwards-in-time, specifically for the leapfrog DKD/KDK integrator. 
    This is also sometimes known as "optimise-then-discretise", the "continuous adjoint method" 
    or simply the "adjoint method".
    Modified from https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_adjoint.py.

    This will compute gradients with respect to the `terms`, `y0` and `args` arguments
    passed to [`diffrax.diffeqsolve`][]. If you attempt to compute gradients with
    respect to anything else (for example `t0`, or arguments passed via closure), then
    a `CustomVJPException` will be raised by JAX. See also
    [this FAQ](../../further_details/faq/#im-getting-a-customvjpexception)
    entry.

    !!! info

        Using this method prevents computing forward-mode autoderivatives of
        [`diffrax.diffeqsolve`][]. (That is to say, `jax.jvp` will not work.)
    """  # noqa: E501

    kwargs: dict[str, Any]

    def __init__(self, **kwargs):
        """
        **Arguments:**

        - `**kwargs`: The arguments for the [`diffrax.diffeqsolve`][] operations that
            are called on the backward pass. For example use
            ```python
            BacksolveAdjointLeapfrog(solver=...)
            ```
            to specify a particular solver to use on the backward pass.
        """
        valid_keys = {
            "dt0",
            "solver",
            "stepsize_controller",
            "adjoint",
            "max_steps",
            "throw",
        }
        given_keys = set(kwargs.keys())
        diff_keys = given_keys - valid_keys
        if len(diff_keys) > 0:
            raise ValueError(
                "The following keyword argments are not valid for `BacksolveAdjointLeapfrog`: "
                f"{diff_keys}"
            )
        self.kwargs = kwargs

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        init_state,
        passed_solver_state,
        passed_controller_state,
        event,
        **kwargs,
    ):
        if jtu.tree_structure(saveat.subs, is_leaf=_is_subsaveat) != jtu.tree_structure(
            0
        ):
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjointLeapfrog()` with `SaveAt(subs=...)`."
            )
        if saveat.dense or saveat.subs.steps:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjointLeapfrog()` with "
                "`saveat=SaveAt(steps=True)` or saveat=SaveAt(dense=True)`."
            )
        if saveat.subs.fn is not save_y:
            raise NotImplementedError(
                "Cannot use `adjoint=BacksolveAdjointLeapfrog()` with `saveat=SaveAt(fn=...)`."
            )
        if event is not None:
            raise NotImplementedError(
                "`diffrax.BacksolveAdjointLeapfrog` is not compatible with events."
            )

        y = init_state.y
        init_state = eqx.tree_at(lambda s: s.y, init_state, object())
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )

        final_state, aux_stats = _loop_backsolve(
            (y, args, terms),
            self=self,
            saveat=saveat,
            init_state=init_state,
            solver=solver,
            event=event,
            **kwargs,
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats
