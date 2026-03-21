import time

import jax
import jax.numpy as jnp
from rich.progress import track

from optymus.methods.utils import BaseOptimizer


class OptimalityCriteria(BaseOptimizer):
    r"""Optimality Criteria (OC) Method

    The Optimality Criteria method is a constrained optimization algorithm
    that updates design variables based on the ratio of objective and
    constraint sensitivities, with a Lagrange multiplier found via bisection.

    It solves problems of the form:

    .. math::
        \min_x \; f(x) \quad \text{s.t.} \quad g(x) \leq 0, \;
        x_{\min} \leq x \leq x_{\max}

    The update rule is derived from the KKT conditions:

    .. math::
        x_i^{new} = x_i \, B_i^{\eta}

    where :math:`B_i = \frac{|\partial f / \partial x_i|}
    {\lambda \, |\partial g / \partial x_i|}` and :math:`\lambda` is a
    Lagrange multiplier found via bisection to satisfy :math:`g(x) = 0`.

    Move limits are applied for stability:

    .. math::
        \max(x_i - m, \, x_{\min}) \leq x_i^{new}
        \leq \min(x_i + m, \, x_{\max})

    Parameters
    ----------
    f_obj : callable
        Objective function to minimize
    f_cons : list of callable
        Single inequality constraint function g(x) <= 0
    x0 : ndarray
        Initial guess
    bounds : list of tuples
        Variable bounds [(lo, hi), ...], required
    tol : float
        Tolerance for density change convergence
    max_iter : int
        Maximum number of iterations
    verbose : bool
        If True, display progress bar

    Returns
    -------
    method_name : str
        Method name
    xopt : ndarray
        Optimal point
    fmin : float
        Minimum value
    num_iter : int
        Number of iterations
    path : ndarray
        Path taken
    f_history : ndarray
        Function values per iteration
    grad_norms : ndarray
        Gradient norms per iteration
    termination_reason : str
        Why the optimizer stopped
    """

    def optimize(self, move=0.2, eta=0.5):
        start_time = time.time()

        if self._lower_bounds is None or self._upper_bounds is None:
            msg = "Optimality Criteria method requires bounds. Provide bounds as [(lo, hi), ...]."
            raise ValueError(msg)

        if self.f_cons is None or len(self.f_cons) == 0:
            msg = "Optimality Criteria method requires exactly one inequality constraint function via f_cons=[g], where g(x) <= 0."
            raise ValueError(msg)

        g_con = self.f_cons[0]

        x = self.x0.astype(float)
        lower = self._lower_bounds
        upper = self._upper_bounds

        grad_f = jax.grad(lambda x_: self.f_obj(x_, *self.args))
        grad_g = jax.grad(lambda x_: g_con(x_, *self.args_cons))

        path = [x]
        f_history = [float(self.f_obj(x, *self.args))]
        grad_norms = []
        num_iter = 0
        termination_reason = "max_iter_reached"

        progress_bar = (
            track(range(self.max_iter), description="Optimality Criteria")
            if self.verbose
            else range(self.max_iter)
        )

        for _ in progress_bar:
            df_dx = grad_f(x)
            dg_dx = grad_g(x)

            grad_norms.append(float(jnp.linalg.norm(df_dx)))

            # Bisection to find Lagrange multiplier lambda
            l1, l2 = 1e-9, 1e9
            dg_safe = jnp.where(jnp.abs(dg_dx) > 1e-12, dg_dx, 1e-12)

            def _oc_update(lam):
                B = -df_dx / (lam * dg_safe)
                B = jnp.maximum(B, 1e-12)
                xn = x * jnp.power(B, eta)
                xn = jnp.maximum(x - move, jnp.minimum(x + move, xn))
                return jnp.clip(xn, lower, upper)

            # Evaluate lower endpoint to establish sign reference for bisection
            g_l1 = float(g_con(_oc_update(l1), *self.args_cons))

            for _ in range(100):
                if l2 / (l1 + 1e-30) < 1.0 + 1e-4:
                    break

                lmid = jnp.sqrt(l1 * l2)
                x_new = _oc_update(lmid)
                g_val = float(g_con(x_new, *self.args_cons))

                # Standard bisection: keep l1 on the same side as g_l1
                if (g_val > 0) == (g_l1 > 0):
                    l1 = lmid
                    g_l1 = g_val
                else:
                    l2 = lmid

            x_new = _oc_update(jnp.sqrt(l1 * l2))

            # Convergence check on density change
            change = float(jnp.max(jnp.abs(x_new - x)))
            x = x_new

            path.append(x)
            f_history.append(float(self.f_obj(x, *self.args)))
            num_iter += 1

            if change < self.tol and num_iter > 1:
                termination_reason = "density_change_below_tol"
                break

        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "method_name": "Optimality Criteria (OC)",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "f_history": jnp.array(f_history),
            "grad_norms": jnp.array(grad_norms),
            "termination_reason": termination_reason,
            "time": elapsed_time,
        }


def oc(move=0.2, eta=0.5, **kwargs):
    """Optimality Criteria (OC) optimization method.

    A constrained optimization method that finds the optimal design
    by iteratively updating variables based on the ratio of objective
    and constraint sensitivities, with a Lagrange multiplier found
    via bisection.

    Solves: min f(x) s.t. g(x) <= 0, x_lower <= x <= x_upper

    Parameters
    ----------
    move : float
        Move limit for stability (default: 0.2)
    eta : float
        Damping exponent for OC update (default: 0.5)
    **kwargs
        Arguments passed to BaseOptimizer
        (f_obj, f_cons, x0, bounds, tol, max_iter, verbose, etc.)

    Returns
    -------
    dict
        Optimization results
    """
    optimizer = OptimalityCriteria(**kwargs)
    return optimizer.optimize(move=move, eta=eta)
