import time

import jax
import jax.numpy as jnp
from rich.progress import track

from optymus.methods.utils import BaseOptimizer


class SteepestDescent(BaseOptimizer):
    _default_line_search = "armijo"
    r"""Steepest Descent

    Steepest Descent is a first-order optimization algorithm that uses the
    gradient of the objective function to compute the step direction.

    We can minimize the objective function :math:`f` by solving the following
    equation:

    .. math::
        x_{k+1} = x_k - \alpha \nabla f(x_k)

    where :math:`x_k` is the current point, :math:`\alpha` is the step size,
    and :math:`\nabla f(x_k)` is the gradient of :math:`f` evaluated at point
    :math:`x_k`.

    Parameters
    ----------
    f_obj : callable
        Objective function to minimize
    f_cons : callable
        Constraint function
    args : tuple
        Arguments for the objective function
    args_cons : tuple
        Arguments for the constraint function
    x0 : ndarray
        Initial guess
    tol : float
        Tolerance for stopping criteria
    learning_rate : float
        Step size
    max_iter : int
        Maximum number of iterations
    maximize : bool
        If True, maximize the objective function

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
    alphas : ndarray
        Step sizes
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)(x)
        d = -grad
        path = [x]
        alphas = []
        f_history = [float(self.penalized_obj(x))]
        grad_norms = [float(jnp.linalg.norm(grad))]
        num_iter = 0
        termination_reason = "max_iter_reached"

        progres_bar = (
            track(
                range(self.max_iter),
                description=f"Steepest Descent {num_iter}",
            )
            if self.verbose
            else range(self.max_iter)
        )

        for _ in progres_bar:
            if jnp.linalg.norm(grad) < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break
            r = self._do_line_search(x, d, grad)
            x = self.project(r["xopt"].astype(float))
            grad = jax.grad(self.penalized_obj)(x)
            d = -grad
            path.append(x)
            alphas.append(r["alpha"])
            f_history.append(float(self.penalized_obj(x)))
            grad_norms.append(float(jnp.linalg.norm(grad)))
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "method_name": "Steepest Descent" if not self.f_cons else "Steepest Descent with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "f_history": jnp.array(f_history),
            "grad_norms": jnp.array(grad_norms),
            "termination_reason": termination_reason,
            "time": elapsed_time,
        }


def steepest_descent(**kwargs):
    optimizer = SteepestDescent(**kwargs)
    return optimizer.optimize()
