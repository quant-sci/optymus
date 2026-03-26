import time

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.methods.utils._result import OptimizeResult


class Univariate(BaseOptimizer):
    r"""Univariate Search Method

    In the Univariate Method, the search direction at iteration :math:`k` is defined by:

    .. math::
        \mathbf{d}_k = \mathbf{e}_k, \quad k = 1, \ldots, n

    where :math:`\mathbf{e}_k` is a vector with zero elements except at position :math:`k`, where the element is 1.

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
        Lerning rate for line search
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
        Lerning rate for line searchs
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)

        n = len(x)
        u = jnp.identity(n)
        path = [x]
        alphas = []
        f_history = [float(self.penalized_obj(x))]
        grad_norms = []
        num_iter = 0
        termination_reason = "max_iter_reached"

        progres_bar = tqdm(range(self.max_iter), desc="Univariate", disable=not self.verbose)

        for _ in progres_bar:
            g_norm = float(jnp.linalg.norm(jax.grad(self.penalized_obj)(x)))
            grad_norms.append(g_norm)
            if g_norm < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break
            for i in range(n):
                v = u[i]
                r = self._do_line_search(x, v)
                x = self.project(r["xopt"])
                alphas.append(r["alpha"])
                path.append(x)
            f_history.append(float(self.penalized_obj(x)))
            num_iter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        return OptimizeResult({
            "method_name": "Univariant" if not self.f_cons else "Univariant with Penalty",
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "f_history": jnp.array(f_history),
            "grad_norms": jnp.array(grad_norms),
            "termination_reason": termination_reason,
            "time": elapsed_time,
        })


def univariate(**kwargs):
    optimizer = Univariate(**kwargs)
    return optimizer.optimize()
